"""
VisRAG Baseline
===============
Implements visual retrieval baselines for image/document-page retrieval.

Supported retrieval modes
-------------------------
1. bm25 / dense
   Legacy caption-description retrieval over "{title}. {generated_caption}".
2. visrag_ret
   Official VisRAG-Ret style retrieval using the released
   `openbmb/VisRAG-Ret` checkpoint directly on page images.
"""

import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Region describer — LLaVA-7B generates a text description of each region
# ---------------------------------------------------------------------------

class RegionDescriber:
    """
    Uses LLaVA-v1.5-7b to produce a short text description for each image region.
    The description is: "{title}. {generated_caption}"
    """

    CAPTION_PROMPT = "USER: <image>\nDescribe this image region briefly.\nASSISTANT:"

    def __init__(self, model_name: str = "llava-hf/llava-1.5-7b-hf", device: str = "cuda"):
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        logger.info(f"Loading region describer: {model_name}")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map=device
        )
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def describe(
        self,
        regions: List[Image.Image],
        titles: List[str],
        max_new_tokens: int = 64,
    ) -> List[str]:
        """
        Generate text descriptions for a list of image regions.

        Returns
        -------
        descriptions : list of str, one per region  ("{title}. {caption}")
        """
        descriptions = []
        for region, title in zip(regions, titles):
            enc = self.processor(
                text=self.CAPTION_PROMPT,
                images=region,
                return_tensors="pt",
            ).to(self.device)
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            # Decode only newly generated tokens
            n_input = enc["input_ids"].shape[-1]
            caption = self.processor.tokenizer.decode(
                out[0][n_input:], skip_special_tokens=True
            ).strip()
            descriptions.append(f"{title}. {caption}" if title else caption)
        return descriptions


# ---------------------------------------------------------------------------
# VisRAG retriever — BM25 or dense over region descriptions
# ---------------------------------------------------------------------------

class VisRAGRetriever:
    """
    Retriever over pre-computed region descriptions.
    Supports BM25 (rank_bm25) and dense (sentence-transformers).
    """

    def __init__(self, method: str = "dense", dense_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.method = method
        if method == "dense":
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(dense_model)
        elif method == "bm25":
            try:
                from rank_bm25 import BM25Okapi
                self.BM25Okapi = BM25Okapi
            except ImportError:
                raise ImportError("pip install rank_bm25")
        else:
            raise ValueError(f"Unknown retrieval method: {method}")

        self.descriptions: List[str] = []
        self.doc_ids: List[str]      = []
        self._index = None

    def build(self, doc_ids: List[str], descriptions: List[str]):
        self.doc_ids      = doc_ids
        self.descriptions = descriptions
        if self.method == "dense":
            self._index = self.encoder.encode(
                descriptions,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        elif self.method == "bm25":
            tokenized = [d.lower().split() for d in descriptions]
            self._index = self.BM25Okapi(tokenized)
        logger.info(f"VisRAG index built: {len(descriptions)} regions ({self.method})")

    def retrieve(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """Returns list of (doc_id, score) sorted by descending relevance."""
        if self.method == "dense":
            q_emb = self.encoder.encode(
                [query],
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            scores = (self._index @ q_emb.T).squeeze(-1).cpu().tolist()
        else:
            scores = self._index.get_scores(query.lower().split()).tolist()

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(self.doc_ids[i], s) for i, s in ranked[:top_k]]


class OfficialVisRAGRetriever:
    """
    Official VisRAG-Ret retriever backed by `openbmb/VisRAG-Ret`.
    """

    QUERY_INSTRUCTION = "Represent this query for retrieving relevant documents: "

    def __init__(
        self,
        model_name: str = "openbmb/VisRAG-Ret",
        device: str = "cuda",
        batch_size: int = 8,
    ):
        # VisRAG-Ret custom code expects an older transformers symbol.
        try:
            from transformers.utils import import_utils as _tf_import_utils
            if not hasattr(_tf_import_utils, "is_torch_fx_available"):
                _tf_import_utils.is_torch_fx_available = lambda: False
        except Exception:
            pass

        from transformers import AutoModel, AutoTokenizer

        logger.info(f"Loading official VisRAG retriever: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModel.from_pretrained(
            model_name,
            dtype=dtype,
            trust_remote_code=True,
        ).to(device)
        self.model.eval()
        self.device = device
        self.batch_size = batch_size
        self.doc_ids: List[str] = []
        self._index = None

    @staticmethod
    def _weighted_mean_pooling(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        attention_mask_ = attention_mask * attention_mask.cumsum(dim=1)
        numer = torch.sum(hidden * attention_mask_.unsqueeze(-1).float(), dim=1)
        denom = attention_mask_.sum(dim=1, keepdim=True).float()
        return numer / denom

    @torch.no_grad()
    def _encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        import torch.nn.functional as F

        all_embs = []
        texts = [self.QUERY_INSTRUCTION + t for t in texts]
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            inputs = {
                "text": batch,
                "image": [None] * len(batch),
                "tokenizer": self.tokenizer,
            }
            outputs = self.model(**inputs)
            reps = self._weighted_mean_pooling(outputs.last_hidden_state, outputs.attention_mask)
            all_embs.append(F.normalize(reps.float(), p=2, dim=1).cpu())
        return torch.cat(all_embs, dim=0)

    @torch.no_grad()
    def build_image_index(self, doc_ids: Sequence[str], images: Sequence[Image.Image]) -> None:
        import torch.nn.functional as F

        all_embs = []
        self.doc_ids = list(doc_ids)
        for start in range(0, len(images), self.batch_size):
            batch = list(images[start:start + self.batch_size])
            inputs = {
                "text": [""] * len(batch),
                "image": batch,
                "tokenizer": self.tokenizer,
            }
            outputs = self.model(**inputs)
            reps = self._weighted_mean_pooling(outputs.last_hidden_state, outputs.attention_mask)
            all_embs.append(F.normalize(reps.float(), p=2, dim=1).cpu())
        self._index = torch.cat(all_embs, dim=0)
        logger.info("Official VisRAG index built: %d pages/images", len(self.doc_ids))

    @torch.no_grad()
    def retrieve(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        q_emb = self._encode_texts([query])[0:1]
        scores = (self._index @ q_emb.T).squeeze(-1).cpu().tolist()
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(self.doc_ids[i], s) for i, s in ranked[:top_k]]


# ---------------------------------------------------------------------------
# VisRAG answer generator — LLaVA-13B sees top-k regions as images
# ---------------------------------------------------------------------------

class VisRAGGenerator:
    """
    LLaVA-v1.5-13b generates the final answer given the query and top-k image regions.
    Each region is passed as a separate <image> token.
    """

    def __init__(self, model_name: str = "llava-hf/llava-1.5-13b-hf", device: str = "cuda"):
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        logger.info(f"Loading VisRAG generator: {model_name}")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map=device
        )
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def answer(
        self,
        question: str,
        regions: List[Image.Image],
        titles: List[str],
        max_new_tokens: int = 64,
    ) -> str:
        """
        Generate an answer given a question and a list of retrieved image regions.
        """
        if not regions:
            return ""

        # Build prompt with one <image> per region
        img_slots = "".join([f"Region {i+1} (Title: {t}): <image>\n"
                             for i, t in enumerate(titles)])
        prompt = f"USER: {img_slots}Question: {question}\nASSISTANT:"

        enc = self.processor(
            text=prompt,
            images=regions,
            return_tensors="pt",
        ).to(self.device)

        out = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        n_input = enc["input_ids"].shape[-1]
        return self.processor.tokenizer.decode(
            out[0][n_input:], skip_special_tokens=True
        ).strip()


# ---------------------------------------------------------------------------
# Full VisRAG pipeline
# ---------------------------------------------------------------------------

class VisRAGBaseline:
    """
    End-to-end VisRAG baseline.

    Evaluation flow per sample:
      1. Describe each candidate region with LLaVA-7b → text index.
      2. Retrieve top-k regions by query similarity (BM25 or dense).
      3. Pass top-k regions as images to LLaVA-13b → answer.
      4. Evaluate with EM / F1 at each top_k in top_k_values.

    Parameters
    ----------
    compressor_model : LLaVA-7b for region description
    generator_model  : LLaVA-13b for answer generation
    top_k_values     : list of k values to evaluate
    retrieval_method : "dense" or "bm25"
    device           : torch device string
    """

    def __init__(
        self,
        compressor_model: str = "llava-hf/llava-1.5-7b-hf",
        generator_model:  str = "llava-hf/llava-1.5-13b-hf",
        top_k_values: List[int] = None,
        retrieval_method: str = "dense",
        device: str = "cuda",
        visrag_ret_model: str = "openbmb/VisRAG-Ret",
        visrag_ret_batch_size: int = 8,
    ):
        if top_k_values is None:
            top_k_values = [1, 2, 5, 10]
        self.top_k_values = top_k_values

        self.describer = None
        if retrieval_method == "visrag_ret":
            self.retriever = OfficialVisRAGRetriever(
                model_name=visrag_ret_model,
                device=device,
                batch_size=visrag_ret_batch_size,
            )
        else:
            self.describer = RegionDescriber(compressor_model, device)
            self.retriever = VisRAGRetriever(method=retrieval_method)
        self.generator  = VisRAGGenerator(generator_model, device)
        self._region_store = {}

    def build_index(
        self,
        samples: List[Dict],
        image_dir: str,
        scale: float = 1 / 8,
        patch_size: int = 24,
        patches_per_side: int = 3,
    ):
        """
        Legacy WebQA path: build an index over image regions from local files.
        """
        from src.region_encoder import image_to_regions, region_doc_id

        doc_ids:      List[str]         = []
        descriptions: List[str]         = []
        region_store: Dict[str, Tuple]  = {}  # doc_id → (PIL region, title)

        seen_img_ids = set()

        for s in samples:
            all_ids  = s.get("pos_image_ids", []) + s.get("neg_image_ids", [])
            all_caps = s.get("pos_captions",  []) + s.get("neg_captions",  [])
            img_dir  = s.get("image_dir", image_dir)

            for img_id, title in zip(all_ids, all_caps):
                if img_id in seen_img_ids:
                    continue
                seen_img_ids.add(img_id)

                path = os.path.join(img_dir, f"{img_id}.jpg")
                if not os.path.exists(path):
                    continue
                try:
                    img = Image.open(path).convert("RGB")
                except Exception:
                    continue

                regions = image_to_regions(img, scale, patch_size, patches_per_side)
                pil_list  = [r for r, _, _ in regions]
                title_rep = [title] * len(pil_list)

                descs = self.describer.describe(pil_list, title_rep)

                for (region_pil, row, col), desc in zip(regions, descs):
                    did = region_doc_id(img_id, row, col)
                    doc_ids.append(did)
                    descriptions.append(desc)
                    region_store[did] = (region_pil, title)

        self.retriever.build(doc_ids, descriptions)
        self._region_store = region_store
        logger.info(f"VisRAG index: {len(doc_ids)} regions from {len(seen_img_ids)} images")

    def build_candidate_index(
        self,
        candidate_sets: Sequence[Sequence[Dict]],
    ) -> None:
        """
        Build a VisRAG index over full candidate images/pages already loaded in memory.

        Each candidate dict must include:
          - image : PIL image
          - title : display title
          - id    : unique id within the corpus
        """
        doc_ids: List[str] = []
        descriptions: List[str] = []
        images: List[Image.Image] = []
        region_store: Dict[str, Tuple] = {}
        seen_ids = set()

        for candidates in candidate_sets:
            for cand in candidates:
                doc_id = str(cand.get("id") or "")
                if not doc_id or doc_id in seen_ids:
                    continue
                image = cand.get("image")
                if image is None:
                    continue
                seen_ids.add(doc_id)
                title = str(cand.get("title") or "")
                doc_ids.append(doc_id)
                region_store[doc_id] = (image, title)
                images.append(image)
                if self.describer is not None:
                    desc = self.describer.describe([image], [title])[0]
                    descriptions.append(desc)

        if isinstance(self.retriever, OfficialVisRAGRetriever):
            self.retriever.build_image_index(doc_ids, images)
        else:
            self.retriever.build(doc_ids, descriptions)
        self._region_store = region_store
        logger.info("VisRAG index: %d pages/images", len(doc_ids))

    def run_sample(
        self,
        question: str,
        answers: List[str],
        max_new_tokens: int = 64,
        positive_doc_ids: Optional[Sequence[str]] = None,
    ) -> Dict[int, Dict]:
        """
        Run evaluation for a single question at all top_k_values.

        Returns
        -------
        dict: top_k → {"prediction": str, "em": float, "f1": float}
        """
        from src.evaluation import exact_match, token_f1

        results = {}
        for k in self.top_k_values:
            retrieved = self.retriever.retrieve(question, top_k=k)
            regions = []
            titles  = []
            retrieved_doc_ids = []
            for did, _ in retrieved:
                if did in self._region_store:
                    pil, title = self._region_store[did]
                    regions.append(pil)
                    titles.append(title)
                    retrieved_doc_ids.append(did)

            pred = self.generator.answer(question, regions, titles, max_new_tokens)
            em = exact_match(pred, answers)
            f1 = token_f1(pred, answers)
            recall = 0.0
            precision = 0.0
            if positive_doc_ids:
                pos_set = set(str(x) for x in positive_doc_ids)
                ret_set = set(retrieved_doc_ids)
                hits = len(pos_set & ret_set)
                recall = hits / len(pos_set) if pos_set else 0.0
                precision = hits / len(ret_set) if ret_set else 0.0
            results[k] = {
                "prediction": pred,
                "em": em,
                "f1": f1,
                "recall_at_k": recall,
                "precision_at_k": precision,
            }

        return results

    def run(
        self,
        samples: List[Dict],
        image_dir: str,
        max_new_tokens: int = 64,
    ) -> Dict[str, float]:
        """
        Run full evaluation over a list of samples.

        Returns aggregated metrics for each k.
        """
        totals = {k: {"em": 0.0, "f1": 0.0, "recall_at_k": 0.0, "precision_at_k": 0.0, "n": 0} for k in self.top_k_values}

        for s in samples:
            res = self.run_sample(
                s["question"],
                s["answers"],
                max_new_tokens,
                positive_doc_ids=s.get("positive_doc_ids"),
            )
            for k, vals in res.items():
                totals[k]["em"] += vals["em"]
                totals[k]["f1"] += vals["f1"]
                totals[k]["recall_at_k"] += vals["recall_at_k"]
                totals[k]["precision_at_k"] += vals["precision_at_k"]
                totals[k]["n"]  += 1

        metrics = {}
        for k in self.top_k_values:
            n = max(totals[k]["n"], 1)
            metrics[f"top{k}/em"] = totals[k]["em"] / n
            metrics[f"top{k}/f1"] = totals[k]["f1"] / n
            metrics[f"top{k}/recall_at_k"] = totals[k]["recall_at_k"] / n
            metrics[f"top{k}/precision_at_k"] = totals[k]["precision_at_k"] / n
            logger.info(
                "VisRAG top%d: EM=%.4f F1=%.4f Recall=%.4f",
                k,
                metrics[f"top{k}/em"],
                metrics[f"top{k}/f1"],
                metrics[f"top{k}/recall_at_k"],
            )

        return metrics
