"""
VisRAG Baseline
===============
Implements the VisRAG (Vision-based RAG) baseline for the WebQA image retrieval task.

VisRAG approach
---------------
Instead of using compressed latent tokens, VisRAG uses the VLM directly:
  1. Each image region is described by LLaVA-v1.5-7b as a short text snippet
     (title + VLM-generated description).
  2. At query time, query text is matched against region descriptions via
     BM25 (sparse) or dense retrieval (sentence-transformers).
  3. Top-k retrieved regions are passed as raw images to LLaVA-v1.5-13b,
     which generates the answer.

Evaluated at top_k ∈ {1, 2, 5, 10}.

Usage
-----
from src.visrag_baseline import VisRAGBaseline

baseline = VisRAGBaseline(
    compressor_model="llava-hf/llava-1.5-7b-hf",
    generator_model="llava-hf/llava-1.5-13b-hf",
    top_k_values=[1, 2, 5, 10],
    device="cuda",
)
results = baseline.run(samples, image_dir)
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

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
                descriptions, convert_to_tensor=True, normalize_embeddings=True
            )
        elif self.method == "bm25":
            tokenized = [d.lower().split() for d in descriptions]
            self._index = self.BM25Okapi(tokenized)
        logger.info(f"VisRAG index built: {len(descriptions)} regions ({self.method})")

    def retrieve(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """Returns list of (doc_id, score) sorted by descending relevance."""
        if self.method == "dense":
            q_emb = self.encoder.encode(
                [query], convert_to_tensor=True, normalize_embeddings=True
            )
            scores = (self._index @ q_emb.T).squeeze(-1).cpu().tolist()
        else:
            scores = self._index.get_scores(query.lower().split()).tolist()

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
    ):
        if top_k_values is None:
            top_k_values = [1, 2, 5, 10]
        self.top_k_values = top_k_values

        self.describer  = RegionDescriber(compressor_model, device)
        self.retriever  = VisRAGRetriever(method=retrieval_method)
        self.generator  = VisRAGGenerator(generator_model, device)

    def build_index(
        self,
        samples: List[Dict],
        image_dir: str,
        scale: float = 1 / 8,
        patch_size: int = 24,
        patches_per_side: int = 3,
    ):
        """
        Build the region description index from a list of WebQA samples.
        Processes ALL pos + neg images from each sample.
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

    def run_sample(
        self,
        question: str,
        answers: List[str],
        max_new_tokens: int = 64,
    ) -> Dict[str, Dict]:
        """
        Run evaluation for a single question at all top_k_values.

        Returns
        -------
        dict: top_k → {"prediction": str, "em": float, "f1": float}
        """
        from src.evaluation import compute_f1, compute_em

        results = {}
        for k in self.top_k_values:
            retrieved = self.retriever.retrieve(question, top_k=k)
            regions = []
            titles  = []
            for did, _ in retrieved:
                if did in self._region_store:
                    pil, title = self._region_store[did]
                    regions.append(pil)
                    titles.append(title)

            pred = self.generator.answer(question, regions, titles, max_new_tokens)
            em   = compute_em(pred, answers)
            f1   = compute_f1(pred, answers)
            results[k] = {"prediction": pred, "em": em, "f1": f1}

        return results

    def run(
        self,
        samples: List[Dict],
        image_dir: str,
        max_new_tokens: int = 64,
    ) -> Dict[str, float]:
        """
        Run full evaluation over a list of samples.

        Returns
        -------
        metrics: {"top{k}/em": float, "top{k}/f1": float, ...} for each k
        """
        totals = {k: {"em": 0.0, "f1": 0.0, "n": 0} for k in self.top_k_values}

        for s in samples:
            res = self.run_sample(
                s["question"], s["answers"], max_new_tokens
            )
            for k, vals in res.items():
                totals[k]["em"] += vals["em"]
                totals[k]["f1"] += vals["f1"]
                totals[k]["n"]  += 1

        metrics = {}
        for k in self.top_k_values:
            n = max(totals[k]["n"], 1)
            metrics[f"top{k}/em"] = totals[k]["em"] / n
            metrics[f"top{k}/f1"] = totals[k]["f1"] / n
            logger.info(f"VisRAG top{k}: EM={metrics[f'top{k}/em']:.4f}  F1={metrics[f'top{k}/f1']:.4f}")

        return metrics
