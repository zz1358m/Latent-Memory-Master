"""
Evaluation Metrics
==================
Standard QA evaluation metrics used in the experiments:

  - Exact Match (EM)  : 1 if prediction matches any reference after normalization
  - Token-level F1   : F1 over shared tokens between prediction and reference
  - ROUGE-L          : Longest Common Subsequence recall/precision
  - Token efficiency : average input tokens used (lower = better for our method)

These match the evaluation protocol of LongBench, NarrativeQA, and Qasper.
"""

import re
import string
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np
from nltk.stem import PorterStemmer


_WEBQA_STEMMER = PorterStemmer()
_WEBQA_SMALL_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_WEBQA_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_WEBQA_COLOR_SET = {
    "orangebrown", "spot", "yellow", "blue", "rainbow", "ivory", "brown", "gray", "teal",
    "bluewhite", "orangepurple", "black", "white", "gold", "redorange", "pink", "blonde",
    "tan", "turquoise", "grey", "beige", "golden", "orange", "bronze", "maroon", "purple",
    "bluere", "red", "rust", "violet", "transparent", "yes", "silver", "chrome", "green", "aqua",
}
_WEBQA_SHAPE_SET = {
    "globular", "octogon", "ring", "hoop", "octagon", "concave", "flat", "wavy", "shamrock",
    "cross", "cylinder", "cylindrical", "pentagon", "point", "pyramidal", "crescent", "rectangular",
    "hook", "tube", "cone", "bell", "spiral", "ball", "convex", "square", "arch", "h", "cuboid",
    "step", "rectangle", "dot", "oval", "circle", "star", "crosse", "crest", "octagonal", "cube",
    "triangle", "semicircle", "domeshape", "obelisk", "corkscrew", "curve", "circular", "xs", "slope",
    "pyramid", "round", "bow", "straight", "triangular", "heart", "fork", "teardrop", "fold", "curl",
    "spherical", "diamond", "keyhole", "conical", "dome", "sphere", "bellshaped", "rounded", "hexagon",
    "flower", "globe", "torus",
}
_WEBQA_YESNO_SET = {"yes", "no"}
_WEBQA_CLOSED = {"color", "shape", "number", "yesno", "yes/no", "y/n", "yn"}


# ------------------------------------------------------------------
# Text normalization (standard for QA benchmarks)
# ------------------------------------------------------------------

def normalize_answer(text: str) -> str:
    """
    Lowercase, remove punctuation, articles, and extra whitespace.
    Standard normalization used in SQuAD and derived benchmarks.
    """
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text


def _webqa_to_num(word: str) -> Any:
    if word == "point":
        return word
    if word in _WEBQA_SMALL_NUMBERS:
        return _WEBQA_SMALL_NUMBERS[word]
    if word in _WEBQA_TENS:
        return _WEBQA_TENS[word]
    if word == "hundred":
        return 100
    return word


def _webqa_remove_punc(text: str) -> str:
    exclude = set(string.punctuation) - {"."}
    text1 = "".join(ch for ch in text if ch not in exclude)
    return re.sub(r"\.(?!\d)", "", text1)


def _normalize_webqa_text(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return ""

    def remove_articles(t: str) -> str:
        return re.sub(re.compile(r"\b(a|an|the)\b", re.UNICODE), " ", t)

    def white_space_fix(t: str) -> str:
        return " ".join(str(_webqa_to_num(w)) for w in t.split())

    def lower(t: str) -> str:
        return t.lower()

    def stem_tokens(t: str) -> str:
        return " ".join(_WEBQA_STEMMER.stem(tok) for tok in t.split())

    if len(s.strip()) == 1:
        return white_space_fix(lower(s))
    if len(s.strip().split()) == 1:
        return stem_tokens(white_space_fix(_webqa_remove_punc(lower(s))))
    return stem_tokens(white_space_fix(remove_articles(_webqa_remove_punc(lower(s)))))


def _normalize_webqa_qcate(qcate: Optional[str]) -> Optional[str]:
    if not qcate:
        return None
    qc_raw = (qcate or "").strip()
    qc = qc_raw.lower().replace(" ", "").replace("/", "")
    if qc in {"yesno", "yn"}:
        return "YesNo"
    if qc == "others":
        return "Others"
    if qc == "choose":
        return "choose"
    if qc == "text":
        return "text"
    if qc == "number":
        return "number"
    if qc == "color":
        return "color"
    if qc == "shape":
        return "shape"
    return qc

def _webqa_detect_num(tokens: List[str]) -> List[str]:
    result = []
    for w in tokens:
        try:
            result.append(str(int(float(w))))
        except Exception:
            pass
    return result


def _webqa_acc_single(prediction: str, ground_truth: str, qcate: str = "text") -> Dict[str, float]:
    bow_pred = _normalize_webqa_text(prediction).split()
    bow_target = _normalize_webqa_text(ground_truth).split()

    domain = {
        "color": _WEBQA_COLOR_SET,
        "shape": _WEBQA_SHAPE_SET,
        "YesNo": _WEBQA_YESNO_SET,
        "number": {"NUMBER"},
        "text": None,
        "Others": None,
        "choose": None,
    }.get(qcate, None)

    if domain == {"NUMBER"}:
        bow_pred = _webqa_detect_num(bow_pred)
        bow_target = _webqa_detect_num(bow_target)
    elif domain is not None:
        bow_pred = list(domain.intersection(bow_pred))
        bow_target = list(domain.intersection(bow_target))

    if bow_pred == bow_target:
        em = 1.0
    else:
        em = 0.0

    common = Counter(bow_target) & Counter(bow_pred)
    num_same = sum(common.values())
    if num_same == 0:
        return {"f1": 0.0, "recall": 0.0, "precision": 0.0, "em": em, "acc": 0.0}

    precision = num_same / max(len(bow_pred), 1)
    recall = num_same / max(len(bow_target), 1)
    f1 = 2 * precision * recall / (precision + recall)
    acc = f1 if qcate in ["color", "shape", "number", "YesNo"] else recall
    return {"f1": f1, "recall": recall, "precision": precision, "em": em, "acc": acc}


def webqa_accuracy(prediction: str, references: List[str], metadata: Optional[Dict[str, Any]] = None) -> float:
    metadata = metadata or {}
    qcate = _normalize_webqa_qcate(
        metadata.get("webqa_qcate") or metadata.get("qcate") or metadata.get("question_category") or "text"
    )
    if not references:
        return 0.0
    best = 0.0
    for ref in references:
        best = max(best, _webqa_acc_single(prediction, ref, qcate)["acc"])
    return best


# ------------------------------------------------------------------
# Per-instance metrics
# ------------------------------------------------------------------

def exact_match(prediction: str, references: List[str]) -> float:
    """1.0 if prediction matches any reference after normalization."""
    pred_norm = normalize_answer(prediction)
    return float(any(pred_norm == normalize_answer(ref) for ref in references))


def token_f1(prediction: str, references: List[str]) -> float:
    """
    Token-level F1 score.
    Takes the maximum F1 over all reference answers.
    """
    pred_tokens = normalize_answer(prediction).split()

    best_f1 = 0.0
    for ref in references:
        ref_tokens = normalize_answer(ref).split()

        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            f1 = float(pred_tokens == ref_tokens)
        else:
            common = Counter(pred_tokens) & Counter(ref_tokens)
            n_common = sum(common.values())
            if n_common == 0:
                f1 = 0.0
            else:
                precision = n_common / len(pred_tokens)
                recall = n_common / len(ref_tokens)
                f1 = 2 * precision * recall / (precision + recall)

        best_f1 = max(best_f1, f1)

    return best_f1


def rouge_l(prediction: str, references: List[str]) -> float:
    """
    ROUGE-L F1 score (LCS-based).
    """
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        best = 0.0
        for ref in references:
            score = scorer.score(ref, prediction)["rougeL"].fmeasure
            best = max(best, score)
        return best
    except ImportError:
        # Fallback: simple LCS
        return _lcs_f1(prediction, references)


def _lcs_f1(prediction: str, references: List[str]) -> float:
    """Fallback LCS-based ROUGE-L if rouge_score not installed."""
    pred_tokens = normalize_answer(prediction).split()
    best = 0.0
    for ref in references:
        ref_tokens = normalize_answer(ref).split()
        lcs_len = _lcs_length(pred_tokens, ref_tokens)
        if lcs_len == 0:
            continue
        precision = lcs_len / max(len(pred_tokens), 1)
        recall = lcs_len / max(len(ref_tokens), 1)
        f1 = 2 * precision * recall / (precision + recall)
        best = max(best, f1)
    return best


def _lcs_length(a: List[str], b: List[str]) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(2)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i % 2][j] = dp[(i - 1) % 2][j - 1] + 1
            else:
                dp[i % 2][j] = max(dp[(i - 1) % 2][j], dp[i % 2][j - 1])
    return dp[m % 2][n]


# ------------------------------------------------------------------
# Aggregate evaluation
# ------------------------------------------------------------------

def evaluate(
    predictions: List[str],
    references: List[List[str]],
    token_counts: Optional[List[int]] = None,
    sample_metadata: Optional[List[Optional[Dict[str, Any]]]] = None,
) -> Dict[str, float]:
    """
    Compute aggregate metrics over a list of predictions.

    Args
    ----
    predictions  : list of generated answer strings
    references   : list of reference answer lists (multiple refs per question)
    token_counts : optional list of input token counts per query

    Returns
    -------
    metrics dict with keys: em, f1, rouge_l, avg_tokens (if provided)
    """
    assert len(predictions) == len(references), "Length mismatch"
    if sample_metadata is not None:
        assert len(sample_metadata) == len(predictions), "Metadata length mismatch"

    em_scores, f1_scores, rl_scores = [], [], []
    webqa_scores = []

    for idx, (pred, refs) in enumerate(zip(predictions, references)):
        em_scores.append(exact_match(pred, refs))
        f1_scores.append(token_f1(pred, refs))
        rl_scores.append(rouge_l(pred, refs))
        if sample_metadata is not None:
            meta = sample_metadata[idx] or {}
            source = (meta.get("source") or "").lower()
            if source == "webqa":
                webqa_scores.append(webqa_accuracy(pred, refs, meta))

    results = {
        "em": float(np.mean(em_scores)),
        "f1": float(np.mean(f1_scores)),
        "rouge_l": float(np.mean(rl_scores)),
        "n_samples": len(predictions),
    }
    if webqa_scores:
        results["acc"] = float(np.mean(webqa_scores))
        results["webqa_acc"] = results["acc"]

    if token_counts is not None:
        results["avg_tokens"] = float(np.mean(token_counts))
        results["total_tokens"] = int(np.sum(token_counts))

    return results


def print_results(results: Dict, method_name: str = ""):
    """Pretty-print evaluation results."""
    header = f"=== {method_name} ===" if method_name else "=== Results ==="
    print(header)
    print(f"  Exact Match : {results.get('em', 0):.4f}")
    print(f"  Token F1    : {results.get('f1', 0):.4f}")
    print(f"  ROUGE-L     : {results.get('rouge_l', 0):.4f}")
    if "acc" in results:
        print(f"  Acc         : {results['acc']:.4f}")
    if "recall_at_k" in results:
        print(f"  Recall@k    : {results['recall_at_k']:.4f}")
    if "precision_at_k" in results:
        print(f"  Precision@k : {results['precision_at_k']:.4f}")
    if "recall_at_1" in results:
        print(f"  Recall@1    : {results['recall_at_1']:.4f}")
    if "avg_tokens" in results:
        print(f"  Avg Tokens  : {results['avg_tokens']:.1f}")
    print(f"  N Samples   : {results.get('n_samples', 0)}")
    print()


def compare_results(results_dict: Dict[str, Dict]) -> None:
    """Print a comparison table of multiple methods."""
    methods = list(results_dict.keys())
    has_recall_k = any("recall_at_k" in r for r in results_dict.values())
    has_recall_1 = any("recall_at_1" in r for r in results_dict.values())
    has_acc = any("acc" in r for r in results_dict.values())
    header = f"{'Method':<25} {'EM':>8} {'F1':>8} {'ROUGE-L':>8}"
    if has_acc:
        header += f" {'Acc':>8}"
    if has_recall_k:
        header += f" {'Recall@k':>10}"
    if has_recall_1:
        header += f" {'Recall@1':>10}"
    header += f" {'Tokens':>10}"
    sep = "=" * len(header)
    print("\n" + sep)
    print(header)
    print("-" * len(header))
    for method, res in results_dict.items():
        tokens = f"{res['avg_tokens']:.0f}" if "avg_tokens" in res else "N/A"
        row = (
            f"{method:<25} "
            f"{res.get('em', 0):>8.4f} "
            f"{res.get('f1', 0):>8.4f} "
            f"{res.get('rouge_l', 0):>8.4f}"
        )
        if has_acc:
            row += f" {res.get('acc', float('nan')):>8.4f}"
        if has_recall_k:
            row += f" {res.get('recall_at_k', float('nan')):>10.4f}"
        if has_recall_1:
            row += f" {res.get('recall_at_1', float('nan')):>10.4f}"
        row += f" {tokens:>10}"
        print(row)
    print(sep + "\n")
