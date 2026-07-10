"""
evaluation/metrics.py
─────────────────────
Automated response quality metrics for ALPS.

Metrics
-------
  ROUGE-1, ROUGE-2, ROUGE-L  via rouge-score library
  Perplexity                  via GPT-2 (transformers + torch)
                              — loads once at first call, reused for all
                                subsequent calls via module-level singleton.
  BERTScore                   via bert-score library (distilbert-base-uncased)
                              — semantic similarity using contextual embeddings.
                              — loads once at first call, reused for all
                                subsequent calls via module-level singleton.

Tier-1 rejection filter (is_above_baseline)
--------------------------------------------
  rouge_l    >= 0.20   minimum lexical overlap with reference answer
  perplexity <= 200    maximum incoherence threshold (skipped if None)

Tier-2 semantic filter (passes_tier2)
--------------------------------------
  bertscore_f1 >= 0.75  minimum semantic similarity with reference answer
"""

import transformers
transformers.logging.set_verbosity_error()

from rouge_score import rouge_scorer as _rs

# ── GPT-2 singleton ────────────────────────────────────────────────────────
# Loaded lazily on first compute_perplexity() call, then reused.
# Both remain None until _load_gpt2() succeeds.

_gpt2_tokenizer = None
_gpt2_model     = None


def _load_gpt2() -> bool:
    """Load GPT-2 weights into module globals on the first call.

    Returns True if the model is ready, False if transformers/torch are
    not installed (compute_perplexity will return None in that case).
    """
    global _gpt2_tokenizer, _gpt2_model
    if _gpt2_model is not None:
        return True
    try:
        import torch  # noqa: F401 — verify torch is importable before loading
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        _gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        _gpt2_model     = GPT2LMHeadModel.from_pretrained("gpt2")
        _gpt2_model.eval()
        return True
    except ImportError:
        return False


# ── Public API ─────────────────────────────────────────────────────────────

def compute_rouge(candidate: str, reference: str) -> dict:
    """Return ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.

    Args:
        candidate:  The model-generated response to evaluate.
        reference:  The gold-standard reference answer.

    Returns:
        {"rouge1": float, "rouge2": float, "rougeL": float}
    """
    scorer = _rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }


def compute_perplexity(text: str):
    """Compute GPT-2 perplexity of the response text.

    GPT-2 is loaded once on the first call and cached for the lifetime of
    the process — subsequent calls reuse the already-loaded model.

    Perplexity interpretation:
        20–100   natural, fluent English  (good tutor response)
        100–200  slightly unusual phrasing
        > 200    incoherent / garbled     (triggers regeneration in pipeline)

    Returns:
        float — perplexity score rounded to 2 dp.
        None  — if transformers or torch are not installed.
    """
    if not _load_gpt2():
        return None

    import torch

    encodings = _gpt2_tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids[:, :_gpt2_model.config.n_positions]

    with torch.no_grad():
        outputs = _gpt2_model(input_ids, labels=input_ids)

    return round(torch.exp(outputs.loss).item(), 2)


_bert_scorer = None


def _load_bert_scorer():
    """Load the BERTScorer model into a module global on the first call.

    Reused for all subsequent compute_bertscore() calls instead of reloading
    the model from disk every time.
    """
    global _bert_scorer
    if _bert_scorer is not None:
        return _bert_scorer
    from bert_score import BERTScorer
    _bert_scorer = BERTScorer(model_type="distilbert-base-uncased")
    return _bert_scorer


def compute_bertscore(candidate: str, reference: str) -> dict:
    """Compute BERTScore precision, recall, and F1 using distilbert-base-uncased.

    Returns:
        {"precision": float, "recall": float, "f1": float}
        — or all None values if bert-score is not installed or fails.
    """
    try:
        scorer = _load_bert_scorer()
        P, R, F1 = scorer.score([candidate], [reference], verbose=False)
        return {
            "precision": round(P[0].item(), 4),
            "recall":    round(R[0].item(), 4),
            "f1":        round(F1[0].item(), 4),
        }
    except Exception:
        return {"precision": None, "recall": None, "f1": None}


def passes_tier2(f1: float) -> bool:
    """Tier-2 semantic filter: response passes if BERTScore F1 >= 0.75."""
    return f1 >= 0.75


def is_above_baseline(rouge_l: float, perplexity) -> bool:
    """Tier-1 rejection filter from the ALPS implementation guide.

    A response passes if:
      1. rouge_l    >= 0.20  (minimum lexical overlap with reference)
      2. perplexity <= 200   (coherence threshold — only applied when not None)

    Args:
        rouge_l:     ROUGE-L F1 score (0.0 – 1.0).
        perplexity:  GPT-2 perplexity (float) or None.

    Returns:
        True if the response meets the minimum quality threshold.
    """
    if rouge_l < 0.20:
        return False
    if perplexity is not None and perplexity > 200:
        return False
    return True


def evaluate_response(response_text: str, reference_text: str) -> dict:
    """Run all available metrics and return a combined result dict.

    Args:
        response_text:   The tutor response to evaluate.
        reference_text:  The gold-standard reference answer for the topic.

    Returns:
        {
            "rouge1":              float,
            "rouge2":              float,
            "rougeL":              float,
            "perplexity":          float | None,
            "passes_baseline":     bool,
            "bertscore_precision": float | None,
            "bertscore_recall":    float | None,
            "bertscore_f1":        float | None,
            "passes_tier2":        bool,
        }
    """
    rouge      = compute_rouge(response_text, reference_text)
    perplexity = compute_perplexity(response_text)
    passes     = is_above_baseline(rouge["rougeL"], perplexity)
    bert       = compute_bertscore(response_text, reference_text)

    return {
        "rouge1":              rouge["rouge1"],
        "rouge2":              rouge["rouge2"],
        "rougeL":              rouge["rougeL"],
        "perplexity":          perplexity,
        "passes_baseline":     passes,
        "bertscore_precision": bert["precision"],
        "bertscore_recall":    bert["recall"],
        "bertscore_f1":        bert["f1"],
        "passes_tier2":        passes_tier2(bert["f1"]) if bert["f1"] is not None else False,
    }
