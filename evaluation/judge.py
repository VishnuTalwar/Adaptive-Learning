"""
evaluation/judge.py
───────────────────
LLM-as-judge pipeline for ALPS tutor response quality.

Scoring model : JUDGE_MODEL (gemini-2.5-pro) — Judge A
Judge B        : reserved for Llama 3 (future work); resolve_scores() currently
                 receives scores_b == scores_a as a placeholder, so disagreement
                 will always be False until Judge B is wired in.

Flow
────
  1. normalize_style()  — strip stylistic bias from the raw response
  2. score_response()   — score on 4 pedagogical dimensions (1–5 each)
  3. resolve_scores()   — average A/B, flag disagreement >= 2 on any dimension
  4. save_evaluation()  — persist to evaluations table via the passed db connection
"""

import json
import re

from google import genai
from config import GEMINI_API_KEY, JUDGE_MODEL
from evaluation.style_normalizer import normalize_style

_client = genai.Client(api_key=GEMINI_API_KEY)

_SCORE_PROMPT = """\
You are an expert educational evaluator. Score the following tutoring response \
on a 1-5 scale across four dimensions. Return ONLY a valid JSON object with no \
markdown, no backticks, no preamble.
Dimensions:
- content_accuracy: is the explanation factually correct? (1=major errors, 5=flawless)
- level_appropriateness: is complexity right for level {user_level} out of 4? (1=wildly mismatched, 5=perfect)
- language_neutrality: free from demographic or linguistic bias? (1=overtly biased, 5=neutral)
- pedagogical_quality: does it support genuine learning with scaffolding? (1=pure answer dump, 5=excellent tutoring)
- reasoning: one sentence explaining your scores
Topic: {topic}
Response to evaluate: {normalized_text}
Return format: {{"content_accuracy": int, "level_appropriateness": int, "language_neutrality": int, "pedagogical_quality": int, "reasoning": str}}"""

_DIMS = ["content_accuracy", "level_appropriateness", "language_neutrality", "pedagogical_quality"]


def score_response(response_text: str, user_level: int, topic: str) -> dict | None:
    """Score a tutor response across four pedagogical dimensions.

    Calls normalize_style() first to remove stylistic bias, then asks
    JUDGE_MODEL to rate the normalized text on 1-5 scales.

    Returns a dict with keys content_accuracy, level_appropriateness,
    language_neutrality, pedagogical_quality (all int), and reasoning (str).
    Returns None if the API call or JSON parse fails.
    """
    normalized_text = normalize_style(response_text)
    prompt = _SCORE_PROMPT.format(
        user_level=user_level,
        topic=topic,
        normalized_text=normalized_text,
    )
    try:
        raw = _client.models.generate_content(
            model=JUDGE_MODEL,
            contents=prompt,
        ).text
        clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        data = json.loads(clean)
        return {
            "content_accuracy":      int(data["content_accuracy"]),
            "level_appropriateness": int(data["level_appropriateness"]),
            "language_neutrality":   int(data["language_neutrality"]),
            "pedagogical_quality":   int(data["pedagogical_quality"]),
            "reasoning":             str(data["reasoning"]),
        }
    except Exception:
        return None


def resolve_scores(scores_a: dict, scores_b: dict) -> dict:
    """Merge Judge A and Judge B scores, flagging large disagreements.

    Disagreement is True when any of the 4 numeric dimensions differ by
    >= 2 points between the two judges.

    NOTE: Judge B (Llama 3) is future work. Until it is wired in,
    callers should pass scores_b == scores_a. This means disagreement
    will always be False in the current implementation.
    """
    disagreement = any(abs(scores_a[d] - scores_b[d]) >= 2 for d in _DIMS)
    averaged = {d: round((scores_a[d] + scores_b[d]) / 2, 2) for d in _DIMS}
    return {
        **averaged,
        "reasoning":    scores_a["reasoning"],
        "disagreement": disagreement,
    }


def save_evaluation(
    user_id: str,
    session_id: str,
    response_text: str,
    scores: dict,
    db_conn,
    rouge_l: float = None,
    bertscore_f1: float = None,
) -> None:
    """Write one evaluation row to the evaluations table.

    Args:
        user_id:       The learner's user ID.
        session_id:    The active session ID.
        response_text: The raw tutor response (stored for audit purposes).
        scores:        Dict returned by resolve_scores() — must contain all
                       four dimension keys plus 'reasoning' and 'disagreement'.
        db_conn:       An open sqlite3 connection (caller owns commit/close).
        rouge_l:       ROUGE-L F1 score from metrics.py, or None if no
                       reference answer was available for the topic.
        bertscore_f1:  BERTScore F1 score from metrics.py, or None if no
                       reference answer was available for the topic.
    """
    db_conn.execute(
        """
        INSERT INTO evaluations
            (user_id, session_id, judge_model,
             content_accuracy, level_appropriateness,
             language_neutrality, pedagogical_quality,
             rouge_l, bertscore_f1,
             reasoning, disagreement)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            session_id,
            JUDGE_MODEL,
            scores["content_accuracy"],
            scores["level_appropriateness"],
            scores["language_neutrality"],
            scores["pedagogical_quality"],
            rouge_l,
            bertscore_f1,
            scores["reasoning"],
            int(scores.get("disagreement", False)),
        ),
    )
    db_conn.commit()
