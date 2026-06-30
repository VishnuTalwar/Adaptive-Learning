"""
evaluation/style_normalizer.py
──────────────────────────────
Rewrites a response into neutral standard academic English via Gemini,
removing stylistic bias before judge scoring. Factual content is
preserved exactly — only writing style is changed.
"""

from google import genai
from config import GEMINI_API_KEY, JUDGE_MODEL

_client = genai.Client(api_key=GEMINI_API_KEY)

_PROMPT = (
    "Rewrite the following text in neutral standard academic English. "
    "Preserve all factual content and reasoning exactly. Change only writing style. "
    "Do not add or remove any claims. Return only the rewritten text with no preamble. "
    "Text: {text}"
)


def normalize_style(text: str) -> str:
    """Return `text` rewritten in neutral academic English.

    If the Gemini call fails for any reason, the original text is returned
    unchanged so the calling code can proceed without interruption.
    """
    try:
        response = _client.models.generate_content(
            model=JUDGE_MODEL,
            contents=_PROMPT.format(text=text),
        )
        return response.text
    except Exception:
        return text
