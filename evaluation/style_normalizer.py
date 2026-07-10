"""
evaluation/style_normalizer.py
──────────────────────────────
Rewrites a response into neutral standard academic English via OpenAI,
removing stylistic bias before judge scoring. Factual content is
preserved exactly — only writing style is changed.
"""

from openai import OpenAI
from config import OPENAI_API_KEY, JUDGE_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

_PROMPT = (
    "Rewrite the following text in neutral standard academic English. "
    "Preserve all factual content and reasoning exactly. Change only writing style. "
    "Do not add or remove any claims. Return only the rewritten text with no preamble. "
    "Text: {text}"
)


def normalize_style(text: str) -> str:
    """Return `text` rewritten in neutral academic English.

    If the OpenAI call fails for any reason, the original text is returned
    unchanged so the calling code can proceed without interruption.
    """
    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": _PROMPT.format(text=text)}],
            temperature=0.1,
            max_tokens=1000,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return text
