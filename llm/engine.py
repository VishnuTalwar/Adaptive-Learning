import json, os, re
from statistics import mean
import ollama
from config import OLLAMA_MODEL, LEVEL_LABELS, ZPD

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")
_LEVEL_FILE = {
    1: "level_1_basic",
    2: "level_2_medium",
    3: "level_3_advanced",
    4: "level_4_super",
    5: "level_5_socratic"
}

def _load_template(level):
    path = os.path.join(PROMPTS_DIR, f"{_LEVEL_FILE[level]}.txt")
    with open(path, encoding="utf-8") as f:
        return f.read()

def _fill(template, **kwargs):
    for k, v in kwargs.items():
        template = template.replace(f"{{{k}}}", str(v))
    return template

def _parse_json(raw):
    # strip markdown fences
    clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    # try direct parse first
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass
    # fall back: grab the first {...} or [...] block from surrounding prose
    match = re.search(r"(\[.*\]|\{.*\})", clean, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    raise ValueError(f"No JSON found in model output: {clean[:200]}")


def _call(messages, stream=False):
    return ollama.chat(model=OLLAMA_MODEL, messages=messages, stream=stream)


# ── Interface 1 ────────────────────────────────────────────────────────────

def generate_response(query, level, socratic_mode=False, strictness=3,
                      topic="General",
                      subject_area="Data Structures and Algorithms",
                      chat_history=None, stream=False):
    if chat_history is None:
        chat_history = []

    tpl_level = 5 if socratic_mode else max(1, min(level, 4))
    system = _fill(
        _load_template(tpl_level),
        subject_area=subject_area,
        topic=topic,
        strictness=strictness,
        query=query,
    )

    messages = [{"role": "system", "content": system}]
    messages += chat_history
    messages += [{"role": "user", "content": query}]

    if stream:
        # Return a generator — caller iterates over it
        return ollama.chat(model=OLLAMA_MODEL, messages=messages, stream=True)
    else:
        resp = _call(messages, stream=False)
        return resp["message"]["content"]

# ── Interface 2 ────────────────────────────────────────────────────────────

def assess_level(chat_history, quiz_scores, current_level):
    zpd_rec = None
    acc_pct = None

    if len(quiz_scores) >= 3:
        acc_pct = mean(quiz_scores) * 100
        z = ZPD
        if   acc_pct < z["decrease"]      * 100: zpd_rec = "DECREASE"
        elif acc_pct < z["scaffold"]       * 100: zpd_rec = "MAINTAIN"
        elif acc_pct <= z["optimal_high"]  * 100: zpd_rec = "MAINTAIN"
        elif acc_pct <= z["increase_soft"] * 100: zpd_rec = "INCREASE"
        else:                                      zpd_rec = "INCREASE"

    acc_str = f"{acc_pct:.1f}%" if acc_pct else "N/A"

    # ── Fast path ──────────────────────────────────────────────────────────
    # When >= 3 quiz scores exist, the ZPD threshold rule is decisive: quiz data
    # takes priority over the LLM by design. The LLM judgment was only ever used
    # for flavor text here, so we skip the (slow) Ollama call entirely. This
    # makes a quiz-driven level assessment effectively instant.
    if zpd_rec is not None:
        return {
            "recommendation": zpd_rec,
            "reasoning": f"Quiz accuracy {acc_str} -> {zpd_rec} (ZPD threshold rule).",
            "confidence": 0.9,
        }

    # ── Slow path ──────────────────────────────────────────────────────────
    # No quiz data yet (fewer than 3 scores). Fall back to the LLM's judgment
    # of the conversation to make a recommendation.
    history_txt = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in chat_history
    )

    prompt = f"""Analyze these learner interactions.
Current level: {current_level}/4 ({LEVEL_LABELS[current_level]})
Quiz accuracy: {acc_str}

Consider:
1. Above 85% accuracy consistently -> INCREASE
2. Below 50% accuracy consistently -> DECREASE
3. Confusion signals ("I don't understand", very short replies) -> DECREASE
4. Optimal 65-85% with engaged replies -> MAINTAIN

Conversation:
{history_txt}

Respond ONLY in JSON (no markdown):
{{"recommendation":"INCREASE or DECREASE or MAINTAIN","reasoning":"...","confidence":0.0}}"""

    try:
        raw = _call([{"role": "user", "content": prompt}])["message"]["content"]
        llm = _parse_json(raw)
    except Exception as e:
        llm = {"recommendation": "MAINTAIN", "reasoning": str(e), "confidence": 0.4}

    final      = llm.get("recommendation", "MAINTAIN")
    reasoning  = f"No quiz data. LLM: {llm.get('reasoning','')}"
    confidence = float(llm.get("confidence", 0.5))

    return {"recommendation": final, "reasoning": reasoning,
            "confidence": round(confidence, 2)}


# ── Interface 3 ────────────────────────────────────────────────────────────

def generate_quiz(topic, level, num_questions=5):
    difficulty = {
        1: "very basic, conceptual, for a complete beginner",
        2: "intermediate, requiring some technical understanding",
        3: "advanced, involving edge cases and complexity analysis",
        4: "expert-level, requiring deep implementation knowledge",
    }.get(level, "intermediate")

    prompt = f"""Generate exactly {num_questions} multiple-choice questions about "{topic}" in Data Structures and Algorithms.
Difficulty: {difficulty} (Level {level}/4).

Rules:
- 4 answer options each. Plain text only, no A/B/C/D prefix.
- correct_answer is the 0-based INDEX (0, 1, 2, or 3).
- explanation: 1-2 sentences why the answer is correct.

Respond ONLY with a valid JSON array. No markdown, no extra text.

[
  {{
    "question": "...",
    "options": ["...", "...", "...", "..."],
    "correct_answer": 0,
    "explanation": "..."
  }}
]"""

    try:
        raw = _call([{"role": "user", "content": prompt}])["message"]["content"]
        qs  = _parse_json(raw)
        return [
            q for q in qs
            if isinstance(q, dict)
            and all(k in q for k in ("question","options","correct_answer","explanation"))
            and len(q["options"]) == 4
            and all(isinstance(o, str) and o.strip() for o in q["options"])
            and isinstance(q["correct_answer"], int)
            and 0 <= q["correct_answer"] <= 3
        ]
    except Exception as e:
        print(f"[generate_quiz] Error: {e}")
        return []