import json, os, re
from statistics import mean
from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_MODEL, LEVEL_LABELS, ZPD

client = OpenAI(api_key=OPENAI_API_KEY)

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
    clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass
    match = re.search(r"(\[.*\]|\{.*\})", clean, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    raise ValueError(f"No JSON found in model output: {clean[:200]}")


# ── Interface 1 ────────────────────────────────────────────────────────────

def generate_response(query, level, socratic_mode=False, scaffold_mode=False, strictness=3,
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
    )

    if scaffold_mode:
        system += (
            "\n\nThe learner's recent quiz accuracy sits in the 50-65% band — they are "
            "struggling but engaged. Stay at the current level, but add extra scaffolding: "
            "break ideas into smaller steps, check understanding more often with a quick "
            "question, and favor guiding questions over full explanations."
        )

    # Anchor the user message to the session topic so vague follow-ups
    # ("explain more", "teach me intro") cannot be misread as off-topic.
    anchored_query = (
        f"The student is currently learning about: {topic}. "
        f"Always interpret this message in the context of {topic}, "
        f"even if it is vague or generic.\n\n"
        f"Student's message: {query}"
    )

    messages = [{"role": "system", "content": system}]
    messages += chat_history
    messages += [{"role": "user", "content": anchored_query}]

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        stream=stream,
        temperature=0.7,
        max_tokens=2000,
    )

    if stream:
        def _wrapper():
            for chunk in response:
                piece = chunk.choices[0].delta.content
                if piece is not None:
                    yield {"message": {"content": piece}}
        return _wrapper()

    return response.choices[0].message.content


# ── Interface 2 ────────────────────────────────────────────────────────────

def assess_level(chat_history, quiz_scores, current_level):
    zpd_rec = None
    acc_pct = None

    if len(quiz_scores) >= 3:
        acc_pct = mean(quiz_scores) * 100
        z = ZPD
        if   acc_pct < z["decrease"]      * 100: zpd_rec = "DECREASE"
        elif acc_pct < z["scaffold"]       * 100: zpd_rec = "SCAFFOLD"
        elif acc_pct <= z["optimal_high"]  * 100: zpd_rec = "MAINTAIN"
        elif acc_pct <= z["increase_soft"] * 100: zpd_rec = "INCREASE"
        else:                                      zpd_rec = "INCREASE"

    acc_str = f"{acc_pct:.1f}%" if acc_pct else "N/A"

    # Fast path: ZPD threshold rule is decisive when >= 3 quiz scores exist.
    if zpd_rec is not None:
        return {
            "recommendation": zpd_rec,
            "reasoning": f"Quiz accuracy {acc_str} -> {zpd_rec} (ZPD threshold rule).",
            "confidence": 0.9,
        }

    # Slow path: no quiz data yet — ask the LLM to judge the conversation.
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
4. Struggling but engaged, roughly 50-65% -> SCAFFOLD (stay at this level, more scaffolding)
5. Optimal 65-85% with engaged replies -> MAINTAIN

Conversation:
{history_txt}

Respond ONLY in JSON (no markdown):
{{"recommendation":"INCREASE or DECREASE or SCAFFOLD or MAINTAIN","reasoning":"...","confidence":0.0}}"""

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500,
            response_format={"type": "json_object"},
        )
        llm = _parse_json(response.choices[0].message.content)
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

Respond ONLY with a valid JSON object of this exact shape. No markdown, no extra text.

{{
  "questions": [
    {{
      "question": "...",
      "options": ["...", "...", "...", "..."],
      "correct_answer": 0,
      "explanation": "..."
    }}
  ]
}}"""

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2000,
            response_format={"type": "json_object"},
        )
        data = _parse_json(response.choices[0].message.content)
        qs   = data.get("questions", []) if isinstance(data, dict) else []
        return [
            q for q in qs
            if isinstance(q, dict)
            and all(k in q for k in ("question","options","correct_answer","explanation"))
            and len(q["options"]) == 4
            and all(isinstance(o, str) and o.strip() for o in q["options"])
            and isinstance(q["correct_answer"], int)
            and 0 <= q["correct_answer"] <= 3
        ]
    except Exception:
        return []


# ── Interface 4 ────────────────────────────────────────────────────────────

def check_pedagogical_output(response_text):
    """Evaluate whether a tutor response gives away the answer without scaffolding.

    Returns (has_scaffolding: bool, gives_away_answer: bool).
    On any parse failure the safe defaults (True, False) are returned so the
    response is never incorrectly flagged and regenerated.
    """
    prompt = f"""You are evaluating a tutoring response. Answer with JSON only.

gives_away_answer: true ONLY if the response provides a complete working solution or direct final answer with NO attempt to make the student think. A response that explains concepts step by step, uses analogies, asks follow-up questions, OR says 'let me walk you through' is NOT giving away the answer even if it is detailed.

has_scaffolding: true if the response contains ANY of these:
- a guiding question
- a hint without the full answer
- an analogy or example that builds understanding
- a step-by-step explanation that requires the student to follow along
- an invitation to try something themselves

Response to evaluate: {response_text}

Return ONLY: {{"gives_away_answer": true_or_false, "has_scaffolding": true_or_false}}"""

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        data = _parse_json(response.choices[0].message.content)
        return bool(data.get("has_scaffolding", True)), bool(data.get("gives_away_answer", False))
    except Exception:
        return True, False
