# ALPS — Technical Guide
## How Every File Works, Every Calculation Explained, and How Everything Connects

---

## Table of Contents

1. [High-Level Architecture](#1-high-level-architecture)
2. [Database Layer — `database/`](#2-database-layer)
3. [Configuration — `config.py`](#3-configuration)
4. [LLM Layer — `llm/`](#4-llm-layer)
5. [Adaptivity Layer — `adaptivity/`](#5-adaptivity-layer)
6. [Evaluation Pipeline — `evaluation/`](#6-evaluation-pipeline)
7. [Gamification — `gamification/xp_engine.py`](#7-gamification)
8. [UI and Orchestration — `app.py`](#8-ui-and-orchestration)
9. [End-to-End Data Flows](#9-end-to-end-data-flows)
10. [All Numeric Thresholds and Formulas in One Place](#10-all-numeric-thresholds-and-formulas)

---

## 1. High-Level Architecture

ALPS is a **Streamlit single-file web app** (`app.py`) that wires together five independent subsystems:

```
Browser
  │
  ▼
app.py  (UI, routing, session state, orchestration)
  │
  ├─► database/db.py          (SQLite — all persistence)
  ├─► llm/engine.py           (Gemini API — tutoring + evaluation)
  ├─► adaptivity/             (ZPD level assessment + quiz scoring)
  ├─► evaluation/             (ROUGE → perplexity → BERTScore → LLM judge)
  └─► gamification/           (XP awards, daily streaks)
```

**Request lifecycle for a single chat message:**

```
User types message
      │
      ▼
classify_query()            ← keyword-based query type detection
check_socratic_mode()       ← ratio check on last 10 messages
      │
      ▼
add_message() to DB         ← persist user turn
      │
      ▼
generate_response()         ← Gemini API, streamed
      │
  ┌───┴──────────────────────────────────────┐
  │  Evaluation gates (in order):            │
  │  1. check_pedagogical_output()           │
  │  2. compute_perplexity()                 │
  │  3. compute_bertscore()  (if ref exists) │
  └───┬──────────────────────────────────────┘
      │
      ▼
add_message() to DB         ← persist assistant turn
award_xp()                  ← gamification
      │
      ▼
should_assess()?            ← every N interactions
  └─► run_assessment()      ← ZPD level recommendation
        └─► assess_level()  ← math or LLM judge
```

**Technology stack:**

| Component | Library / Service |
|---|---|
| Web UI | Streamlit |
| Tutoring LLM | Google Gemini (`gemini-2.5-flash`) |
| Evaluation judge | Google Gemini (`gemini-2.5-pro`) |
| Database | SQLite (via `sqlite3` stdlib) |
| ROUGE scoring | `rouge-score` |
| Fluency scoring | GPT-2 via `transformers` + `torch` |
| Semantic similarity | `bert-score` (DistilBERT) |
| Fonts | Inter (body), JetBrains Mono (code/labels) |

---

## 2. Database Layer

### `database/schema.sql` — Six tables

The database lives at `alps.db` (configurable via `DB_PATH` in `config.py`).

#### `users` table
Stores one row per learner. Key columns:

| Column | Type | Purpose |
|---|---|---|
| `user_id` | TEXT PK | Username (lowercase, e.g. `haseeb`) |
| `display_name` | TEXT | Original casing |
| `current_level` | INT (1–4) | The learner's active ZPD level |
| `subject_area` | TEXT | Always "Data Structures and Algorithms" |
| `strictness` | INT (1–5) | Tutor strictness knob, sidebar-controlled |
| `total_interactions` | INT | Running count of all chat turns |
| `xp` | INT | Cumulative XP points earned |
| `last_seen` | DATETIME | Updated on every login |
| `last_level_change` | DATETIME | When the level was last modified |

#### `sessions` table
One row per study session (one topic = one session). Created when a user picks a topic.

| Column | Type | Purpose |
|---|---|---|
| `session_id` | TEXT PK | UUID generated at topic selection |
| `user_id` | TEXT FK | Links to the user |
| `topic` | TEXT | e.g. "Linked Lists", "Binary Search" |
| `direct_answer_count` | INT | Reserved for direct-answer tracking |
| `hint_abuse_flag` | INT (0/1) | Whether abuse was detected |
| `socratic_mode_active` | INT (0/1) | Current Socratic mode status |

#### `conversations` table
One row per message (both user and assistant turns).

| Column | Type | Purpose |
|---|---|---|
| `session_id` | TEXT FK | Which session this message belongs to |
| `user_id` | TEXT FK | Which user |
| `role` | TEXT | `"user"` or `"assistant"` |
| `content` | TEXT | Full message text |
| `level_at_time` | INT | Level when this message was generated |
| `was_socratic_mode` | INT (0/1) | Whether Socratic mode was active |
| `query_classification` | TEXT | `LEARNING_QUERY`, `DIRECT_ANSWER_REQUEST`, or `LEVEL_ADJUSTMENT` |

#### `quiz_results` table
One row per submitted quiz.

| Column | Type | Purpose |
|---|---|---|
| `score` | REAL | Fraction correct (0.0–1.0) |
| `total_q` | INT | Number of questions |
| `correct_q` | INT | Questions answered correctly |
| `level_at_time` | INT | Level when quiz was taken |

#### `evaluations` table
One row per automated quality evaluation of an assistant response.

| Column | Type | Purpose |
|---|---|---|
| `content_accuracy` | REAL | LLM judge score 1–5 |
| `level_appropriateness` | REAL | LLM judge score 1–5 |
| `language_neutrality` | REAL | LLM judge score 1–5 |
| `pedagogical_quality` | REAL | LLM judge score 1–5 |
| `reasoning` | TEXT | Judge's explanation |
| `disagreement` | INT (0/1) | Whether Judge A and B disagreed by ≥ 2 points |
| `judge_model` | TEXT | e.g. `gemini-2.5-pro` |

#### `streaks` table
One row per user, updated daily.

| Column | Type | Purpose |
|---|---|---|
| `current_streak` | INT | Consecutive days of activity |
| `longest_streak` | INT | Historical peak |
| `last_activity_date` | TEXT | ISO date string, e.g. `2026-06-30` |

---

### `database/db.py` — All database access functions

This module provides a **connection factory** and all read/write functions. Every function opens a fresh connection, does its work, and closes it — there is no persistent connection pool.

```python
def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row   # rows are dict-like, accessed as row["column"]
    c.execute("PRAGMA foreign_keys = ON")
    return c
```

**Key functions:**

| Function | What it does |
|---|---|
| `init_db()` | Creates all tables from schema.sql; runs migration ALTER TABLE statements for columns added after initial deployment |
| `upsert_user(...)` | INSERT OR UPDATE: creates a new user or refreshes `last_seen` on re-login. Returns the full user row as a dict |
| `create_session(session_id, user_id, topic)` | INSERT OR IGNORE so clicking "Start" twice doesn't duplicate |
| `add_message(user_id, session_id, role, content, level, socratic, classification)` | Appends one conversation turn to the conversations table |
| `get_session_history(session_id, n=20)` | Returns the last N messages **filtered by session_id** — ensures history from other sessions doesn't bleed in. Returns `[{"role": ..., "content": ...}, ...]` in chronological order (reversed after DESC fetch) |
| `get_session_topic(session_id)` | Reads the topic from the sessions table — used as the authoritative source instead of `session_state.topic` |
| `get_history(user_id, n=20)` | Cross-session history (legacy, still available for `assess_level`'s slow path) |
| `get_quiz_scores(user_id, n=5)` | Returns the last N quiz scores as a list of floats in chronological order |
| `add_xp(user_id, amount)` | Atomic increment: `UPDATE users SET xp = xp + ?` |
| `set_streak(user_id, current, longest, date)` | UPSERT on the streaks table |
| `bump_interactions(user_id)` | Atomic increment of `total_interactions` |
| `update_level(user_id, level)` | Updates `current_level` and `last_level_change` |
| `set_socratic(session_id, active)` | Flips `socratic_mode_active` on the sessions row |

---

## 3. Configuration

### `config.py`

All tunable constants live here. Changing one value changes behavior system-wide.

```python
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "<fallback>")
GEMINI_MODEL    = "gemini-2.5-flash"    # tutoring responses
JUDGE_MODEL     = "gemini-2.5-pro"      # evaluation judge (heavier, more accurate)
DB_PATH         = "alps.db"
DEFAULT_SUBJECT = "Data Structures and Algorithms"

LEVEL_LABELS = {1: "Basic", 2: "Medium", 3: "Advanced", 4: "Super-Advanced"}
LEVEL_COLORS = {1: "#4ade80", 2: "#facc15", 3: "#fb923c", 4: "#f87171"}

ASSESSMENT_INTERVAL = 2     # run ZPD check every 2 interactions
MIN_QUIZZES_FOR_ZPD = 1     # minimum quizzes before using quiz-score fast path
JUDGE_SAMPLE_RATE   = 5     # evaluate 1 in 5 responses with the LLM judge

# ZPD accuracy bands (quiz score as fraction 0.0–1.0)
ZPD = {
    "decrease":      0.50,   # below 50% → recommend DECREASE
    "scaffold":      0.65,   # 50–65% → MAINTAIN with extra scaffolding
    "optimal_high":  0.85,   # 65–85% → optimal zone, MAINTAIN
    "increase_soft": 0.95,   # 85–95% → ready for more, INCREASE
}                            # above 95% → also INCREASE

# Socratic mode thresholds
HINT_ABUSE_ON_RATIO  = 0.30  # if ≥30% of last 10 messages are direct-answer requests → activate
HINT_ABUSE_OFF_RATIO = 0.20  # deactivation threshold (defined but not yet wired into the app)
HINT_ABUSE_WINDOW    = 10    # rolling window size
```

---

## 4. LLM Layer

### `llm/engine.py` — Four Gemini interfaces

This module initializes a single Gemini client at import time and exposes four functions that the rest of the app calls.

```python
_client = genai.Client(api_key=GEMINI_API_KEY)
```

---

#### Interface 1 — `generate_response()`

This is the core tutoring function. Every chat message goes through here.

**Signature:**
```python
generate_response(query, level, socratic_mode=False, strictness=3,
                  topic="General", subject_area="DSA",
                  chat_history=None, stream=False)
```

**What it does step by step:**

1. Picks the prompt template level:
   - If `socratic_mode=True` → always use level 5 (Socratic)
   - Otherwise → `max(1, min(level, 4))`

2. Loads and fills the template:
   ```python
   system = _fill(_load_template(tpl_level),
                  subject_area=subject_area,
                  topic=topic,
                  strictness=strictness)
   ```
   `_fill()` is a simple string replace: `{topic}` → actual topic, etc.

3. Builds the anchored user message:
   ```python
   anchored_query = (
       f"The student is currently learning about: {topic}. "
       f"Always interpret this message in the context of {topic}, "
       f"even if it is vague or generic.\n\n"
       f"Student's message: {query}"
   )
   ```
   The topic is embedded **directly in the user turn** — not just in the system instruction — so the model sees topic context immediately before generating its response. This prevents topic drift on vague follow-ups like "explain more."

4. Assembles the message list:
   ```
   [system instruction]  ← persona + level behavior + topic enforcement
   [chat history]        ← prior turns from THIS session only
   [anchored user msg]   ← current question with topic prefix
   ```

5. Calls `_call(messages, stream=True/False)`.

**How `_call()` works:**

```python
def _call(messages, stream=False):
    system, history = _to_gemini(messages)
    # _to_gemini extracts the system role into system_instruction
    # and converts "assistant" → "model" for Gemini compatibility

    config = types.GenerateContentConfig(system_instruction=system)

    prior    = history[:-1]   # all history except the last message
    user_msg = history[-1]["parts"][0]["text"]  # the final user turn

    if prior:
        chat = _client.chats.create(model=GEMINI_MODEL, history=prior, config=config)
        response = chat.send_message(user_msg)   # or send_message_stream()
    else:
        response = _client.models.generate_content(model=GEMINI_MODEL,
                                                   contents=user_msg,
                                                   config=config)
```

When there is no prior history (first message in a session), Gemini is called with `generate_content` directly. When there is history, a Chat object is created with the prior turns baked in, then the current message is sent. This gives Gemini proper conversation context.

---

#### Interface 2 — `assess_level()`

Determines whether to increase, decrease, or maintain the learner's level. Takes chat history, recent quiz scores, and current level as input.

**Fast path — used when ≥ `MIN_QUIZZES_FOR_ZPD` quiz scores exist:**

```python
acc_pct = mean(quiz_scores) * 100  # average accuracy as percentage

if   acc_pct < ZPD["decrease"]      * 100:  recommendation = "DECREASE"
elif acc_pct < ZPD["scaffold"]       * 100:  recommendation = "MAINTAIN"
elif acc_pct <= ZPD["optimal_high"]  * 100:  recommendation = "MAINTAIN"
elif acc_pct <= ZPD["increase_soft"] * 100:  recommendation = "INCREASE"
else:                                         recommendation = "INCREASE"
```

The thresholds map to these accuracy ranges:

```
0%  ──── 50% → DECREASE    (struggling, level is too hard)
50% ──── 65% → MAINTAIN    (borderline — scaffold without moving)
65% ──── 85% → MAINTAIN    (optimal ZPD zone)
85% ──── 95% → INCREASE    (ready for a challenge)
95% ─── 100% → INCREASE    (clearly too easy)
```

This is called "ZPD" (Zone of Proximal Development) — a concept from educational psychology. The ZPD is the range just above what a learner can do independently but within reach with support. Accuracy of 65–85% represents that zone: the learner is working but not overwhelmed.

**Slow path — used when there are fewer than `MIN_QUIZZES_FOR_ZPD` quiz scores:**

The LLM is asked to read the conversation history and produce a recommendation:

```python
prompt = f"""Analyze these learner interactions.
Current level: {current_level}/4 ({LEVEL_LABELS[current_level]})
Quiz accuracy: {acc_str}

Consider:
1. Above 85% consistently → INCREASE
2. Below 50% consistently → DECREASE
3. Confusion signals → DECREASE
4. Optimal 65-85% with engaged replies → MAINTAIN

Conversation: {history_txt}

Respond ONLY in JSON: {"recommendation": ..., "reasoning": ..., "confidence": ...}"""
```

Returns: `{"recommendation": "INCREASE|DECREASE|MAINTAIN", "reasoning": str, "confidence": float}`

---

#### Interface 3 — `generate_quiz()`

Calls Gemini to produce N multiple-choice questions for a given topic and level. The response is expected as a raw JSON array with no markdown fencing.

The prompt specifies exactly:
- 4 options per question
- `correct_answer` as a 0-based index (not A/B/C/D)
- A brief explanation per question

After getting the response, it validates every question:
```python
[
  q for q in qs
  if all(k in q for k in ("question","options","correct_answer","explanation"))
  and len(q["options"]) == 4
  and all(isinstance(o, str) and o.strip() for o in q["options"])
  and isinstance(q["correct_answer"], int)
  and 0 <= q["correct_answer"] <= 3
]
```
Any malformed question is silently dropped.

---

#### Interface 4 — `check_pedagogical_output()`

Audits a completed tutor response to determine if it is pedagogically appropriate. Returns `(has_scaffolding: bool, gives_away_answer: bool)`.

The LLM is asked two yes/no questions:
1. Does the response give away a complete answer without requiring the learner to think?
2. Does the response include at least one scaffolding element (guiding question, hint, or partial solution)?

If the model says `gives_away_answer=True` AND `has_scaffolding=False`, the entire response is regenerated with an instruction to add scaffolding. If the JSON parse fails, safe defaults `(True, False)` are returned so no incorrect regeneration occurs.

---

### `llm/prompts/` — Five level-specific templates

All five templates follow the same structure but tune pedagogy for the learner's level:

**`level_1_basic.txt`** — Beginner
```
You are a beginner-friendly {subject_area} tutor. Use simple language, real-world
analogies, and short examples. Show 1-2 reasoning steps. End with one check-in
question to confirm understanding.
Strictness: {strictness}/5

The student is currently studying: {topic}. Treat every message as being
specifically about {topic}, even if phrased vaguely (e.g. "explain more",
"teach me intro", "give me an example").
```

**`level_2_medium.txt`** — Intermediate: technical vocabulary, 2–3 reasoning steps, deeper follow-up

**`level_3_advanced.txt`** — Advanced: full vocabulary, edge cases, complexity analysis, challenging question

**`level_4_super.txt`** — Expert: implementation-level detail, tradeoffs, open research questions

**`level_5_socratic.txt`** — Socratic mode (overrides level):
```
NEVER give direct answers. Every response MUST contain at least one guiding
question. Even vague prompts like "teach me intro" should be answered with a
Socratic question that steers the student to discover {topic} concepts themselves.
```

**Key design decision:** `{subject_area}` and `{topic}` appear in every template (including `{topic}` in the topic enforcement paragraph). `{query}` does NOT appear in any template — the student's actual question is passed separately as the user message, not embedded in the system instruction.

---

## 5. Adaptivity Layer

### `adaptivity/level_assessor.py`

Three functions manage the ZPD assessment lifecycle:

**`should_assess(interaction_count) → bool`**
```python
return interaction_count > 0 and interaction_count % ASSESSMENT_INTERVAL == 0
```
With `ASSESSMENT_INTERVAL = 2`, this triggers every 2 chat interactions. The check happens after the assistant's response is saved, so it fires on turns 2, 4, 6, 8, etc.

**`run_assessment(user_id, current_level) → dict | None`**
```python
history = get_history(user_id, n=10)       # last 10 messages across all sessions
scores  = get_quiz_scores(user_id, n=5)    # last 5 quiz scores
result  = assess_level(history, scores, current_level)
return result if result["recommendation"] != "MAINTAIN" else None
```
Returns `None` when the recommendation is MAINTAIN — no banner is shown in that case.

**`apply_recommendation(user_id, current_level, recommendation) → new_level`**
```python
if recommendation == "INCREASE":
    new = min(current_level + 1, 4)    # cap at 4 (Super-Advanced)
elif recommendation == "DECREASE":
    new = max(current_level - 1, 1)    # floor at 1 (Basic)
```
The level always changes by exactly one step — never jumps across multiple levels at once.

**`level_change_message(recommendation, current_level) → str`**
Generates the human-readable suggestion text shown in the banner, e.g.:
- INCREASE: "You're ready for **Medium** level! Would you like to move up?"
- DECREASE: "The material seems challenging. Moving to **Basic** might help."

---

### `adaptivity/quiz_manager.py`

**`run_quiz_session(user_id, session_id, topic, level, num_questions=5)`**
Calls `generate_quiz()` in the LLM engine and wraps the resulting question list with metadata (topic, level, user_id, session_id). Returns `None` if the LLM returned no valid questions.

**`submit_quiz(quiz_data, answers) → result_dict`**
Scores the submitted answers:
```python
score = correct / total    # fraction, e.g. 0.8 for 4/5

# Persists to DB
add_quiz_result(user_id=..., score=score, total_q=total, correct_q=correct, ...)

# Returns detailed breakdown per question
{"is_correct": bool, "explanation": str, ...}
```

**`recent_accuracy(user_id, n=5) → float | None`**
```python
scores = get_quiz_scores(user_id, n)
if len(scores) < 3: return None          # need at least 3 for a meaningful average
return sum(scores) / len(scores)
```

---

## 6. Evaluation Pipeline

The evaluation pipeline runs on **every** tutor response in sequence. It is a three-gate waterfall: a failed gate triggers regeneration, and the regenerated response is NOT re-evaluated through earlier gates.

```
Response generated
     │
     ▼
Gate 1: check_pedagogical_output()     ← LLM judge (always runs)
  gives_away AND !has_scaffolding?
  YES → regenerate with "add scaffolding" instruction
     │
     ▼
Gate 2: compute_perplexity()           ← GPT-2 (always runs)
  perplexity > 200?
  YES → regenerate with "rewrite clearly" instruction
     │
     ▼
Gate 3: compute_bertscore()            ← DistilBERT (only if topic has a reference)
  f1 < 0.75?
  YES → regenerate with "cover key concepts" instruction
     │
     ▼
Final response saved to DB
```

---

### `evaluation/metrics.py`

#### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

ROUGE measures lexical overlap between the generated response and a reference answer.

```python
scorer = RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
scores = scorer.score(reference, candidate)
```

Three variants are computed:
- **ROUGE-1**: unigram overlap (individual words in common)
- **ROUGE-2**: bigram overlap (pairs of consecutive words)
- **ROUGE-L**: longest common subsequence — captures sentence-level structure

All scores are **F1** (harmonic mean of precision and recall):
```
F1 = 2 × (precision × recall) / (precision + recall)

where:
  precision = matched_grams / total_grams_in_candidate
  recall    = matched_grams / total_grams_in_reference
```

`use_stemmer=True` means "running" and "run" count as the same word, reducing false negatives from inflection.

ROUGE-L is the threshold used for the baseline filter: `rouge_l >= 0.20`.

#### Perplexity (GPT-2)

Perplexity measures how "surprised" a language model is by the text. A well-formed, natural English sentence has low perplexity; garbled, incoherent text has high perplexity.

The formula:
```
Perplexity = exp(cross-entropy loss)
           = exp( -1/N × Σ log P(token_i | token_0...token_{i-1}) )
```

Where N is the number of tokens and P is GPT-2's probability for each token given the preceding context.

GPT-2 interpretation for tutor responses:
- **20–100**: natural, fluent English — good response
- **100–200**: slightly unusual phrasing — acceptable
- **> 200**: incoherent or garbled — triggers regeneration

The GPT-2 model (117M parameters) is loaded **lazily on the first call** and cached as a module-level singleton for the lifetime of the process. Loading takes a few seconds but subsequent calls are fast.

```python
encodings = _gpt2_tokenizer(text, return_tensors="pt")
input_ids = encodings.input_ids[:, :_gpt2_model.config.n_positions]  # truncate to 1024 tokens

with torch.no_grad():
    outputs = _gpt2_model(input_ids, labels=input_ids)

perplexity = torch.exp(outputs.loss).item()
```

#### BERTScore (Semantic Similarity)

BERTScore uses contextual embeddings (DistilBERT) to compare meaning rather than just surface words. Two sentences that mean the same thing but use different words will score high; paraphrases that ROUGE would miss will be caught.

```python
P, R, F1 = bert_score_fn(
    [candidate], [reference],
    model_type="distilbert-base-uncased",
)
```

Under the hood, DistilBERT encodes both texts into high-dimensional vectors for each token. Then for each token in the candidate, it finds the most similar token in the reference (cosine similarity), and vice versa:

```
Precision = average max cosine similarity from each candidate token to any reference token
Recall    = average max cosine similarity from each reference token to any candidate token
F1        = 2 × (P × R) / (P + R)
```

Threshold used: **BERTScore F1 ≥ 0.75** to pass Tier 2.

#### Reference answers (`evaluation/reference_answers.py`)

A curated dictionary mapping topic names to expert-written reference answers. Currently covers: Binary Search, Linked Lists, Recursion, Sorting Algorithms, Hash Tables. ROUGE and BERTScore comparisons only run for topics that have a matching entry:

```python
ref_entry = REFERENCE_BY_TOPIC.get(topic.lower())
if ref_entry:
    bert = compute_bertscore(full_response, ref_entry["reference_answer"])
```

For topics not in `REFERENCE_BY_TOPIC`, Gate 3 is skipped entirely.

---

### `evaluation/style_normalizer.py`

Before the LLM judge scores a response, the response is rewritten into neutral academic English via `JUDGE_MODEL`. This removes stylistic bias — an enthusiastic response ("Great question!") and a terse one should be judged equally on content, not tone.

```python
_PROMPT = (
    "Rewrite the following text in neutral standard academic English. "
    "Preserve all factual content and reasoning exactly. Change only writing style. "
    "Do not add or remove any claims. Return only the rewritten text with no preamble."
)
```

If the Gemini call fails, the original text is returned unchanged. The judge still runs, but on the un-normalized text.

---

### `evaluation/judge.py`

After style normalization, `JUDGE_MODEL` (gemini-2.5-pro) scores the response across four dimensions on a 1–5 scale:

| Dimension | What it measures |
|---|---|
| `content_accuracy` | Is the explanation factually correct? |
| `level_appropriateness` | Is the complexity right for this learner's level? |
| `language_neutrality` | Is the response free from demographic or linguistic bias? |
| `pedagogical_quality` | Does it support genuine learning with scaffolding? |

The judge is prompted to return raw JSON with no markdown.

**`resolve_scores(scores_a, scores_b)`** averages Judge A and Judge B scores and flags disagreement:
```python
disagreement = any(abs(scores_a[d] - scores_b[d]) >= 2 for d in _DIMS)
averaged = {d: round((scores_a[d] + scores_b[d]) / 2, 2) for d in _DIMS}
```
Currently Judge B is a placeholder (scores_b == scores_a), so `disagreement` is always False until a second judge (e.g., Llama 3) is wired in.

Note: `save_evaluation()` writes results to the `evaluations` table, but this function is not yet called from the live chat pipeline. The pipeline runs the pedagogical gate (`check_pedagogical_output`) but the multi-dimension LLM judge scoring and DB persistence are not wired in.

---

### `evaluation/bias_report.py` — CLI tool

Run separately from the app:
```bash
python evaluation/bias_report.py
```

Reads all rows from the `evaluations` table, joins with `conversations` to get the learner's level at the time, groups by level, and computes per-level mean scores:

```python
for dim in ["content_accuracy", "level_appropriateness",
            "language_neutrality", "pedagogical_quality"]:
    means[dim] = sum(values) / len(values)
```

Emits a warning if any level's `language_neutrality` score falls below 3.5:
```python
_NEUTRALITY_THRESHOLD = 3.5
if neutrality < _NEUTRALITY_THRESHOLD:
    print(f"[WARN] Level {level}: language_neutrality = {neutrality:.2f}")
```

Output is printed to stdout and saved to `evaluation/bias_report.txt`.

---

## 7. Gamification

### `gamification/xp_engine.py`

#### XP awards

Every meaningful learner action earns XP. The award table:

| Action | XP |
|---|---|
| `LEARNING_QUERY` | 10 — a regular chat message |
| `QUIZ_CORRECT` | 25 — per correct quiz answer |
| `QUIZ_COMPLETED` | 50 — completing a full quiz |
| `LEVEL_UP` | 100 — accepting a level increase |
| `SOCRATIC_EXIT` | 75 — Socratic mode deactivates (learner stopped asking for direct answers) |

```python
def award_xp(user_id, action, count=1):
    amount = XP_TABLE.get(action, 0) * count
    return add_xp(user_id, amount)    # atomic DB increment
```

`count` enables batching, e.g., `award_xp(user_id, "QUIZ_CORRECT", count=3)` for 3 correct answers.

#### Daily streak tracking

```python
def update_streak(user_id):
    today = date.today()
    row   = get_streak(user_id)

    last    = date.fromisoformat(row["last_activity_date"])
    current = row["current_streak"]
    longest = row["longest_streak"]

    if last == today:
        return {..., "is_new_day": False}      # already active today, no change

    if last == today - timedelta(days=1):
        current += 1                            # consecutive day: increment streak
        longest  = max(longest, current)        # update record if beaten
    else:
        current = 1                             # gap in activity: reset to 1

    set_streak(user_id, current, longest, today.isoformat())
    return {..., "is_new_day": True}
```

Three milestone streak values trigger a congratulation message: 3, 7, and 14 days.

The streak progress bar in the sidebar shows `min(streak / 30 * 100, 100)%` — a 30-day streak fills the bar completely.

---

## 8. UI and Orchestration

### `app.py`

This is the entire Streamlit application — routing, rendering, CSS, and orchestration all in one file. Streamlit reruns the entire script from top to bottom on every user interaction.

#### Session state initialization (`_init()`)

```python
{
    "user": None,                  # full user dict from DB
    "session_id": None,            # current session UUID
    "topic": None,                 # current topic string
    "page": "login",               # current page
    "messages": [],                # in-memory message buffer for the UI
    "quiz_data": None,             # in-flight quiz state
    "quiz_result": None,           # completed quiz result
    "socratic_mode": False,        # whether Socratic mode is active
    "direct_history": [],          # list of booleans for last 10 messages
    "assessment_pending": None,    # ZPD result waiting for user confirmation
    "interaction_count": 0,        # total interactions this session
    "pending_classification": None,
    "pending_socratic_exit": False,
    "level_up_animation": False,   # triggers st.balloons() once
    "streak_milestone": None,      # 3/7/14 or None
    "xp_just_changed": False,      # triggers XP flash animation
    "level_just_changed": False,   # triggers level badge pulse animation
}
```

#### Page routing

```python
if   st.session_state.page == "login":  page_login()
elif st.session_state.page == "home":   page_home()
elif st.session_state.page == "chat":   page_chat()
elif st.session_state.page == "quiz":   page_quiz()
elif st.session_state.page == "stats":  page_stats()
```

#### Query classification (Phase B stub)

```python
def classify_query(query):
    direct = ["just tell me","give me the answer","solve this","skip the explanation"]
    adjust = ["simpler","too hard","don't understand","make it easier","eli5"]
    q = query.lower()
    if any(k in q for k in direct):  return "DIRECT_ANSWER_REQUEST"
    if any(k in q for k in adjust):  return "LEVEL_ADJUSTMENT"
    return "LEARNING_QUERY"
```

This is a keyword list, not an LLM classifier. It is marked as a Phase B stub — future versions will replace this with a proper intent classifier.

#### Socratic mode activation

```python
def check_socratic_mode(history):
    if len(history) < 3: return False
    ratio = sum(history[-10:]) / min(len(history), 10)
    return ratio >= 0.30
```

`history` is a list of booleans: `True` = the message was a `DIRECT_ANSWER_REQUEST`. If ≥ 30% of the last 10 messages were direct-answer requests, Socratic mode activates. There is no hysteresis deactivation — once below 30%, mode stays on until the flag is explicitly cleared (the `HINT_ABUSE_OFF_RATIO = 0.20` deactivation threshold is defined in `config.py` but not yet wired in).

#### Dynamic level color injection

```python
def _inject_level_color(level: int):
    hex_color = _LEVEL_HEX.get(level, "#8888a0")
    st.markdown(
        f'<style>:root {{ --current-level-color: {hex_color}; }}</style>',
        unsafe_allow_html=True,
    )
```

This injects a CSS custom property (`--current-level-color`) into the page root on every rerun. The assistant message `border-left` in the chat uses `var(--current-level-color)`, so the accent color automatically changes when the learner's level changes without rebuilding any HTML.

#### Level badges

```python
def render_level_badge(level: int, pulse: bool = False) -> str:
    label = {1: "L1  Basic", 2: "L2  Medium", 3: "L3  Advanced", 4: "L4  Super"}[level]
    color = {1: "var(--level-1)", 2: "var(--level-2)", 3: "var(--level-3)", 4: "var(--level-4)"}[level]
    anim  = " animation: pulse 1s ease-out;" if pulse else ""
    return f'<span style="border:1px solid {color};color:{color};...">{label}</span>'
```

Badges are border-only spans (no fill). `pulse=True` triggers the CSS `pulse` keyframe animation — used on the sidebar badge immediately after a level change.

#### Chat page render order (`page_chat()`)

```
1. _inject_level_color(level)          → CSS variable for message borders
2. level_up_animation check            → st.balloons() if level just changed
3. Topic selection screen (if no topic) → returns early
4. Header: "{topic} Studio" + level badge + "New topic" button
5. ────────────────────────────────
6. Message history loop                 → render all past messages from session_state
7. Assessment banner (if pending)       → ZPD suggestion + Accept/Dismiss buttons
8. st.chat_input()                      → floating input bar (fixed CSS position)
9. User input handler                   → classify, persist, append to messages, rerun
10. Generation pipeline                  → stream response, 3 evaluation gates, persist
```

The assessment banner is positioned **after** the messages and **before** the input. This means it always appears at the bottom of the visible scroll area — where the user already is after reading the latest message — rather than at the top where it would be invisible without scrolling.

#### Generation pipeline (the heart of the app)

```python
if st.session_state.generating and st.session_state.messages:
    if st.session_state.messages[-1]["role"] == "user":
        query = st.session_state.messages[-1]["content"]
        history = get_session_history(sid, n=20)    # session-scoped, no cross-topic bleed

        # Gate 0: stream the response
        stream_gen = generate_response(query=query, level=..., topic=topic,
                                       chat_history=history, stream=True)
        full_response = "".join chunks

        # Gate 1: pedagogical check
        has_scaffolding, gives_away = check_pedagogical_output(full_response)
        if gives_away and not has_scaffolding:
            full_response = generate_response(query + "\n\nRevise to include scaffolding...",
                                              stream=False)

        # Gate 2: perplexity
        ppl = compute_perplexity(full_response)
        if ppl is not None and ppl > 200:
            full_response = generate_response(query + "\n\nRewrite clearly...", stream=False)

        # Gate 3: BERTScore (only for topics with a reference)
        ref_entry = REFERENCE_BY_TOPIC.get(topic.lower())
        if ref_entry:
            bert = compute_bertscore(full_response, ref_entry["reference_answer"])
            if bert["f1"] is not None and not passes_tier2(bert["f1"]):
                full_response = generate_response(query + "\n\nBetter cover key concepts...",
                                                  stream=False)

        # Persist and update state
        add_message(..., "assistant", full_response, ...)
        bump_interactions(user_id)
        award_xp(user_id, classification or "LEARNING_QUERY")

        # ZPD check
        if should_assess(interaction_count):
            result = run_assessment(user_id, current_level)
            if result:
                st.session_state.assessment_pending = result

        st.session_state.generating = False
        st.rerun()
```

The `st.rerun()` at the end causes Streamlit to restart the script from the top. On that second pass, `generating=False` so the pipeline block is skipped, and the messages loop renders the now-complete conversation including the new assistant message.

---

### CSS Design System

The entire visual design is injected at startup as a `<style>` block via `st.markdown(..., unsafe_allow_html=True)`.

**Color tokens:**
```css
--bg-base:     #0c0c14   /* page background */
--bg-surface:  #14141f   /* sidebar, cards */
--bg-raised:   #1c1c2c   /* inputs, user messages */
--bg-border:   #3c3c58   /* dividers, outlines */
--text-primary:   #f2f2f8
--text-secondary: #b4b4cc
--text-muted:     #6a6a88
--level-1: #4ade80  /* green  — Basic */
--level-2: #facc15  /* yellow — Medium */
--level-3: #fb923c  /* orange — Advanced */
--level-4: #f87171  /* red    — Super */
--socratic: #a78bfa /* purple — Socratic mode */
--accent:    #6366f1  /* indigo — primary brand */
--accent-dim: rgba(99,102,241,0.12)  /* accent tint for banners/hover */
```

**Typography:** Inter for body text, JetBrains Mono for all code-style labels (badge text, XP display, message labels like "TUTOR", "YOU").

**Animations:**
- `slideIn` — new messages fade in and slide up 6px
- `pulse` — level badge glows outward when level changes
- `xpFlash` — XP number cycles white → accent when XP is awarded

---

## 9. End-to-End Data Flows

### Flow A: First message in a new session

```
User picks "Linked Lists"
→ create_session(uuid, user_id, "Linked Lists")          DB: sessions table
→ session_state: {session_id: uuid, topic: "Linked Lists", messages: []}
→ st.rerun() → page_chat renders empty chat

User types "teach me intro"
→ classify_query("teach me intro") → "LEARNING_QUERY"
→ check_socratic_mode([]) → False (< 3 messages)
→ add_message(user_id, session_id, "user", "teach me intro", ...)  DB: conversations
→ session_state.messages.append({role:"user", content:"teach me intro"})
→ session_state.generating = True
→ st.rerun()

Generation block triggers:
→ get_session_topic(session_id) → "Linked Lists"           DB: sessions
→ get_session_history(session_id, n=20) → []               DB: conversations (empty, first msg)
→ generate_response(
    query         = "teach me intro",
    level         = 1,
    topic         = "Linked Lists",
    chat_history  = [],
    stream        = True
  )
  → loads level_1_basic.txt, fills {subject_area}, {strictness}, {topic}
  → builds system instruction with "studying: Linked Lists"
  → builds anchored user message:
    "The student is currently learning about: Linked Lists.
     Always interpret this message in the context of Linked Lists...
     Student's message: teach me intro"
  → calls Gemini: generate_content(contents=anchored_msg, config=system_instruction)
  → streams response chunks back

→ check_pedagogical_output(response) → (True, False) → no regeneration
→ compute_perplexity(response) → ~45 → below 200, no regeneration
→ REFERENCE_BY_TOPIC.get("linked lists") → has reference
→ compute_bertscore(response, linked_lists_reference) → f1=0.82 → passes_tier2 → no regeneration

→ add_message(user_id, session_id, "assistant", response, ...)   DB: conversations
→ award_xp(user_id, "LEARNING_QUERY") → +10 XP                  DB: users.xp
→ interaction_count = 1
→ should_assess(1) → 1 % 2 ≠ 0 → False, no assessment

→ session_state.generating = False
→ st.rerun()
→ page_chat renders: header + [YOU: teach me intro] [TUTOR: response] + input bar
```

### Flow B: ZPD assessment triggers (interaction 2)

```
User sends second message, generation completes:
→ interaction_count = 2
→ should_assess(2) → 2 % 2 == 0 → True

→ run_assessment(user_id, current_level=1)
  → get_history(user_id, n=10) → [2 messages]
  → get_quiz_scores(user_id, n=5) → []  (no quizzes yet)
  → assess_level(history=2_msgs, scores=[], level=1)
    → len(scores) < MIN_QUIZZES_FOR_ZPD (1) → slow path
    → sends conversation to JUDGE_MODEL: "Analyze these learner interactions..."
    → LLM returns: {"recommendation": "MAINTAIN", "confidence": 0.7}
  → result["recommendation"] == "MAINTAIN" → return None

→ assessment_pending = None → no banner shown
→ st.rerun()
```

### Flow C: Quiz submitted → level increase

```
User completes quiz on "Linked Lists", scores 5/5 (100%)
→ submit_quiz(quiz_data, answers=[0,2,1,3,0])
  → correct = 5, total = 5, score = 1.0
  → add_quiz_result(user_id, session_id, "Linked Lists", score=1.0, ...)  DB

→ quiz_result shown with ZPD banner:
  → assess_level(history, scores=[1.0], level=1)
    → len(scores) >= MIN_QUIZZES_FOR_ZPD (1) → fast path
    → acc_pct = 1.0 * 100 = 100%
    → 100% > 95% → recommendation = "INCREASE"
  → banner: "You're ready for Medium level!"

User clicks "Yes, update level":
→ apply_recommendation(user_id, 1, "INCREASE")
  → new = min(1+1, 4) = 2
  → update_level(user_id, 2)                       DB: users.current_level = 2
→ award_xp(user_id, "LEVEL_UP") → +100 XP
→ level_up_animation = True → st.balloons() on next render
→ session_state.user["current_level"] = 2
→ st.rerun()
→ level badge now shows "L2  Medium", messages get yellow accent border
```

---

## 10. All Numeric Thresholds and Formulas

| Threshold | Value | Where used | Effect |
|---|---|---|---|
| `ASSESSMENT_INTERVAL` | 2 | `should_assess()` | ZPD check every 2 interactions |
| `MIN_QUIZZES_FOR_ZPD` | 1 | `assess_level()` | Minimum quizzes to use score-based fast path |
| `JUDGE_SAMPLE_RATE` | 5 | (reserved) | LLM judge every 5 responses |
| ZPD `decrease` | 0.50 (50%) | `assess_level()` | Below → DECREASE |
| ZPD `scaffold` | 0.65 (65%) | `assess_level()` | 50–65% → MAINTAIN (scaffold hint) |
| ZPD `optimal_high` | 0.85 (85%) | `assess_level()` | 65–85% → MAINTAIN (optimal zone) |
| ZPD `increase_soft` | 0.95 (95%) | `assess_level()` | 85–100% → INCREASE |
| `HINT_ABUSE_ON_RATIO` | 0.30 (30%) | `check_socratic_mode()` | ≥30% direct requests → Socratic on |
| `HINT_ABUSE_OFF_RATIO` | 0.20 (20%) | config only | Defined, not yet used |
| `HINT_ABUSE_WINDOW` | 10 | `check_socratic_mode()` | Rolling window of last 10 messages |
| Perplexity threshold | 200 | Generation pipeline | > 200 → regenerate |
| ROUGE-L threshold | 0.20 | `is_above_baseline()` | < 0.20 → fails Tier 1 |
| BERTScore F1 threshold | 0.75 | `passes_tier2()` | < 0.75 → regenerate |
| Bias neutrality warn | 3.5 | `bias_report.py` | language_neutrality < 3.5 → WARN |
| Judge disagreement | 2 | `resolve_scores()` | ≥ 2 point gap between judges → flag |

**XP formula:**
```
XP earned = XP_TABLE[action] × count
Total XP   = sum of all earned XP (stored in users.xp, updated atomically)
```

**Streak formula:**
```
last_date == today           → no change (already active today)
last_date == today - 1 day  → current_streak += 1
otherwise                   → current_streak = 1 (reset)
longest_streak = max(longest_streak, current_streak)
```

**Quiz score formula:**
```
score = correct_answers / total_questions    (float 0.0–1.0)
```

**ZPD recommendation formula:**
```
acc_pct = mean(quiz_scores) × 100

if   acc_pct < 50:  DECREASE
elif acc_pct < 65:  MAINTAIN
elif acc_pct ≤ 85:  MAINTAIN
else:               INCREASE
```

**ROUGE-L F1:**
```
LCS = length of longest common subsequence
P   = LCS / len(candidate_tokens)
R   = LCS / len(reference_tokens)
F1  = 2PR / (P + R)
```

**BERTScore F1:**
```
For each candidate token c_i, find max cosine similarity to any reference token r_j
P = mean over all c_i of max_j cos_sim(c_i, r_j)
R = mean over all r_j of max_i cos_sim(c_i, r_j)
F1 = 2PR / (P + R)
```

**GPT-2 Perplexity:**
```
PPL = exp( -1/N × Σ_i log P_GPT2(token_i | token_0...token_{i-1}) )
```

---

*This document covers every file, every calculation, and every connection in the ALPS V1 codebase as of the current commit.*
