# ALPS — Adaptive Learning Pathway System
## Project Progress Document

---

## 1. Overview

ALPS is a locally-hosted, AI-powered adaptive tutoring system focused on **Data Structures and Algorithms (DSA)**. It runs entirely on-device using a fine-tuned Ollama model and a Streamlit web UI. The system adapts the difficulty and teaching style of responses based on each learner's performance over time.

---

## 2. Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit (Python) |
| LLM Backend | Ollama (`llama3.2:3b` base, custom `ALPS_New` modelfile) |
| Database | SQLite (`alps.db`) |
| Language | Python 3.12 |
| Styling | Custom CSS injected via `st.markdown` (Space Mono + DM Sans fonts) |

---

## 3. Repository Structure

```
Adaptive-Learning/
├── app.py                        # Main Streamlit app — UI routing and page logic
├── config.py                     # Global constants (model name, ZPD thresholds, levels)
├── Modelfile.txt                 # Ollama custom model definition (ALPS_New)
├── alps.db                       # SQLite database (auto-created on first run)
├── check_zpd.py                  # Standalone ZPD debug/test script
│
├── database/
│   ├── db.py                     # All database access functions
│   └── schema.sql                # Table definitions
│
├── llm/
│   ├── engine.py                 # Three LLM interfaces: chat, assessment, quiz generation
│   └── prompts/
│       ├── level_1_basic.txt
│       ├── level_2_medium.txt
│       ├── level_3_advanced.txt
│       ├── level_4_super.txt
│       └── level_5_socratic.txt
│
└── adaptivity/
    ├── quiz_manager.py           # Quiz generation + scoring + DB persistence
    └── level_assessor.py         # ZPD-based level change logic
```

---

## 4. Custom LLM Model

**File:** `Modelfile.txt`

A custom Ollama model named `ALPS_New` was created from `llama3.2:3b`:

```
FROM llama3.2:3b
PARAMETER temperature 0.7
PARAMETER num_ctx 2048
PARAMETER num_predict 2000
```

The system prompt primes the model as a patient, adaptive DSA tutor. Per-request behaviour is controlled by the level prompt templates (see Section 7), not the modelfile.

---

## 5. Database Schema

**File:** `database/schema.sql`

Five tables are defined:

### `users`
Stores one row per learner.

| Column | Type | Notes |
|---|---|---|
| `user_id` | TEXT PK | Lowercased username |
| `display_name` | TEXT | Display name |
| `current_level` | INTEGER | 1–4 |
| `subject_area` | TEXT | Default: "Data Structures and Algorithms" |
| `strictness` | INTEGER | 1–5 slider value |
| `total_interactions` | INTEGER | Incremented on each AI reply |
| `created_at` / `last_seen` | DATETIME | Timestamps |
| `last_level_change` | DATETIME | When level was last adjusted |

### `sessions`
One row per chat session (unique UUID per topic start).

| Column | Notes |
|---|---|
| `session_id` | UUID PK |
| `topic` | Topic the user chose |
| `socratic_mode_active` | 0/1 flag |
| `direct_answer_count` / `hint_abuse_flag` | Behaviour tracking fields |

### `conversations`
Every message (user + assistant) in every session.

| Column | Notes |
|---|---|
| `role` | `user` or `assistant` |
| `content` | Full message text |
| `level_at_time` | Level when the message was sent |
| `was_socratic_mode` | 0/1 |
| `query_classification` | `LEARNING_QUERY`, `DIRECT_ANSWER_REQUEST`, or `LEVEL_ADJUSTMENT` |

### `quiz_results`
One row per completed quiz.

| Column | Notes |
|---|---|
| `topic` | Quiz topic |
| `score` | Float 0.0–1.0 |
| `total_q` / `correct_q` | Raw counts |
| `level_at_time` | Level when quiz was taken |

### `evaluations`
Placeholder table for a future LLM-judge evaluation pipeline.

| Column | Notes |
|---|---|
| `judge_model` | Model used for evaluation |
| `content_accuracy` / `level_appropriateness` / `language_neutrality` / `pedagogical_quality` | Score fields (REAL) |
| `reasoning` | Judge's explanation text |

---

## 6. Application Pages

### Login (`page_login`)
- Shows returning users in a dropdown (ordered by `last_seen`)
- New user form: username + starting level selector (1–4)
- On submit: calls `upsert_user()`, sets `st.session_state.user`, navigates to Home

### Home (`page_home`)
- Metric cards: Total Chats, Quiz Accuracy (last 10), Quizzes Taken, Current Level
- Quick navigation buttons: "Start studying" → Chat, "Take a quiz" → Quiz
- Recent quiz history (last 5): icon + topic + score + level badge + date

### Chat (`page_chat`)
Two-stage flow:

**Stage 1 — Topic selection:**
- 8 quick-pick topic buttons (Arrays, Linked Lists, Binary Search, Sorting, Binary Trees, DP, Graphs, Hash Tables)
- Free-text input for custom topics
- On selection: creates a new `session_id` (UUID), stores in DB

**Stage 2 — Active chat:**
- Floating glass-pill chat input (fixed at bottom via CSS)
- Streaming responses rendered token-by-token with a `▌` cursor
- Input locks during generation (`st.session_state.generating = True`)
- Every message (user + assistant) is persisted to `conversations` table
- `bump_interactions()` called after each reply
- Level assessment triggered every N interactions (configurable via `ASSESSMENT_INTERVAL`)
- Adaptive suggestion banner shown when assessment recommends INCREASE or DECREASE

### Quiz (`page_quiz`)
Three states:

1. **Setup:** topic input, question count slider (3–10), difficulty selector
2. **Active quiz:** radio buttons for each question, Submit button (disabled until all answered)
3. **Results:** score + level badge + ZPD recommendation + expandable per-question breakdown with explanations

### Stats (`page_stats`)
- Same 4 metric cards as Home
- Line chart of score trend over time
- "By Topic" table: average score per topic
- Full history table with level labels mapped from integers

### Sidebar (`render_sidebar`)
- Brand header + username + level badge + subject
- Navigation buttons (Home / Chat / Quiz / Stats) — active page shown with `primary` button style
- Strictness slider (1–5): persisted to DB on change
- Level override dropdown: immediate DB update + `st.rerun()`
- Socratic mode warning indicator
- Logout button: clears all session state

---

## 7. LLM Engine

**File:** `llm/engine.py`

### Interface 1 — `generate_response()`
Builds the prompt by selecting one of five level templates, filling placeholders, and prepending chat history (last 20 messages). Supports both streaming and non-streaming modes.

### Interface 2 — `assess_level()`
Dual-path assessment:

- **Fast path (≥3 quiz scores exist):** Pure ZPD threshold math — no LLM call. Returns instantly.
  - `< 50%` → DECREASE
  - `50–85%` → MAINTAIN
  - `> 85%` → INCREASE
- **Slow path (< 3 quiz scores):** Sends conversation history to the LLM and asks for a JSON recommendation.

### Interface 3 — `generate_quiz()`
Prompts the LLM to return a JSON array of N MCQ questions. Validates each question for: required keys, exactly 4 options, non-empty strings, valid `correct_answer` index (0–3). Malformed questions are filtered out.

---

## 8. Prompt Templates

**Directory:** `llm/prompts/`

Each template receives: `{topic}`, `{strictness}`, `{query}`, `{subject_area}`.

| File | Level | Behaviour |
|---|---|---|
| `level_1_basic.txt` | 1 — Basic | Simple language + analogies, 1–2 reasoning steps, ends with a check-in question |
| `level_2_medium.txt` | 2 — Medium | Technical vocabulary with brief definitions, 2–3 steps + example, deeper follow-up |
| `level_3_advanced.txt` | 3 — Advanced | Full technical vocabulary, multi-step reasoning, edge cases, complexity, challenging question |
| `level_4_super.txt` | 4 — Super-Advanced | Implementation-level detail, complexity tradeoffs, open research question |
| `level_5_socratic.txt` | Socratic mode | Never gives direct answers; every reply must contain a guiding question |

---

## 9. Adaptivity System

### Zone of Proximal Development (ZPD)

**File:** `config.py`

```python
ZPD = {
    "decrease":      0.50,   # below this → too hard
    "scaffold":      0.65,   # 50–65% → maintain (lower edge)
    "optimal_high":  0.85,   # 65–85% → optimal zone → maintain
    "increase_soft": 0.95,   # above 85% → ready to advance
}
```

### Level Assessment Trigger

`should_assess(interaction_count)` fires every `ASSESSMENT_INTERVAL` interactions (currently set to **2** for testing; production value commented as 10).

### Apply Recommendation

`apply_recommendation()` clamps level changes to [1, 4] and writes the new level to the DB.

### Socratic Mode

Triggered in `app.py` when the ratio of "direct answer request" queries in the last 10 messages exceeds **30%**. The hint-abuse window and thresholds are in `config.py`:

```python
HINT_ABUSE_WINDOW    = 10
HINT_ABUSE_ON_RATIO  = 0.30
HINT_ABUSE_OFF_RATIO = 0.20
```

### Query Classification

Three classes detected by keyword matching in `classify_query()`:
- `DIRECT_ANSWER_REQUEST` — e.g. "just tell me", "give me the answer"
- `LEVEL_ADJUSTMENT` — e.g. "simpler", "too hard", "eli5"
- `LEARNING_QUERY` — everything else

---

## 10. Configuration Constants

**File:** `config.py`

| Constant | Value | Purpose |
|---|---|---|
| `OLLAMA_MODEL` | `"ALPS_New"` | Ollama model name |
| `DB_PATH` | `"alps.db"` | SQLite file path |
| `DEFAULT_SUBJECT` | `"Data Structures and Algorithms"` | Fixed subject area |
| `ASSESSMENT_INTERVAL` | `2` (test) / `10` (prod) | Interactions between level checks |
| `MIN_QUIZZES_FOR_ZPD` | `3` | Min quiz history before ZPD fires |

---

## 11. UI / Styling

Custom CSS is injected via `st.markdown(..., unsafe_allow_html=True)` in `app.py`.

Key design decisions:
- **Dark theme:** `#0f1117` background, `#161b27` sidebar
- **Fonts:** Space Mono (monospace, headings/badges) + DM Sans (body)
- **Chat bubbles:** User messages right-aligned (`margin-left: 15%`), assistant left-aligned (`margin-right: 15%`), Socratic messages highlighted with amber border
- **Floating input:** Chat input box fixed to `bottom: 30px` with glassmorphism effect (backdrop blur, rounded pill shape)
- **Sidebar:** Locked open at 244px, collapse button hidden
- **Level badges:** Color-coded pills — green (Basic), yellow (Medium), orange (Advanced), red (Super-Advanced)

---

## 12. Known Issues / In-Progress

- `st.write("DEBUG chat assessment:", result)` — a temporary debug line left in `page_chat()` (line 576 of `app.py`) that prints raw assessment output to the UI during development.
- The assessment banner in chat (lines 441–455) has a commented-out version showing earlier iteration; the current version uses a styled div instead of `st.info()`.
- `check_zpd.py` — a standalone debug script for testing ZPD thresholds, not part of the main app flow.
- `evaluations` table schema is defined but no code writes to it yet — reserved for a future LLM-judge pipeline.
- `get_full_history()` in `db.py` is defined but not called anywhere in the current UI — prepared for future use.

---

## 13. Phase Markers (from source comments)

The codebase contains stub functions in `app.py` labelled **"Phase B — Vishnu"**:

- `classify_query()` — keyword-based query classification (implemented as stub)
- `check_socratic_mode()` — ratio-based socratic trigger (implemented as stub)

These stubs are functional but were designated for further refinement in a second development phase.

---

## 14. Running the App

**Prerequisites:**
```bash
pip install streamlit ollama pandas
ollama pull llama3.2:3b
ollama create ALPS_New -f Modelfile.txt
```

**Start:**
```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.