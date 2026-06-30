# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ALPS (Adaptive Learning Platform for Students) is a Streamlit-based DSA tutoring system that adapts difficulty using Zone of Proximal Development (ZPD) theory. It uses Google Gemini for tutoring and LLM-as-judge evaluation, SQLite for persistence, and includes XP/streak gamification.

## Commands

```bash
# Install dependencies (preferred: uv)
uv pip install -r requirements.txt
# or
pip install -r requirements.txt

# Run the app
streamlit run app.py

# Required environment variable — no hardcoded fallback should exist
export GEMINI_API_KEY=your_key_here
```

**Warning:** `config.py:4` currently contains a hardcoded fallback API key. That key must be revoked and the fallback removed — any environment that skips the `export` will silently use the embedded key.

No test suite exists — manual testing is done via the Streamlit UI.

## Architecture

### Module Responsibilities

| Module | Role |
|---|---|
| `app.py` | All Streamlit UI, page routing, session state, orchestration |
| `config.py` | All tunable constants (ZPD thresholds, model names, intervals) |
| `database/db.py` | Singleton SQLite connection; all DB read/write functions |
| `database/schema.sql` | 6 tables: users, streaks, sessions, conversations, quiz_results, evaluations |
| `llm/engine.py` | 4 Gemini interfaces: `generate_response`, `assess_level`, `generate_quiz`, `check_pedagogical_output` |
| `llm/prompts/` | Level-specific prompt templates (level_1–level_5); level 5 = Socratic mode |
| `adaptivity/level_assessor.py` | ZPD assessment trigger logic and level-change application |
| `adaptivity/quiz_manager.py` | Quiz generation, scoring, DB persistence |
| `evaluation/` | Multi-stage quality pipeline: ROUGE → perplexity → BERTScore → LLM judge |
| `gamification/xp_engine.py` | XP awards by event type, daily streak tracking |

### Key Flows

**Tutoring loop (page_chat):**
1. Classify query via `classify_query()` → `LEARNING_QUERY | DIRECT_ANSWER_REQUEST | LEVEL_ADJUSTMENT`
2. Check Socratic mode via `check_socratic_mode()` — activates when direct-answer ratio in the last 10 messages reaches ≥30%
3. Call `llm.engine.generate_response()` with level 1–5 prompt template
4. Every `ASSESSMENT_INTERVAL` (default 10) interactions, trigger `level_assessor.run_assessment()`
5. Award XP via `xp_engine` per event type

**Phase B stubs (`app.py:21–35`):** `classify_query` and `check_socratic_mode` are explicitly marked as stubs for future work. `classify_query` uses a hardcoded keyword list (not an LLM classifier). `check_socratic_mode` uses only the 30% on-threshold — `HINT_ABUSE_OFF_RATIO = 0.20` is defined in `config.py` but never read; there is no hysteresis. Do not build features on top of these assuming production fidelity.

**ZPD assessment (`assess_level` in engine.py):**
- **Fast path** (≥3 quiz scores available): pure threshold math — no LLM call
- **Slow path** (<3 quizzes): sends conversation history to LLM for JSON recommendation
- Returns `{"recommendation": "INCREASE|DECREASE|MAINTAIN", "reasoning": str, "confidence": float}`

**Response evaluation (sampled at 1-in-`JUDGE_SAMPLE_RATE`):**
- Tier 1: ROUGE-L ≥ 0.20 AND perplexity ≤ 200
- Tier 2: BERTScore F1 ≥ 0.75
- Tier 3: LLM judge scores 4 dimensions (1–5 each) after style normalization

### State Management

All persistent state lives in `alps.db` (SQLite). Streamlit `st.session_state` holds transient in-flight state (current session ID, chat history buffer, UI flags). The DB singleton in `db.py` uses `sqlite3.Row` as row factory (dict-like access).

### LLM Models

- `GEMINI_MODEL = "gemini-2.5-flash"` — tutoring responses
- `JUDGE_MODEL = "gemini-2.5-pro"` — evaluation judge

The `Modelfile.txt` (Ollama/llama3.2 definition) is vestigial from a prior architecture; the app exclusively uses Gemini.

### Configuration Knobs (`config.py`)

```python
ASSESSMENT_INTERVAL = 10      # interactions between ZPD checks
MIN_QUIZZES_FOR_ZPD = 3       # quizzes required before fast-path ZPD
JUDGE_SAMPLE_RATE = 5         # evaluate 1 in every N responses
HINT_ABUSE_ON_RATIO = 0.30    # threshold to activate Socratic mode
HINT_ABUSE_OFF_RATIO = 0.20   # threshold to deactivate Socratic mode
HINT_ABUSE_WINDOW = 10        # rolling message window for ratio calc
```

ZPD thresholds (as score fraction): `decrease=0.50, scaffold=0.65, optimal_high=0.85, increase_soft=0.95`

Note: `HINT_ABUSE_OFF_RATIO = 0.20` is defined here but currently unused — see Phase B stubs note above.

### Known Issues

- `evaluations` table is defined and the pipeline exists but results are not currently written to DB
- `get_full_history()` in `db.py` is defined but unused in `app.py`
- Hardcoded API key fallback in `config.py:4` — must be removed and key rotated
