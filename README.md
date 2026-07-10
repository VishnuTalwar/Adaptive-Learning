# ALPS — Adaptive Learning Pathway System

ALPS is a Streamlit-based adaptive DSA tutoring application that adjusts difficulty in real time using Zone of Proximal Development (ZPD) theory. It tracks each learner's quiz accuracy across four proficiency levels (Basic → Super-Advanced), automatically recommending level changes when performance consistently falls above or below ZPD thresholds. The system includes a Socratic mode that activates when direct-answer requests become too frequent, a gamification layer (XP points, daily streaks, level-up animations), and a multi-stage response evaluation pipeline (ROUGE, GPT-2 perplexity, BERTScore, and an LLM-as-judge scoring tutor responses on accuracy, level fit, neutrality, and pedagogical quality).

---

## Setup

```bash
pip install -r requirements.txt
```

Set your OpenAI API key — either as an environment variable:

```bash
export OPENAI_API_KEY=your_key_here
```

or by copying `.env.example` to `.env` and filling in your key (load it with `python-dotenv` or your shell before running).

> **Note:** Ollama is no longer required. The LLM backend was migrated to OpenAI (`gpt-4o-mini` for tutoring, `gpt-4o` for evaluation).

```bash
streamlit run app.py
```

---

## Module Overview

| Folder | Description |
|---|---|
| `database/` | SQLite schema and all read/write helpers — users, sessions, conversations, quiz results, evaluations, XP, and streaks. |
| `llm/` | OpenAI API client, prompt template loader, and the four core interfaces: `generate_response`, `assess_level`, `generate_quiz`, and `check_pedagogical_output`. |
| `adaptivity/` | ZPD-based level assessment (`level_assessor.py`) and quiz session management (`quiz_manager.py`). |
| `evaluation/` | Automated response quality metrics (ROUGE, perplexity, BERTScore), an LLM-as-judge scoring pipeline with style normalization, and a bias report generator grouped by learner level. |
| `gamification/` | XP award table, daily streak tracking, and helper functions for reading a user's current XP and streak. |

---

## Tuning

Two constants in `config.py` are the main levers for production tuning:

- **`ASSESSMENT_INTERVAL`** — how many chat interactions between ZPD level-change assessments (default: `10`).
- **`JUDGE_SAMPLE_RATE`** — evaluate 1 in every N tutor responses with the LLM judge (default: `3`).
