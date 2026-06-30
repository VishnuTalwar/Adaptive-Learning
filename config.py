import os

# ── API / Model ────────────────────────────────────────────────────────────
# Set GEMINI_API_KEY as an environment variable before running:
#   export GEMINI_API_KEY=your_key_here
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")  # required — no hardcoded fallback
GEMINI_MODEL    = "gemini-2.5-flash"                        # Model used for live tutoring responses
JUDGE_MODEL     = "gemini-2.5-pro"                          # Model used for automated response evaluation
DB_PATH         = "alps.db"                                 # SQLite database file
DEFAULT_SUBJECT = "Data Structures and Algorithms"          # Subject area shown to new users

# ── Level display ──────────────────────────────────────────────────────────
LEVEL_LABELS = {1: "Basic", 2: "Medium", 3: "Advanced", 4: "Super-Advanced"}
LEVEL_COLORS = {1: "#4ade80", 2: "#facc15", 3: "#fb923c", 4: "#f87171"}

# ── Adaptive assessment ────────────────────────────────────────────────────
ASSESSMENT_INTERVAL = 2    # Trigger ZPD assessment every N chat interactions
MIN_QUIZZES_FOR_ZPD = 1     # Minimum completed quizzes before ZPD rule applies

# ZPD accuracy thresholds (quiz score fractions, not percentages)
ZPD = {
    "decrease":      0.50,  # Below this → recommend DECREASE level
    "scaffold":      0.65,  # Below this → MAINTAIN with extra scaffolding
    "optimal_high":  0.85,  # Below this → MAINTAIN
    "increase_soft": 0.95,  # Below this → INCREASE (soft ceiling)
}

# ── Socratic mode ──────────────────────────────────────────────────────────
HINT_ABUSE_WINDOW    = 10   # Rolling window of messages to check for direct-answer requests
HINT_ABUSE_ON_RATIO  = 0.30 # Fraction of direct-answer requests that triggers Socratic mode
HINT_ABUSE_OFF_RATIO = 0.20 # Fraction below which Socratic mode is deactivated

# ── Evaluation / judge ─────────────────────────────────────────────────────
JUDGE_SAMPLE_RATE = 5       # Evaluate 1 in every N tutor responses with the LLM judge
