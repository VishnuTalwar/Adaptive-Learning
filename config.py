import os

OLLAMA_MODEL    = "ALPS_New"
DB_PATH         = "alps.db"
DEFAULT_SUBJECT = "Data Structures and Algorithms"

LEVEL_LABELS = {1: "Basic", 2: "Medium", 3: "Advanced", 4: "Super-Advanced"}
LEVEL_COLORS = {1: "#4ade80", 2: "#facc15", 3: "#fb923c", 4: "#f87171"}

# ASSESSMENT_INTERVAL = 10
ASSESSMENT_INTERVAL = 2
MIN_QUIZZES_FOR_ZPD = 3

ZPD = {
    "decrease":      0.50,
    "scaffold":      0.65,
    "optimal_high":  0.85,
    "increase_soft": 0.95,
}

HINT_ABUSE_WINDOW   = 10
HINT_ABUSE_ON_RATIO = 0.30
HINT_ABUSE_OFF_RATIO= 0.20