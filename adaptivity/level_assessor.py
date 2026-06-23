from database.db import get_history, get_quiz_scores, update_level
from llm.engine import assess_level
from config import ASSESSMENT_INTERVAL, LEVEL_LABELS

def should_assess(interaction_count):
    return interaction_count > 0 and interaction_count % ASSESSMENT_INTERVAL == 0

def run_assessment(user_id, current_level):
    history = get_history(user_id, n=10)
    scores  = get_quiz_scores(user_id, n=5)
    result  = assess_level(history, scores, current_level)
    return result if result["recommendation"] != "MAINTAIN" else None

def apply_recommendation(user_id, current_level, recommendation):
    if recommendation == "INCREASE":
        new = min(current_level + 1, 4)
    elif recommendation == "DECREASE":
        new = max(current_level - 1, 1)
    else:
        return current_level
    if new != current_level:
        update_level(user_id, new)
    return new

def level_change_message(recommendation, current_level):
    if recommendation == "INCREASE":
        next_l = min(current_level + 1, 4)
        return (f"📈 You're ready for **{LEVEL_LABELS[next_l]}** level! "
                f"Would you like to move up?")
    elif recommendation == "DECREASE":
        prev_l = max(current_level - 1, 1)
        return (f"📉 The material seems challenging. Moving to "
                f"**{LEVEL_LABELS[prev_l]}** might help. Would you like to adjust?")
    return ""