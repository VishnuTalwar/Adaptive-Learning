from database.db import get_history, get_quiz_scores, update_level
from llm.engine import assess_level
from config import ASSESSMENT_INTERVAL, LEVEL_LABELS, DECLINE_COOLDOWN_INTERACTIONS

CONFUSION_PHRASES = [
    "i don't understand", "i dont understand", "i'm confused", "im confused",
    "this doesn't make sense", "makes no sense", "i'm lost", "im lost",
    "still confused", "not clear", "i don't get it", "i dont get it",
    "can you explain again", "huh?",
]

def should_assess(interaction_count):
    if interaction_count < ASSESSMENT_INTERVAL:
        return False
    return interaction_count > 0 and interaction_count % ASSESSMENT_INTERVAL == 0

def detect_confusion(chat_history, window=6):
    """Heuristic, down-only confusion signal from the learner's recent turns."""
    user_msgs = [m["content"].lower().strip()
                 for m in chat_history if m["role"] == "user"][-window:]
    hits = sum(1 for msg in user_msgs if any(p in msg for p in CONFUSION_PHRASES))
    return hits >= 2

def in_cooldown(user, recommendation):
    """True if this direction was declined too recently to resurface it."""
    if recommendation not in ("INCREASE", "DECREASE"):
        return False
    if user.get("last_decline_direction") != recommendation:
        return False
    last_decline = user.get("last_decline_interaction")
    if last_decline is None:
        return False
    since = user.get("total_interactions", 0) - last_decline
    return since < DECLINE_COOLDOWN_INTERACTIONS

def run_assessment(user_id, current_level, check_confusion=True):
    history = get_history(user_id, n=10)
    scores  = get_quiz_scores(user_id, n=5)
    result  = assess_level(history, scores, current_level)

    if check_confusion and result["recommendation"] != "DECREASE" and detect_confusion(history):
        result = {
            "recommendation": "DECREASE",
            "reasoning": "Confusion signals detected in recent messages (down-only override).",
            "confidence": 0.7,
        }

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