from datetime import date, timedelta
from database.db import add_xp, get_xp, get_streak, set_streak

XP_TABLE = {
    "LEARNING_QUERY":  10,
    "QUIZ_CORRECT":    25,
    "QUIZ_COMPLETED":  50,
    "LEVEL_UP":       100,
    "SOCRATIC_EXIT":   75,
}


def award_xp(user_id, action, count=1):
    """Award XP for a given action. count allows batching (e.g. multiple correct answers)."""
    amount = XP_TABLE.get(action, 0) * count
    if amount == 0:
        return get_xp(user_id)
    return add_xp(user_id, amount)


def update_streak(user_id):
    """Update the learner's daily streak based on today's date.

    Returns a dict with current_streak, longest_streak, and is_new_day.
    """
    today = date.today()
    row   = get_streak(user_id)

    if row is None:
        set_streak(user_id, 1, 1, today.isoformat())
        return {"current_streak": 1, "longest_streak": 1, "is_new_day": True}

    last    = date.fromisoformat(row["last_activity_date"])
    current = row["current_streak"]
    longest = row["longest_streak"]

    if last == today:
        return {"current_streak": current, "longest_streak": longest, "is_new_day": False}

    if last == today - timedelta(days=1):
        current += 1
        longest  = max(longest, current)
    else:
        current = 1

    set_streak(user_id, current, longest, today.isoformat())
    return {"current_streak": current, "longest_streak": longest, "is_new_day": True}


def get_user_xp_and_streak(user_id):
    """Return current xp, current_streak, and longest_streak for the user."""
    xp  = get_xp(user_id)
    row = get_streak(user_id)
    if row:
        return {
            "xp":             xp,
            "current_streak": row["current_streak"],
            "longest_streak": row["longest_streak"],
        }
    return {"xp": xp, "current_streak": 0, "longest_streak": 0}
