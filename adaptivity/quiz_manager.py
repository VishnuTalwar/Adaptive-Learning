from llm.engine import generate_quiz
from database.db import add_quiz_result, get_quiz_scores

def run_quiz_session(user_id, session_id, topic, level, num_questions=5):
    questions = generate_quiz(topic=topic, level=level, num_questions=num_questions)
    if not questions:
        return None
    return {
        "questions":  questions,
        "topic":      topic,
        "level":      level,
        "user_id":    user_id,
        "session_id": session_id,
    }

def submit_quiz(quiz_data, answers):
    questions = quiz_data["questions"]
    correct   = 0
    breakdown = []

    for q, ans in zip(questions, answers):
        is_correct = (ans == q["correct_answer"])
        if is_correct:
            correct += 1
        breakdown.append({
            "question":       q["question"],
            "options":        q["options"],
            "user_answer":    ans,
            "correct_answer": q["correct_answer"],
            "is_correct":     is_correct,
            "explanation":    q["explanation"],
        })

    total = len(questions)
    score = correct / total if total > 0 else 0.0

    add_quiz_result(
        user_id    = quiz_data["user_id"],
        session_id = quiz_data["session_id"],
        topic      = quiz_data["topic"],
        score      = score,
        total_q    = total,
        correct_q  = correct,
        level      = quiz_data["level"],
    )

    return {"score": score, "correct": correct, "total": total,
            "breakdown": breakdown, "topic": quiz_data["topic"],
            "level": quiz_data["level"]}

def recent_accuracy(user_id, n=5):
    scores = get_quiz_scores(user_id, n)
    if len(scores) < 3:
        return None
    return sum(scores) / len(scores)