import sqlite3, os
from config import DB_PATH

SCHEMA = os.path.join(os.path.dirname(__file__), "schema.sql")

def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON")
    return c

def init_db():
    c = _conn()
    with open(SCHEMA) as f:
        c.executescript(f.read())
    c.close()

# ── Users ──────────────────────────────────────────────────────────────────

def upsert_user(user_id, display_name, subject, level=1, strictness=3):
    c = _conn()
    c.execute("""
        INSERT INTO users (user_id, display_name, current_level, subject_area, strictness)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET last_seen = datetime('now')
    """, (user_id, display_name, level, subject, strictness))
    c.commit()
    row = c.execute("SELECT * FROM users WHERE user_id=?", (user_id,)).fetchone()
    c.close()
    return dict(row)

def get_user(user_id):
    c = _conn()
    row = c.execute("SELECT * FROM users WHERE user_id=?", (user_id,)).fetchone()
    c.close()
    return dict(row) if row else None

def all_users():
    c = _conn()
    rows = c.execute("SELECT * FROM users ORDER BY last_seen DESC").fetchall()
    c.close()
    return [dict(r) for r in rows]

def update_level(user_id, level):
    c = _conn()
    c.execute("UPDATE users SET current_level=?, last_level_change=datetime('now') WHERE user_id=?",
              (level, user_id))
    c.commit(); c.close()

def update_strictness(user_id, strictness):
    c = _conn()
    c.execute("UPDATE users SET strictness=? WHERE user_id=?", (strictness, user_id))
    c.commit(); c.close()

def bump_interactions(user_id):
    c = _conn()
    c.execute("UPDATE users SET total_interactions=total_interactions+1 WHERE user_id=?", (user_id,))
    c.commit(); c.close()

# ── Sessions ───────────────────────────────────────────────────────────────

def create_session(session_id, user_id, topic):
    c = _conn()
    c.execute("INSERT OR IGNORE INTO sessions (session_id, user_id, topic) VALUES (?,?,?)",
              (session_id, user_id, topic))
    c.commit(); c.close()

def set_socratic(session_id, active):
    c = _conn()
    c.execute("UPDATE sessions SET socratic_mode_active=? WHERE session_id=?",
              (int(active), session_id))
    c.commit(); c.close()

# ── Conversations ──────────────────────────────────────────────────────────

def add_message(user_id, session_id, role, content, level, socratic, classification=None):
    c = _conn()
    c.execute("""
        INSERT INTO conversations
            (user_id, session_id, role, content, level_at_time,
             was_socratic_mode, query_classification)
        VALUES (?,?,?,?,?,?,?)
    """, (user_id, session_id, role, content, level, int(socratic), classification))
    c.commit(); c.close()

def get_history(user_id, n=20):
    c = _conn()
    rows = c.execute("""
        SELECT role, content FROM conversations
        WHERE user_id=? ORDER BY timestamp DESC, id DESC LIMIT ?
    """, (user_id, n)).fetchall()
    c.close()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

def get_full_history(user_id, n=10):
    c = _conn()
    rows = c.execute("""
        SELECT role, content, level_at_time, was_socratic_mode, query_classification
        FROM conversations WHERE user_id=?
        ORDER BY timestamp DESC, id DESC LIMIT ?
    """, (user_id, n)).fetchall()
    c.close()
    return [dict(r) for r in reversed(rows)]

# ── Quiz Results ───────────────────────────────────────────────────────────

def add_quiz_result(user_id, session_id, topic, score, total_q, correct_q, level):
    c = _conn()
    c.execute("""
        INSERT INTO quiz_results
            (user_id, session_id, topic, score, total_q, correct_q, level_at_time)
        VALUES (?,?,?,?,?,?,?)
    """, (user_id, session_id, topic, round(score, 4), total_q, correct_q, level))
    c.commit(); c.close()

def get_quiz_scores(user_id, n=5):
    c = _conn()
    rows = c.execute("""
        SELECT score FROM quiz_results WHERE user_id=?
        ORDER BY timestamp DESC LIMIT ?
    """, (user_id, n)).fetchall()
    c.close()
    return list(reversed([r["score"] for r in rows]))

def get_quiz_history(user_id):
    c = _conn()
    rows = c.execute("""
        SELECT topic, score, total_q, correct_q, level_at_time, timestamp
        FROM quiz_results WHERE user_id=?
        ORDER BY timestamp DESC
    """, (user_id,)).fetchall()
    c.close()
    return [dict(r) for r in rows]