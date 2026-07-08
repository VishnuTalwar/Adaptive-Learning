CREATE TABLE IF NOT EXISTS users (
    user_id            TEXT    PRIMARY KEY,
    display_name       TEXT    NOT NULL,
    current_level      INTEGER NOT NULL DEFAULT 1,
    subject_area       TEXT    NOT NULL DEFAULT 'Data Structures and Algorithms',
    strictness         INTEGER NOT NULL DEFAULT 3,
    total_interactions INTEGER NOT NULL DEFAULT 0,
    xp                 INTEGER NOT NULL DEFAULT 0,
    created_at         DATETIME DEFAULT (datetime('now')),
    last_seen          DATETIME DEFAULT (datetime('now')),
    last_level_change  DATETIME DEFAULT (datetime('now')),
    last_decline_direction   TEXT    DEFAULT NULL,
    last_decline_interaction INTEGER DEFAULT NULL
);

CREATE TABLE IF NOT EXISTS streaks (
    user_id            TEXT    PRIMARY KEY,
    current_streak     INTEGER NOT NULL DEFAULT 1,
    longest_streak     INTEGER NOT NULL DEFAULT 1,
    last_activity_date TEXT    NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id           TEXT    PRIMARY KEY,
    user_id              TEXT    NOT NULL,
    topic                TEXT    NOT NULL,
    started_at           DATETIME DEFAULT (datetime('now')),
    direct_answer_count  INTEGER DEFAULT 0,
    hint_abuse_flag      INTEGER DEFAULT 0,
    socratic_mode_active INTEGER DEFAULT 0,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE IF NOT EXISTS conversations (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id              TEXT    NOT NULL,
    session_id           TEXT    NOT NULL,
    role                 TEXT    NOT NULL,
    content              TEXT    NOT NULL,
    level_at_time        INTEGER NOT NULL,
    was_socratic_mode    INTEGER DEFAULT 0,
    query_classification TEXT    DEFAULT NULL,
    timestamp            DATETIME DEFAULT (datetime('now')),
    FOREIGN KEY (user_id)    REFERENCES users(user_id),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE TABLE IF NOT EXISTS quiz_results (
    quiz_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       TEXT    NOT NULL,
    session_id    TEXT    NOT NULL,
    topic         TEXT    NOT NULL,
    score         REAL    NOT NULL,
    total_q       INTEGER NOT NULL,
    correct_q     INTEGER NOT NULL,
    level_at_time INTEGER NOT NULL,
    timestamp     DATETIME DEFAULT (datetime('now')),
    FOREIGN KEY (user_id)    REFERENCES users(user_id),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE TABLE IF NOT EXISTS evaluations (
    eval_id               INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id               TEXT    NOT NULL,
    session_id            TEXT    NOT NULL,
    judge_model           TEXT    NOT NULL,
    content_accuracy      REAL,
    level_appropriateness REAL,
    language_neutrality   REAL,
    pedagogical_quality   REAL,
    rouge_l               REAL,
    bertscore_f1          REAL,
    reasoning             TEXT,
    disagreement          INTEGER DEFAULT 0,
    timestamp             DATETIME DEFAULT (datetime('now')),
    FOREIGN KEY (user_id)    REFERENCES users(user_id),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);