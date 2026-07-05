"""
evaluation/dump_metrics.py
──────────────────────────
Full metrics dump for the ALPS paper — aggregates every metric captured in
the users, sessions, conversations, quiz_results, and evaluations tables
into one timestamped text report (dataset overview, bias detection table,
ROUGE/BERTScore summary, adaptivity metrics, query behaviour, per-topic
performance, system reliability, and a paper-readiness checklist).

CLI usage
---------
  python evaluation/dump_metrics.py            # uses DB_PATH from config.py
  python evaluation/dump_metrics.py alps.db    # explicit path

Output is printed to stdout AND saved to
evaluation/metrics_dump_{YYYY_MM_DD}.txt
"""

import os
import sqlite3
import sys
from datetime import datetime

from config import DB_PATH, JUDGE_SAMPLE_RATE

_NEUTRALITY_THRESHOLD = 3.5
_PEDAGOGY_THRESHOLD   = 3.0
_ROUGE_L_THRESHOLD    = 0.20

_ZONES = [
    ("Below 50% (above ZPD — too hard)",              "score < 0.50"),
    ("50-65% (upper edge)",                            "score >= 0.50 AND score < 0.65"),
    ("65-85% (optimal zone)",                          "score >= 0.65 AND score <= 0.85"),
    ("85-95% (lower edge — ready to advance)",         "score > 0.85 AND score <= 0.95"),
    ("Above 95% (mastered)",                           "score > 0.95"),
]


# ── DB helpers ───────────────────────────────────────────────────────────────

def _connect(db_path=None):
    conn = sqlite3.connect(db_path or DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _fetch_all(conn, query, params=()):
    return [dict(r) for r in conn.execute(query, params).fetchall()]


def _fetch_one(conn, query, params=()):
    row = conn.execute(query, params).fetchone()
    return dict(row) if row else {}


def _fmt(val, decimals=2):
    if val is None:
        return "N/A"
    return f"{val:.{decimals}f}"


def _section(lines, title):
    lines.append("")
    lines.append(f"--- {title} ---")


# ── Section 1 ──────────────────────────────────────────────────────────────

def _dataset_overview(lines, conn):
    _section(lines, "1. DATASET OVERVIEW")

    total_users          = _fetch_one(conn, "SELECT COUNT(*) as n FROM users")["n"]
    total_sessions        = _fetch_one(conn, "SELECT COUNT(*) as n FROM sessions")["n"]
    total_conversations   = _fetch_one(conn, "SELECT COUNT(*) as n FROM conversations")["n"]
    total_assistant       = _fetch_one(conn, "SELECT COUNT(*) as n FROM conversations WHERE role='assistant'")["n"]
    total_quizzes         = _fetch_one(conn, "SELECT COUNT(*) as n FROM quiz_results")["n"]
    total_evals           = _fetch_one(conn, "SELECT COUNT(*) as n FROM evaluations")["n"]
    topics = [r["topic"] for r in _fetch_all(
        conn, "SELECT DISTINCT topic FROM sessions ORDER BY topic")]
    date_range = _fetch_one(
        conn, "SELECT MIN(timestamp) as earliest, MAX(timestamp) as latest FROM conversations")

    lines.append(f"{'Total users':<30}{total_users}")
    lines.append(f"{'Total sessions':<30}{total_sessions}")
    lines.append(f"{'Total conversations':<30}{total_conversations}")
    lines.append(f"{'Total assistant responses':<30}{total_assistant}")
    lines.append(f"{'Total quizzes taken':<30}{total_quizzes}")
    lines.append(f"{'Total evaluations run':<30}{total_evals}")
    lines.append(f"{'Topics covered':<30}{len(topics)}  ({', '.join(topics) if topics else 'none'})")
    lines.append(
        f"{'Date range':<30}"
        f"{date_range.get('earliest') or 'N/A'}  ->  {date_range.get('latest') or 'N/A'}"
    )

    return {
        "total_users": total_users,
        "total_sessions": total_sessions,
        "total_conversations": total_conversations,
        "total_assistant": total_assistant,
        "total_quizzes": total_quizzes,
        "total_evals": total_evals,
        "topics": topics,
    }


# ── Section 2 ──────────────────────────────────────────────────────────────

def _eval_scores_per_level(lines, conn, stats):
    _section(lines, "2. EVALUATION SCORES PER LEVEL (main bias detection table)")

    rows = _fetch_all(conn, """
        SELECT
          c.level_at_time as level,
          COUNT(*) as n,
          ROUND(AVG(e.content_accuracy), 2) as content_accuracy,
          ROUND(AVG(e.level_appropriateness), 2) as level_appropriateness,
          ROUND(AVG(e.language_neutrality), 2) as language_neutrality,
          ROUND(AVG(e.pedagogical_quality), 2) as pedagogical_quality,
          ROUND(AVG(e.rouge_l), 3) as mean_rouge_l,
          ROUND(AVG(e.bertscore_f1), 3) as mean_bertscore_f1
        FROM evaluations e
        JOIN conversations c ON e.session_id = c.session_id
        WHERE c.role = 'assistant'
        GROUP BY c.level_at_time
        ORDER BY c.level_at_time
    """)

    header = (
        f"{'Level':<8}{'n':>5}{'ContentAcc':>12}{'LevelFit':>10}"
        f"{'Neutrality':>12}{'Pedagogy':>10}{'RougeL':>9}{'BertF1':>9}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    per_level_counts     = {}
    per_level_neutrality = {}
    per_level_pedagogy    = {}
    for r in rows:
        lvl = r["level"]
        per_level_counts[lvl]     = r["n"]
        per_level_neutrality[lvl] = r["language_neutrality"]
        per_level_pedagogy[lvl]   = r["pedagogical_quality"]
        lines.append(
            f"{lvl:<8}{r['n']:>5}"
            f"{_fmt(r['content_accuracy']):>12}"
            f"{_fmt(r['level_appropriateness']):>10}"
            f"{_fmt(r['language_neutrality']):>12}"
            f"{_fmt(r['pedagogical_quality']):>10}"
            f"{_fmt(r['mean_rouge_l'], 3):>9}"
            f"{_fmt(r['mean_bertscore_f1'], 3):>9}"
        )

    lines.append("")
    neutrality_vals = {lvl: v for lvl, v in per_level_neutrality.items() if v is not None}
    max_gap = None
    if len(neutrality_vals) >= 2:
        max_gap = round(max(neutrality_vals.values()) - min(neutrality_vals.values()), 2)
        lines.append(f"Max score gap (language_neutrality across levels): {max_gap}")
    else:
        lines.append("Max score gap (language_neutrality across levels): N/A (need >= 2 levels with data)")

    bias_flags = []
    for lvl in sorted(per_level_neutrality):
        v = per_level_neutrality[lvl]
        if v is not None and v < _NEUTRALITY_THRESHOLD:
            lines.append(f"BIAS FLAG: Level {lvl} language_neutrality below threshold (mean={v})")
            bias_flags.append(lvl)

    quality_flags = []
    for lvl in sorted(per_level_pedagogy):
        v = per_level_pedagogy[lvl]
        if v is not None and v < _PEDAGOGY_THRESHOLD:
            lines.append(f"QUALITY FLAG: Level {lvl} pedagogical_quality below threshold (mean={v})")
            quality_flags.append(lvl)

    if not bias_flags and not quality_flags:
        lines.append("No bias or quality flags triggered.")

    stats["per_level_counts"] = per_level_counts
    stats["bias_flags"]       = bias_flags
    stats["quality_flags"]    = quality_flags
    stats["max_neutrality_gap"] = max_gap


# ── Section 3 ──────────────────────────────────────────────────────────────

def _overall_eval_summary(lines, conn, stats):
    _section(lines, "3. OVERALL EVALUATION SUMMARY")

    lines.append(f"{'Total evaluations':<30}{stats['total_evals']}")

    overall = _fetch_one(conn, """
        SELECT
          ROUND(AVG(content_accuracy), 2) as content_accuracy,
          ROUND(AVG(level_appropriateness), 2) as level_appropriateness,
          ROUND(AVG(language_neutrality), 2) as language_neutrality,
          ROUND(AVG(pedagogical_quality), 2) as pedagogical_quality,
          ROUND(AVG(rouge_l), 3) as rouge_l,
          ROUND(AVG(bertscore_f1), 3) as bertscore_f1
        FROM evaluations
    """)
    lines.append(f"{'Mean content_accuracy':<30}{_fmt(overall.get('content_accuracy'))}")
    lines.append(f"{'Mean level_appropriateness':<30}{_fmt(overall.get('level_appropriateness'))}")
    lines.append(f"{'Mean language_neutrality':<30}{_fmt(overall.get('language_neutrality'))}")
    lines.append(f"{'Mean pedagogical_quality':<30}{_fmt(overall.get('pedagogical_quality'))}")
    lines.append(f"{'Mean rouge_l':<30}{_fmt(overall.get('rouge_l'), 3)}")
    lines.append(f"{'Mean bertscore_f1':<30}{_fmt(overall.get('bertscore_f1'), 3)}")

    agreement = _fetch_one(conn, """
        SELECT ROUND(100.0 * SUM(CASE WHEN disagreement=0 THEN 1 ELSE 0 END) / COUNT(*), 1) as pct
        FROM evaluations
    """).get("pct")
    lines.append(f"{'Judge agreement rate':<30}{_fmt(agreement, 1)}%")

    judge_models = [r["judge_model"] for r in _fetch_all(
        conn, "SELECT DISTINCT judge_model FROM evaluations")]
    lines.append(f"{'Judge model(s) used':<30}{', '.join(judge_models) if judge_models else 'N/A'}")

    lines.append("")
    lines.append("Evaluations per level (balance check):")
    for lvl in sorted(stats.get("per_level_counts", {})):
        lines.append(f"  Level {lvl}: {stats['per_level_counts'][lvl]}")

    stats["agreement_rate"] = agreement if agreement is not None else 0.0


# ── Section 4 ──────────────────────────────────────────────────────────────

def _rouge_bertscore_summary(lines, conn, stats):
    _section(lines, "4. ROUGE AND BERTSCORE SUMMARY")

    per_level = _fetch_all(conn, """
        SELECT c.level_at_time as level,
               ROUND(AVG(e.rouge_l), 3) as mean_rouge_l,
               ROUND(AVG(e.bertscore_f1), 3) as mean_bertscore_f1
        FROM evaluations e
        JOIN conversations c ON e.session_id = c.session_id
        WHERE c.role = 'assistant'
        GROUP BY c.level_at_time
        ORDER BY c.level_at_time
    """)
    lines.append(f"{'Level':<10}{'Mean RougeL':>14}{'Mean BertF1':>14}")
    for r in per_level:
        lines.append(
            f"{r['level']:<10}{_fmt(r['mean_rouge_l'], 3):>14}{_fmt(r['mean_bertscore_f1'], 3):>14}"
        )

    overall = _fetch_one(conn, """
        SELECT ROUND(AVG(rouge_l), 3) as rouge_l, ROUND(AVG(bertscore_f1), 3) as bertscore_f1
        FROM evaluations
    """)
    lines.append("")
    lines.append(f"{'Overall mean rouge_l':<30}{_fmt(overall.get('rouge_l'), 3)}")
    lines.append(f"{'Overall mean bertscore_f1':<30}{_fmt(overall.get('bertscore_f1'), 3)}")

    null_rouge = _fetch_one(conn, "SELECT COUNT(*) as n FROM evaluations WHERE rouge_l IS NULL")["n"]
    lines.append(
        f"{'Responses with rouge_l = NULL':<30}{null_rouge}  (no reference answer found for topic)"
    )

    lines.append("")
    lines.append(f"Tier 1 filter threshold: ROUGE-L >= {_ROUGE_L_THRESHOLD}")
    passed = _fetch_one(
        conn, "SELECT COUNT(*) as n FROM evaluations WHERE rouge_l >= ?", (_ROUGE_L_THRESHOLD,)
    )["n"]
    failed = _fetch_one(
        conn, "SELECT COUNT(*) as n FROM evaluations WHERE rouge_l < ?", (_ROUGE_L_THRESHOLD,)
    )["n"]
    lines.append(f"  Passed: {passed}   Failed: {failed}   (NULL excluded: {null_rouge})")


# ── Section 5 ──────────────────────────────────────────────────────────────

def _adaptivity_metrics(lines, conn, stats):
    _section(lines, "5. ADAPTIVITY METRICS")

    level_changes = _fetch_one(
        conn, "SELECT COUNT(*) as n FROM users WHERE last_level_change IS NOT NULL"
    )["n"]
    lines.append(f"{'Total level changes recorded':<32}{level_changes}")

    lines.append("")
    lines.append("Level distribution of current users:")
    dist = _fetch_all(
        conn, "SELECT current_level, COUNT(*) as n FROM users GROUP BY current_level ORDER BY current_level"
    )
    for r in dist:
        lines.append(f"  Level {r['current_level']}: {r['n']}")

    lines.append("")
    lines.append("Quiz accuracy per level:")
    qa = _fetch_all(conn, """
        SELECT level_at_time, COUNT(*) as quizzes, ROUND(AVG(score)*100, 1) as mean_accuracy_pct
        FROM quiz_results GROUP BY level_at_time ORDER BY level_at_time
    """)
    lines.append(f"{'Level':<10}{'Quizzes':>10}{'Mean Acc %':>14}")
    for r in qa:
        lines.append(f"{r['level_at_time']:<10}{r['quizzes']:>10}{_fmt(r['mean_accuracy_pct'], 1):>14}")

    overall_acc = _fetch_one(conn, "SELECT ROUND(AVG(score)*100, 1) as pct FROM quiz_results").get("pct")
    lines.append(f"{'Overall mean quiz accuracy':<32}{_fmt(overall_acc, 1)}%")

    lines.append("")
    lines.append("ZPD zone breakdown (based on quiz scores):")
    for label, cond in _ZONES:
        n = _fetch_one(conn, f"SELECT COUNT(*) as n FROM quiz_results WHERE {cond}")["n"]
        lines.append(f"  {label:<42}{n}")

    stats["total_level_changes"] = level_changes


# ── Section 6 ──────────────────────────────────────────────────────────────

def _query_behaviour_metrics(lines, conn, stats):
    _section(lines, "6. QUERY BEHAVIOUR METRICS")

    lines.append("Query classification breakdown:")
    rows = _fetch_all(conn, """
        SELECT query_classification, COUNT(*) as count,
        ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM conversations WHERE role='user'), 1) as pct
        FROM conversations WHERE role='user'
        GROUP BY query_classification
    """)
    for r in rows:
        label = r["query_classification"] or "(unclassified)"
        lines.append(f"  {label:<24}{r['count']:>6}   {_fmt(r['pct'], 1)}%")

    lines.append("")
    total_responses = stats["total_assistant"]
    socratic_responses = _fetch_one(
        conn, "SELECT COUNT(*) as n FROM conversations WHERE role='assistant' AND was_socratic_mode=1"
    )["n"]
    socratic_pct = round(100.0 * socratic_responses / total_responses, 1) if total_responses else 0.0
    lines.append(
        f"{'Socratic-mode responses':<38}{socratic_responses} / {total_responses}  ({socratic_pct}%)"
    )

    lines.append("")
    activation = _fetch_one(conn, """
        SELECT COUNT(*) total, SUM(hint_abuse_flag) activated,
        ROUND(100.0*SUM(hint_abuse_flag)/COUNT(*), 1) as activation_pct
        FROM sessions
    """)
    activated = activation.get("activated") or 0
    total_sess = activation.get("total") or 0
    lines.append(
        f"{'Socratic activation rate (sessions)':<38}"
        f"{activated} / {total_sess}  ({_fmt(activation.get('activation_pct'), 1)}%)"
    )

    mean_direct = _fetch_one(
        conn, "SELECT ROUND(AVG(direct_answer_count), 1) as n FROM sessions"
    ).get("n")
    lines.append(f"{'Mean direct_answer_count / session':<38}{_fmt(mean_direct, 1)}")

    stats["socratic_activations"] = activated


# ── Section 7 ──────────────────────────────────────────────────────────────

def _per_topic_performance(lines, conn, stats):
    _section(lines, "7. PER-TOPIC PERFORMANCE")

    rows = _fetch_all(conn, """
        SELECT
          s.topic,
          COUNT(DISTINCT s.session_id) as sessions,
          COUNT(qr.quiz_id) as quizzes_taken,
          ROUND(AVG(qr.score)*100, 1) as mean_quiz_accuracy,
          ROUND(AVG(e.pedagogical_quality), 2) as mean_pedagogical_quality,
          ROUND(AVG(e.language_neutrality), 2) as mean_language_neutrality
        FROM sessions s
        LEFT JOIN quiz_results qr ON s.session_id = qr.session_id
        LEFT JOIN evaluations e ON s.session_id = e.session_id
        GROUP BY s.topic
        ORDER BY mean_quiz_accuracy DESC NULLS LAST
    """)

    header = f"{'Topic':<24}{'Sessions':>10}{'Quizzes':>10}{'QuizAcc%':>10}{'Pedagogy':>10}{'Neutrality':>12}"
    lines.append(header)
    lines.append("-" * len(header))
    for r in rows:
        lines.append(
            f"{r['topic']:<24}{r['sessions']:>10}{r['quizzes_taken']:>10}"
            f"{_fmt(r['mean_quiz_accuracy'], 1):>10}{_fmt(r['mean_pedagogical_quality'], 2):>10}"
            f"{_fmt(r['mean_language_neutrality'], 2):>12}"
        )


# ── Section 8 ──────────────────────────────────────────────────────────────

def _system_reliability(lines, conn, stats):
    _section(lines, "8. SYSTEM RELIABILITY")

    total_llm_calls = stats["total_assistant"]
    lines.append(f"{'Total LLM calls attempted (assistant msgs)':<46}{total_llm_calls}")

    expected_evals = total_llm_calls / JUDGE_SAMPLE_RATE if JUDGE_SAMPLE_RATE else 0
    coverage = round(100.0 * stats["total_evals"] / expected_evals, 1) if expected_evals else 0.0
    lines.append(
        f"{'Evaluation pipeline coverage':<46}"
        f"{coverage}%  ({stats['total_evals']} / ~{expected_evals:.1f} expected)"
    )

    null_score_rows = _fetch_one(conn, """
        SELECT COUNT(*) as n FROM evaluations
        WHERE content_accuracy IS NULL OR level_appropriateness IS NULL
           OR language_neutrality IS NULL OR pedagogical_quality IS NULL
    """)["n"]
    lines.append(f"{'Rows with NULL judge score (parse failures)':<46}{null_score_rows}")

    ts = _fetch_one(conn, "SELECT MIN(timestamp) as first, MAX(timestamp) as last FROM evaluations")
    lines.append(f"{'First evaluation timestamp':<46}{ts.get('first') or 'N/A'}")
    lines.append(f"{'Last evaluation timestamp':<46}{ts.get('last') or 'N/A'}")


# ── Paper checklist ─────────────────────────────────────────────────────────

def _paper_checklist(lines, stats):
    lines.append("")
    lines.append("=" * 60)
    lines.append("PAPER CHECKLIST")
    lines.append("=" * 60)

    checks = []

    total_assistant = stats["total_assistant"]
    checks.append((total_assistant >= 80,
                    f"Total assistant responses >= 80 (actual: {total_assistant})"))

    total_evals = stats["total_evals"]
    checks.append((total_evals >= 40,
                    f"Total evaluations >= 40 (actual: {total_evals})"))

    per_level_counts = stats.get("per_level_counts", {})
    level_actual = " ".join(f"L{lvl}={per_level_counts.get(lvl, 0)}" for lvl in (1, 2, 3, 4))
    level_ok = all(per_level_counts.get(lvl, 0) >= 8 for lvl in (1, 2, 3, 4))
    checks.append((level_ok,
                    f"Evaluations at each level >= 8 (actual: {level_actual})"))

    total_quizzes = stats["total_quizzes"]
    checks.append((total_quizzes >= 20,
                    f"Total quizzes >= 20 (actual: {total_quizzes})"))

    level_changes = stats["total_level_changes"]
    checks.append((level_changes >= 1,
                    f"At least 1 level change recorded (actual: {level_changes})"))

    socratic_activations = stats["socratic_activations"]
    checks.append((socratic_activations >= 2,
                    f"At least 2 Socratic activations (actual: {socratic_activations})"))

    n_topics = len(stats["topics"])
    checks.append((n_topics >= 5,
                    f"At least 5 distinct topics covered (actual: {n_topics})"))

    agreement_rate = stats.get("agreement_rate", 0.0)
    checks.append((agreement_rate >= 70.0,
                    f"Judge agreement rate >= 70% (actual: {agreement_rate}%)"))

    bias_flags = stats.get("bias_flags", [])
    if bias_flags:
        checks.append((False, f"No bias flags triggered (flagged levels: {bias_flags})"))
    else:
        checks.append((True, "No bias flags triggered"))

    all_pass = True
    for passed, desc in checks:
        tag = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        lines.append(f"[ {tag} ] {desc}")

    lines.append("")
    if all_pass:
        lines.append("All minimums met. Ready to write results section.")
    else:
        lines.append("Run more test sessions before writing results section.")


# ── Top-level dump ───────────────────────────────────────────────────────────

def generate_dump(db_path: str = None) -> str:
    """Generate the full paper metrics dump, save it to a timestamped file,
    and return it as a string."""
    path = db_path or DB_PATH
    conn = _connect(path)

    lines = []
    lines.append("=" * 60)
    lines.append("ALPS — Full Metrics Dump (Paper Edition)")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Database: {path}")
    lines.append("=" * 60)

    stats = _dataset_overview(lines, conn)
    _eval_scores_per_level(lines, conn, stats)
    _overall_eval_summary(lines, conn, stats)
    _rouge_bertscore_summary(lines, conn, stats)
    _adaptivity_metrics(lines, conn, stats)
    _query_behaviour_metrics(lines, conn, stats)
    _per_topic_performance(lines, conn, stats)
    _system_reliability(lines, conn, stats)

    _paper_checklist(lines, stats)

    lines.append("")
    lines.append("=" * 60)
    lines.append("END OF METRICS DUMP")
    lines.append("=" * 60)

    conn.close()

    report = "\n".join(lines)
    _write(report)
    return report


def _write(content: str) -> None:
    filename = f"metrics_dump_{datetime.now().strftime('%Y_%m_%d')}.txt"
    path = os.path.join(os.path.dirname(__file__), filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
        f.write("\n")


# ── CLI entrypoint ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    db_arg = sys.argv[1] if len(sys.argv) > 1 else None
    print(generate_dump(db_arg))