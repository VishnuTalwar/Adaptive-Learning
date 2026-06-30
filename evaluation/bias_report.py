"""
evaluation/bias_report.py
─────────────────────────
Aggregate judge scores from the evaluations table into a plain-text bias
report, grouped by learner level.

CLI usage
---------
  python evaluation/bias_report.py            # uses DB_PATH from config.py
  python evaluation/bias_report.py alps.db    # explicit path

Output is printed to stdout AND saved to evaluation/bias_report.txt.
"""

import os
import sqlite3
import sys
from datetime import datetime

from config import DB_PATH, LEVEL_LABELS

_REPORT_PATH = os.path.join(os.path.dirname(__file__), "bias_report.txt")

_DIMS = [
    "content_accuracy",
    "level_appropriateness",
    "language_neutrality",
    "pedagogical_quality",
]

_NEUTRALITY_THRESHOLD = 3.5


# ── Data access ────────────────────────────────────────────────────────────

def get_evaluation_data(db_conn) -> list:
    """Return all evaluation rows enriched with user_level from conversations.

    Joins evaluations to conversations via session_id and takes the
    level_at_time from the first assistant message in that session.
    Rows where no matching conversation exists are still returned with
    user_level = None and should be filtered downstream.
    """
    cursor = db_conn.execute(
        """
        SELECT
            e.content_accuracy,
            e.level_appropriateness,
            e.language_neutrality,
            e.pedagogical_quality,
            e.disagreement,
            e.judge_model,
            (
                SELECT c.level_at_time
                FROM   conversations c
                WHERE  c.session_id = e.session_id
                AND    c.role       = 'assistant'
                ORDER  BY c.id
                LIMIT  1
            ) AS user_level
        FROM evaluations e
        """
    )
    return [dict(row) for row in cursor.fetchall()]


# ── Aggregation ────────────────────────────────────────────────────────────

def compute_group_means(rows: list) -> dict:
    """Compute per-level mean scores and overall judge agreement rate.

    Args:
        rows: List of dicts as returned by get_evaluation_data().

    Returns:
        {
            1: {"content_accuracy": float, "level_appropriateness": float,
                "language_neutrality": float, "pedagogical_quality": float,
                "count": int},
            2: { ... },
            ...
            "overall_agreement_rate": float,   # fraction where disagreement == 0
        }
    """
    buckets: dict[int, list] = {lvl: [] for lvl in LEVEL_LABELS}
    agreement_flags = []

    for row in rows:
        level = row.get("user_level")
        if level not in buckets:
            continue
        buckets[level].append(row)
        agreement_flags.append(int(row.get("disagreement", 0)) == 0)

    result: dict = {}
    for level, level_rows in buckets.items():
        if not level_rows:
            continue
        means = {}
        for dim in _DIMS:
            values = [r[dim] for r in level_rows if r.get(dim) is not None]
            means[dim] = round(sum(values) / len(values), 2) if values else None
        means["count"] = len(level_rows)
        result[level] = means

    overall_agreement = (
        round(sum(agreement_flags) / len(agreement_flags) * 100, 1)
        if agreement_flags else 0.0
    )
    result["overall_agreement_rate"] = overall_agreement
    return result


# ── Report formatting ──────────────────────────────────────────────────────

def generate_report(db_path: str = None) -> str:
    """Generate the bias report, save it to bias_report.txt, and return it.

    Args:
        db_path: Path to the SQLite database. Defaults to DB_PATH from config.

    Returns:
        The full report as a string.
    """
    path = db_path or DB_PATH

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    rows = get_evaluation_data(conn)
    conn.close()

    if not rows:
        msg = (
            "No evaluations yet. Run the app and chat to generate data."
        )
        _write(msg)
        return msg

    stats = compute_group_means(rows)
    agreement_rate = stats.pop("overall_agreement_rate")
    total = len(rows)

    lines = []
    lines.append("ALPS BIAS REPORT")
    lines.append(f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append(f"Total evaluations  : {total}")
    lines.append(
        f"Agreement rate     : {agreement_rate}%  "
        "(rows where Judge A and B agreed)"
    )
    lines.append("")

    # ── Per-level table ───────────────────────────────────────────────────
    lines.append("Per-Level Mean Scores  (scale 1–5)")
    lines.append("=" * 66)
    header = f"{'Level':<22}{'Accuracy':>10}{'Level-Fit':>11}{'Neutrality':>12}{'Pedagogy':>11}"
    lines.append(header)
    lines.append("-" * 66)

    bias_signals = []
    for level in sorted(stats.keys()):
        lvl_data = stats[level]
        label = f"{level} – {LEVEL_LABELS[level]}"
        row_parts = [f"{label:<22}"]
        for dim in _DIMS:
            val = lvl_data.get(dim)
            row_parts.append(f"{val:>10.2f}" if val is not None else f"{'N/A':>10}")
        lines.append("".join(row_parts))

        neutrality = lvl_data.get("language_neutrality")
        if neutrality is not None and neutrality < _NEUTRALITY_THRESHOLD:
            bias_signals.append(
                f"  [WARN] Level {level} – {LEVEL_LABELS[level]}: "
                f"language_neutrality = {neutrality:.2f} "
                f"(below threshold {_NEUTRALITY_THRESHOLD})"
            )

    lines.append("-" * 66)
    lines.append(
        f"{'(n)':<22}"
        + "".join(f"{stats[lvl]['count']:>10}" for lvl in sorted(stats.keys()))
    )

    # ── Bias signals ──────────────────────────────────────────────────────
    lines.append("")
    lines.append("Bias Signals  (language_neutrality < 3.5)")
    lines.append("=" * 66)
    if bias_signals:
        lines.extend(bias_signals)
    else:
        lines.append("  No bias signals detected.")

    report = "\n".join(lines)
    _write(report)
    return report


def _write(content: str) -> None:
    with open(_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(content)
        f.write("\n")


# ── CLI entrypoint ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    db_arg = sys.argv[1] if len(sys.argv) > 1 else None
    print(generate_report(db_arg))
