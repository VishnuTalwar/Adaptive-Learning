import uuid
import sqlite3
import logging
import threading
import streamlit as st

from database.db import (
    init_db, upsert_user, get_user, all_users,
    create_session, add_message,
    get_session_history, get_session_topic,
    get_quiz_history, get_quiz_scores,
    bump_interactions, update_strictness,
    update_level, set_socratic,
)
from llm.engine import generate_response, check_pedagogical_output
from evaluation.metrics import compute_perplexity, compute_bertscore, passes_tier2, evaluate_response
from evaluation.reference_answers import REFERENCE_BY_TOPIC, REFERENCE_ANSWERS
from evaluation.judge import score_response, resolve_scores, save_evaluation
from adaptivity.quiz_manager import run_quiz_session, submit_quiz
from adaptivity.level_assessor import (
    should_assess, run_assessment,
    apply_recommendation, level_change_message,
)
from config import LEVEL_LABELS, DEFAULT_SUBJECT, DB_PATH, JUDGE_SAMPLE_RATE, HINT_ABUSE_OFF_RATIO
from gamification.xp_engine import award_xp, update_streak, get_user_xp_and_streak

logger = logging.getLogger(__name__)

# ── Stubs for Vishnu (Phase B) ────────────────────────────────────────────

def classify_query(query):
    direct = ["just tell me","give me the answer","what is the answer",
              "solve this","skip the explanation","tell me the solution"]
    adjust = ["simpler","too hard","don't understand","make it easier","eli5"]
    q = query.lower()
    if any(k in q for k in direct):  return "DIRECT_ANSWER_REQUEST"
    if any(k in q for k in adjust):  return "LEVEL_ADJUSTMENT"
    return "LEARNING_QUERY"

def check_socratic_mode(history):
    if len(history) < 5: return False
    ratio = sum(history[-10:]) / min(len(history), 10)
    currently_on = st.session_state.socratic_mode
    if not currently_on and ratio >= 0.30:
        return True
    if currently_on and ratio < HINT_ABUSE_OFF_RATIO:
        return False
    return currently_on

def _streak_milestone(n):
    return n if n in (3, 7, 14) else None

# ── Page config ───────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ALPS",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dynamic level-color injection ─────────────────────────────────────────

_LEVEL_HEX = {1: "#4ade80", 2: "#facc15", 3: "#fb923c", 4: "#f87171"}

def _inject_level_color(level: int):
    hex_color = _LEVEL_HEX.get(level, "#8888a0")
    st.markdown(
        f'<style>:root {{ --current-level-color: {hex_color}; }}</style>',
        unsafe_allow_html=True,
    )

# ── Global stylesheet ─────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg-base:     #0c0c14;
  --bg-surface:  #14141f;
  --bg-raised:   #1c1c2c;
  --bg-border:   #3c3c58;
  --text-primary:   #f2f2f8;
  --text-secondary: #b4b4cc;
  --text-muted:     #6a6a88;
  --level-1: #4ade80;
  --level-2: #facc15;
  --level-3: #fb923c;
  --level-4: #f87171;
  --socratic: #a78bfa;
  --accent:     #6366f1;
  --accent-dim: rgba(99,102,241,0.12);
  --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
  --font-body: 'Inter', -apple-system, sans-serif;
  --current-level-color: #4ade80;
}

html, body, [class*="css"] {
  font-family: var(--font-body) !important;
  box-shadow: none !important;
}
.stApp {
  background: var(--bg-base) !important;
  color: var(--text-secondary) !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container {
  padding: 2rem 2.5rem !important;
  max-width: 900px !important;
  margin: 0 auto !important;
}

h1, h2, h3, h4, h5, h6 {
  font-family: var(--font-body) !important;
  color: var(--text-primary) !important;
}
h1 { font-size: 22px !important; font-weight: 600 !important; }
h2 { font-size: 20px !important; font-weight: 600 !important; }
h3 { font-size: 17px !important; font-weight: 500 !important; }
p, li { color: var(--text-secondary) !important; font-size: 14px !important; line-height: 1.7 !important; }
strong, b { color: var(--text-primary) !important; }
label { color: var(--text-secondary) !important; font-size: 13px !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--bg-surface) !important;
  border-right: 1px solid var(--bg-border) !important;
  min-width: 244px !important;
  max-width: 244px !important;
  transform: none !important;
  visibility: visible !important;
}
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"],
[data-testid="baseButton-headerNoPadding"] {
  display: none !important;
}

/* ── Brand ── */
.alps-brand {
  font-family: var(--font-mono) !important;
  font-size: 15px !important;
  font-weight: 600 !important;
  color: var(--text-primary) !important;
  border-left: 2px solid var(--accent) !important;
  padding-left: 12px !important;
  display: block !important;
  letter-spacing: 0.02em !important;
}
.alps-sub {
  font-size: 11px !important;
  color: var(--text-muted) !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
  padding-left: 14px !important;
}

/* ── Section header ── */
.section-header {
  font-size: 13px !important;
  font-weight: 500 !important;
  color: var(--text-secondary) !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  margin-bottom: 8px !important;
  margin-top: 4px !important;
}

/* ── Divider ── */
hr {
  border: none !important;
  border-top: 1px solid var(--bg-border) !important;
  margin: 1.5rem 0 !important;
}

/* ── Buttons ── */
.stButton > button {
  background: transparent !important;
  color: var(--text-primary) !important;
  border: 1px solid var(--bg-border) !important;
  border-radius: 6px !important;
  font-family: var(--font-body) !important;
  font-size: 14px !important;
  font-weight: 500 !important;
  padding: 6px 14px !important;
  transition: all 0.15s ease !important;
  box-shadow: none !important;
}
.stButton > button:hover {
  border-color: var(--accent) !important;
  color: var(--text-primary) !important;
  background: var(--accent-dim) !important;
  box-shadow: none !important;
}
/* Primary */
[data-testid="stBaseButton-primary"],
.stButton > button[kind="primary"] {
  background: var(--accent) !important;
  border-color: var(--accent) !important;
  color: white !important;
}
[data-testid="stBaseButton-primary"]:hover,
.stButton > button[kind="primary"]:hover {
  opacity: 0.88 !important;
  color: white !important;
  box-shadow: none !important;
}
/* Sidebar active nav */
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"],
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
  background: var(--accent-dim) !important;
  border: 1px solid transparent !important;
  border-left: 2px solid var(--accent) !important;
  color: var(--text-primary) !important;
  opacity: 1 !important;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"]:hover,
[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
  opacity: 1 !important;
  background: var(--accent-dim) !important;
}

/* ── Inputs ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
  background: var(--bg-raised) !important;
  border: 1px solid var(--bg-border) !important;
  border-radius: 6px !important;
  color: var(--text-primary) !important;
  font-family: var(--font-body) !important;
  font-size: 14px !important;
  box-shadow: none !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
  border-color: var(--accent) !important;
  box-shadow: none !important;
}
.stSelectbox [data-baseweb="select"] > div {
  background: var(--bg-raised) !important;
  border: 1px solid var(--bg-border) !important;
  border-radius: 6px !important;
  color: var(--text-primary) !important;
  box-shadow: none !important;
}
[data-testid="stSlider"] [role="slider"] {
  background: var(--accent) !important;
  box-shadow: none !important;
}
[data-testid="stSlider"] [data-testid="stSliderTrackFill"] {
  background: var(--accent) !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
  background: var(--bg-surface) !important;
  border: 1px solid var(--bg-border) !important;
  border-radius: 8px !important;
}
[data-testid="stDataFrame"] th {
  background: var(--bg-raised) !important;
  font-size: 12px !important;
  font-weight: 500 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.06em !important;
  color: var(--text-muted) !important;
}

/* ── Metric cards ── */
.metric-card {
  background: var(--bg-surface);
  border: 1px solid var(--bg-border);
  border-radius: 10px;
  padding: 20px;
}
.metric-label {
  font-size: 11px;
  font-weight: 500;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: 6px;
}
.metric-value {
  font-size: 28px;
  font-weight: 600;
  color: var(--text-primary);
  line-height: 1.2;
}

/* ── Chat messages ── */
.chat-container { padding-bottom: 160px !important; }
.msg-user {
  background: var(--bg-raised);
  border-left: 2px solid var(--bg-border);
  border-radius: 0 8px 8px 0;
  padding: 12px 16px;
  margin: 8px 0 8px 15%;
  font-size: 14px;
  color: var(--text-primary);
  line-height: 1.75;
  animation: slideIn 0.18s ease-out;
}
.msg-assistant {
  background: transparent;
  border-left: 2px solid var(--current-level-color);
  border-radius: 0;
  padding: 12px 16px;
  margin: 8px 15% 8px 0;
  font-size: 14px;
  color: var(--text-primary);
  line-height: 1.75;
  animation: slideIn 0.18s ease-out;
}
.msg-socratic {
  border-left-color: var(--socratic) !important;
  background: rgba(167,139,250,0.03) !important;
}
.msg-label {
  font-family: var(--font-mono);
  font-size: 10px;
  font-weight: 500;
  letter-spacing: 0.1em;
  color: var(--text-muted);
  text-transform: uppercase;
  margin-bottom: 6px;
}

/* ── Floating chat input ── */
div[data-testid="stChatInput"] {
  position: fixed !important;
  bottom: 30px !important;
  left: 55% !important;
  transform: translateX(-50%) !important;
  max-width: 850px !important;
  width: 75% !important;
  background: var(--bg-surface) !important;
  backdrop-filter: blur(12px) !important;
  -webkit-backdrop-filter: blur(12px) !important;
  border: 1px solid var(--bg-border) !important;
  border-radius: 12px !important;
  padding: 4px 4px 4px 16px !important;
  box-shadow: none !important;
  z-index: 99999 !important;
}
div[data-testid="stChatInput"]:focus-within {
  border-color: var(--accent) !important;
}
div[data-testid="stChatInput"] textarea {
  background: transparent !important;
  border: none !important;
  color: var(--text-primary) !important;
  font-size: 14px !important;
  box-shadow: none !important;
}
div[data-testid="stChatInputContainer"] {
  background-color: transparent !important;
  border: none !important;
}

/* ── Assessment / info banners ── */
.assessment-banner {
  background: var(--accent-dim);
  border: 1px solid rgba(99,102,241,0.3);
  border-left: 3px solid var(--accent);
  border-radius: 8px;
  padding: 12px 16px;
  font-size: 13px;
  color: var(--text-secondary);
  margin-bottom: 12px;
}

/* ── XP / streak display ── */
.xp-row {
  display: flex;
  align-items: center;
  gap: 7px;
  margin-bottom: 2px;
}
.xp-display {
  font-family: var(--font-mono);
  font-size: 19px;
  font-weight: 500;
  color: var(--accent);
  border-bottom: 1px solid var(--accent);
  display: inline-block;
  padding-bottom: 1px;
  letter-spacing: 0.02em;
}
.streak-row {
  display: flex;
  align-items: center;
  gap: 6px;
  margin: 8px 0 5px;
}
.streak-count {
  font-family: var(--font-mono);
  font-size: 15px;
  font-weight: 500;
  color: var(--level-3);
}
.streak-label {
  font-size: 12px;
  color: var(--text-secondary);
}
.streak-bar-bg {
  background: var(--bg-raised);
  border-radius: 3px;
  height: 3px;
  width: 100%;
}
.streak-bar-fill {
  background: var(--level-3);
  border-radius: 3px;
  height: 3px;
}

/* ── Alerts / expanders / captions ── */
[data-testid="stAlert"] {
  background: var(--bg-surface) !important;
  border: 1px solid var(--bg-border) !important;
  border-radius: 8px !important;
  color: var(--text-secondary) !important;
  box-shadow: none !important;
}
[data-testid="stExpander"] {
  background: var(--bg-surface) !important;
  border: 1px solid var(--bg-border) !important;
  border-radius: 8px !important;
  box-shadow: none !important;
}
[data-testid="stExpander"] summary {
  color: var(--text-primary) !important;
  font-size: 14px !important;
}
[data-testid="stCaptionContainer"] p {
  color: var(--text-muted) !important;
  font-size: 12px !important;
}
[data-testid="stRadio"] label {
  color: var(--text-secondary) !important;
  font-size: 14px !important;
}

/* ── Animations ── */
@keyframes slideIn {
  from { opacity: 0; transform: translateY(6px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulse {
  0%, 100% { box-shadow: 0 0 0 0 currentColor; }
  50%       { box-shadow: 0 0 0 4px transparent; }
}
@keyframes xpFlash {
  0%   { color: var(--accent); }
  50%  { color: #fff; }
  100% { color: var(--accent); }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation-duration: 0.01ms !important; }
}
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────

def _init():
    for k, v in {
        "user": None, "session_id": None, "topic": None,
        "page": "login", "messages": [], "quiz_data": None,
        "quiz_result": None, "socratic_mode": False,
        "direct_history": [], "assessment_pending": None,
        "interaction_count": 0,
        "pending_classification": None, "pending_socratic_exit": False,
        "level_up_animation": False, "streak_milestone": None,
        "xp_just_changed": False, "level_just_changed": False,
        "diagnostic_mode": False,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()
init_db()

# ── Helpers ───────────────────────────────────────────────────────────────

_BADGE_LABELS = {1: "L1  Basic", 2: "L2  Medium", 3: "L3  Advanced", 4: "L4  Super"}
_BADGE_VARS   = {1: "var(--level-1)", 2: "var(--level-2)", 3: "var(--level-3)", 4: "var(--level-4)"}

def render_level_badge(level: int, pulse: bool = False) -> str:
    label = _BADGE_LABELS.get(level, f"L{level}")
    c     = _BADGE_VARS.get(level, "var(--text-secondary)")
    anim  = " animation: pulse 1s ease-out;" if pulse else ""
    return (
        f'<span style="font-family:var(--font-mono);font-size:11px;font-weight:500;'
        f'border-radius:4px;padding:2px 8px;border:1px solid {c};color:{c};'
        f'display:inline-block;{anim}">{label}</span>'
    )

level_badge = render_level_badge

def _socratic_badge() -> str:
    return (
        '<span style="font-family:var(--font-mono);font-size:11px;font-weight:500;'
        'border-radius:4px;padding:2px 8px;border:1px solid var(--socratic);'
        'color:var(--socratic);display:inline-block;">Socratic</span>'
    )

def render_message(msg):
    sc = " msg-socratic" if msg.get("socratic") else ""
    if msg["role"] == "user":
        st.markdown(
            f'<div class="msg-user"><div class="msg-label">YOU</div>'
            f'{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        label = "SOCRATIC" if msg.get("socratic") else "TUTOR"
        st.markdown(
            f'<div class="msg-assistant{sc}"><div class="msg-label">{label}</div>'
            f'{msg["content"]}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# LOGIN PAGE
# ─────────────────────────────────────────────────────────────────────────

def page_login():
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown('<div class="alps-brand">ALPS</div>', unsafe_allow_html=True)
        st.markdown('<div class="alps-sub">Adaptive Learning Pathway System &middot; DSA</div>',
                    unsafe_allow_html=True)
        st.markdown("---")

        existing = all_users()
        if existing:
            st.markdown("#### Returning user")
            names = [u["user_id"] for u in existing]
            pick  = st.selectbox("Select username", ["— new user —"] + names,
                                 label_visibility="collapsed")
            if pick != "— new user —":
                if st.button("Continue", type="primary", use_container_width=True):
                    user = get_user(pick)
                    st.session_state.user = user
                    st.session_state.interaction_count = user["total_interactions"]
                    st.session_state.page = "home"
                    st.rerun()
            st.markdown("---")

        st.markdown("#### New user")
        username = st.text_input("Username", placeholder="e.g. haseeb",
                                 label_visibility="collapsed")
        level = st.selectbox("Starting level",
                             [f"{k} — {v}" for k, v in LEVEL_LABELS.items()])
        level_int = int(level.split(" ")[0])

        if st.button("Create & Start", use_container_width=True, type="primary"):
            if username.strip():
                uid  = username.strip().lower()
                user = upsert_user(uid, username.strip(), DEFAULT_SUBJECT, level_int)
                st.session_state.user = user
                st.session_state.interaction_count = 0
                st.session_state.page = "home"
                st.rerun()
            else:
                st.error("Please enter a username.")

# ─────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────

def render_sidebar():
    user = st.session_state.user
    if not user: return
    with st.sidebar:
        st.markdown('<div class="alps-brand">ALPS</div>', unsafe_allow_html=True)
        st.markdown('<div class="alps-sub">Adaptive Learning</div>', unsafe_allow_html=True)
        st.markdown('<hr>', unsafe_allow_html=True)

        pulse = st.session_state.get("level_just_changed", False)
        if pulse:
            st.session_state.level_just_changed = False
        st.markdown(
            f'<div style="font-family:var(--font-body);font-size:13px;font-weight:500;'
            f'color:var(--text-primary);margin-bottom:6px;">{user["user_id"]}</div>'
            + render_level_badge(user["current_level"], pulse=pulse)
            + f'<div style="font-size:12px;color:var(--text-muted);margin-top:6px;">'
            f'{user["subject_area"]}</div>',
            unsafe_allow_html=True)

        st.markdown('<hr>', unsafe_allow_html=True)

        gam = get_user_xp_and_streak(user["user_id"])
        xp_anim = "animation: xpFlash 0.4s ease;" if st.session_state.get("xp_just_changed") else ""
        if st.session_state.get("xp_just_changed"):
            st.session_state.xp_just_changed = False
        streak_val = gam["current_streak"]
        streak_pct = min(streak_val / 30 * 100, 100)

        _bolt = (
            '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" '
            'stroke="var(--accent)" stroke-width="2.5" stroke-linecap="round" '
            'stroke-linejoin="round" style="flex-shrink:0;vertical-align:middle">'
            '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>'
        )
        _flame = (
            '<svg viewBox="0 0 24 24" width="15" height="15" fill="var(--level-3)" '
            'stroke="none" style="flex-shrink:0;vertical-align:middle">'
            '<path d="M12 23a7 7 0 01-5-11.95C8.33 9.67 9.78 8 9 5c3 2 5 5 3 8 '
            '1.33 0 2.67-.33 3.5-1.5C16 13 16 15.5 14.5 17.5 15.5 19 16 21 16 22c'
            '-1.33 1-2.67 1-4 1z"/></svg>'
        )

        st.markdown(
            f'<div class="xp-row">'
            f'{_bolt}'
            f'<span class="xp-display" style="{xp_anim}">{gam["xp"]:,} XP</span>'
            f'</div>'
            f'<div class="streak-row">'
            f'{_flame}'
            f'<span class="streak-count">{streak_val}</span>'
            f'<span class="streak-label">day streak</span>'
            f'</div>'
            f'<div class="streak-bar-bg">'
            f'<div class="streak-bar-fill" style="width:{streak_pct:.0f}%"></div>'
            f'</div>',
            unsafe_allow_html=True)

        if st.session_state.streak_milestone:
            m = st.session_state.streak_milestone
            msgs = {3: "3-day streak — building a habit.",
                    7: "7-day streak — one full week.",
                    14: "14-day streak — unstoppable."}
            st.success(msgs.get(m, f"{m}-day streak."))
            st.session_state.streak_milestone = None

        st.markdown('<hr>', unsafe_allow_html=True)

        st.markdown('<div class="section-header">Navigate</div>', unsafe_allow_html=True)
        for label, pg in [("Home", "home"), ("Chat", "chat"),
                           ("Quiz", "quiz"), ("Stats", "stats")]:
            kind = "primary" if st.session_state.page == pg else "secondary"
            if st.button(label, use_container_width=True, type=kind, key=f"nav_{pg}"):
                st.session_state.page = pg
                st.rerun()

        st.markdown('<hr>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Settings</div>', unsafe_allow_html=True)

        strictness = st.slider("Strictness", 1, 5, user.get("strictness", 3))
        if strictness != user.get("strictness", 3):
            update_strictness(user["user_id"], strictness)
            st.session_state.user["strictness"] = strictness

        new_level = st.selectbox(
            "Switch level",
            options=list(LEVEL_LABELS.keys()),
            format_func=lambda x: f"{x} — {LEVEL_LABELS[x]}",
            index=user["current_level"] - 1,
        )
        if new_level != user["current_level"]:
            update_level(user["user_id"], new_level)
            st.session_state.user["current_level"] = new_level
            st.session_state.level_just_changed = True
            st.rerun()

        if st.session_state.socratic_mode:
            st.markdown(
                f'<div style="margin-top:10px;">{_socratic_badge()}</div>',
                unsafe_allow_html=True)

        st.markdown('<hr>', unsafe_allow_html=True)
        if st.button("Sign out", use_container_width=True, key="signout_btn"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            _init()
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────
# HOME PAGE
# ─────────────────────────────────────────────────────────────────────────

def page_home():
    user   = st.session_state.user
    scores = get_quiz_scores(user["user_id"], n=10)
    acc    = (sum(scores) / len(scores) * 100) if scores else 0
    recent = get_quiz_history(user["user_id"])

    st.markdown(f"## Welcome back, {user['user_id']}")
    st.markdown(
        f"Level: {render_level_badge(user['current_level'])} &nbsp;&nbsp; {user['subject_area']}",
        unsafe_allow_html=True)
    st.markdown("---")

    new_user = user["total_interactions"] == 0
    needs_diagnostic = (
        new_user
        and user["current_level"] > 1
        and not get_quiz_history(user["user_id"])
    )

    if new_user:
        st.markdown(
            f'<div style="background:var(--bg-surface);border:1px solid var(--bg-border);'
            f'border-radius:10px;padding:24px;">'
            f'<div style="font-family:var(--font-body);font-weight:600;font-size:18px;'
            f'color:var(--text-primary);margin-bottom:6px;">Welcome to ALPS</div>'
            f'<div style="font-family:var(--font-body);font-weight:400;font-size:14px;'
            f'color:var(--text-secondary);">Your adaptive DSA tutor. Start a topic to begin.</div>'
            f'</div>',
            unsafe_allow_html=True)
        st.markdown("")

        if needs_diagnostic:
            level_n     = user["current_level"]
            level_label = LEVEL_LABELS[level_n]
            st.markdown(
                f'<div style="background:var(--accent-dim);border-left:3px solid var(--accent);'
                f'border-radius:8px;padding:12px 16px;font-size:13px;'
                f'color:var(--text-secondary);margin-bottom:12px;">'
                f'You selected Level {level_n} — {level_label}. Take a quick 3-question '
                f'diagnostic to confirm your starting level. This helps ALPS calibrate to '
                f'you faster.'
                f'</div>',
                unsafe_allow_html=True)
            dcol1, dcol2 = st.columns(2)
            with dcol1:
                if st.button("Take diagnostic", use_container_width=True, type="primary",
                             key="diagnostic_take"):
                    st.session_state.diagnostic_mode = True
                    st.session_state.page = "quiz"
                    st.rerun()
            with dcol2:
                if st.button(f"Skip, start at Level {level_n}", use_container_width=True,
                             key="diagnostic_skip"):
                    st.session_state.diagnostic_mode = False
                    st.session_state.page = "chat"
                    st.rerun()
        else:
            if st.button("Start studying", use_container_width=True, type="primary",
                         key="welcome_start_studying"):
                st.session_state.page = "chat"; st.rerun()
    else:
        c1, c2, c3, c4 = st.columns(4)
        for col, val, lbl, is_badge in [
            (c1, str(user["total_interactions"]),        "Total Chats",  False),
            (c2, f"{acc:.0f}%",                          "Quiz Accuracy", False),
            (c3, str(len(recent)),                       "Quizzes Taken", False),
            (c4, render_level_badge(user["current_level"]), "Level",      True),
        ]:
            with col:
                inner = (f'<div style="margin-top:8px;">{val}</div>' if is_badge
                         else f'<div class="metric-value">{val}</div>')
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-label">{lbl}</div>'
                    f'{inner}</div>',
                    unsafe_allow_html=True)

    st.markdown("---")
    ca, cb = st.columns(2)
    with ca:
        if st.button("Start studying", use_container_width=True, type="primary"):
            st.session_state.page = "chat"; st.rerun()
    with cb:
        if st.button("Take a quiz", use_container_width=True):
            st.session_state.page = "quiz"; st.rerun()

    if recent:
        st.markdown('<div class="section-header">Recent Quizzes</div>',
                    unsafe_allow_html=True)
        for r in recent[:5]:
            pct = r["score"] * 100
            score_color = ("var(--level-1)" if pct >= 70
                           else "var(--level-4)" if pct < 50
                           else "var(--text-secondary)")
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:12px;'
                f'padding:10px 0;border-bottom:1px solid var(--bg-border);">'
                f'<span style="flex:1;font-size:14px;color:var(--text-primary);">'
                f'{r["topic"]}</span>'
                f'<span style="font-size:14px;font-weight:500;color:{score_color};">'
                f'{pct:.0f}%</span>'
                f'{render_level_badge(r["level_at_time"])}'
                f'<span style="font-size:12px;color:var(--text-muted);">'
                f'{r["timestamp"][:10]}</span>'
                f'</div>',
                unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# EVALUATION PIPELINE (background, sampled — never surfaces to the UI)
# ─────────────────────────────────────────────────────────────────────────

def find_reference(topic: str) -> str | None:
    topic_lower = topic.lower()
    for entry in REFERENCE_ANSWERS:
        if entry["topic"].lower() in topic_lower:
            return entry["reference_answer"]
    return None


def run_evaluation_pipeline(response_text, topic, user_level, user_id, session_id):
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row

        rouge_l = None
        bertscore_f1 = None
        reference_text = find_reference(topic)
        if reference_text:
            metrics = evaluate_response(response_text, reference_text)
            rouge_l = metrics["rougeL"]
            bertscore_f1 = metrics["bertscore_f1"]
            logger.info(
                "Eval metrics user=%s session=%s rougeL=%s bertscore_f1=%s",
                user_id, session_id, rouge_l, bertscore_f1,
            )

        row = conn.execute(
            "SELECT total_interactions FROM users WHERE user_id=?", (user_id,)
        ).fetchone()
        total_interactions = row["total_interactions"] if row else 0

        if total_interactions % JUDGE_SAMPLE_RATE == 0:
            scores = score_response(response_text, user_level, topic)
            if scores is not None:
                final_scores = resolve_scores(scores, scores)
                save_evaluation(
                    user_id, session_id, response_text, final_scores, conn,
                    rouge_l=rouge_l, bertscore_f1=bertscore_f1,
                )
    except Exception as e:
        logger.error(f"Eval pipeline failed: {e}")
        return
    finally:
        if conn is not None:
            conn.close()

# ─────────────────────────────────────────────────────────────────────────
# CHAT PAGE
# ─────────────────────────────────────────────────────────────────────────

def page_chat():
    user = st.session_state.user
    _inject_level_color(user["current_level"])

    if st.session_state.level_up_animation:
        st.balloons()
        st.success("Level up! You've advanced to a new difficulty — keep it up!")
        st.session_state.level_just_changed = True
        st.session_state.level_up_animation = False

    # ── Topic selection ───────────────────────────────────────────────────
    if not st.session_state.topic:
        st.markdown("## Study Chat")
        st.markdown("---")
        st.markdown("### What topic do you want to study today?")
        st.markdown(
            "Enter a DSA topic you want to learn or get help with. "
            "Your entire chat session will be focused on this topic."
        )
        st.markdown("")
        st.markdown('<div class="section-header">Quick picks</div>', unsafe_allow_html=True)
        cols = st.columns(4)
        quick = ["Arrays", "Linked Lists", "Binary Search",
                 "Sorting Algorithms", "Binary Trees", "Dynamic Programming",
                 "Graphs", "Hash Tables"]
        for i, suggestion in enumerate(quick):
            with cols[i % 4]:
                if st.button(suggestion, use_container_width=True):
                    sid = str(uuid.uuid4())
                    create_session(sid, user["user_id"], suggestion)
                    s_info = update_streak(user["user_id"])
                    if s_info["is_new_day"]:
                        ms = _streak_milestone(s_info["current_streak"])
                        if ms:
                            st.session_state.streak_milestone = ms
                    st.session_state.session_id = sid
                    st.session_state.topic      = suggestion
                    st.session_state.messages   = []
                    st.rerun()
        st.markdown("")
        st.markdown("**Or type your own:**")
        c1, c2 = st.columns([4, 1])
        with c1:
            topic_input = st.text_input(
                "Topic", placeholder="e.g. AVL Trees, Dijkstra's Algorithm...",
                label_visibility="collapsed",
            )
        with c2:
            start = st.button("Start", type="primary", use_container_width=True)
        if start and topic_input.strip():
            sid = str(uuid.uuid4())
            create_session(sid, user["user_id"], topic_input.strip())
            s_info = update_streak(user["user_id"])
            if s_info["is_new_day"]:
                ms = _streak_milestone(s_info["current_streak"])
                if ms:
                    st.session_state.streak_milestone = ms
            st.session_state.session_id = sid
            st.session_state.topic      = topic_input.strip()
            st.session_state.messages   = []
            st.rerun()
        elif start:
            st.warning("Please enter a topic or pick one above.")
        return

    topic = st.session_state.topic
    sid   = st.session_state.session_id
    # Always read topic from DB so stale session_state cannot cause drift
    topic = get_session_topic(sid) or topic

    if "generating" not in st.session_state:
        st.session_state.generating = False

    # ── Header ────────────────────────────────────────────────────────────
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"## {topic} Studio")
        st.markdown(render_level_badge(user["current_level"]), unsafe_allow_html=True)
    with col2:
        if st.button("New topic", disabled=st.session_state.generating):
            st.session_state.topic      = None
            st.session_state.session_id = None
            st.session_state.messages   = []
            st.session_state.generating = False
            st.rerun()

    st.markdown("---")

    # ── Message container ─────────────────────────────────────────────────
    chat_block = st.container()

    with chat_block:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for msg in st.session_state.messages:
            render_message(msg)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Assessment banner (below latest message, above input) ─────────────
    if st.session_state.assessment_pending and st.session_state.assessment_pending["recommendation"] != "MAINTAIN":
        result = st.session_state.assessment_pending
        rec    = result["recommendation"]
        st.markdown(
            f'<div class="assessment-banner">'
            f'<strong style="color:var(--text-primary)">Adaptive Suggestion</strong><br>'
            f'{level_change_message(rec, user["current_level"])}'
            f'</div>',
            unsafe_allow_html=True)
        cy, cn = st.columns(2)
        with cy:
            if st.button("Yes, update level", disabled=st.session_state.generating):
                new = apply_recommendation(user["user_id"], user["current_level"],
                                           result["recommendation"])
                if result["recommendation"] == "INCREASE":
                    award_xp(user["user_id"], "LEVEL_UP")
                    st.session_state.level_up_animation = True
                st.session_state.user["current_level"] = new
                st.session_state.assessment_pending    = None
                st.rerun()
        with cn:
            if st.button("Stay at current level", disabled=st.session_state.generating):
                st.session_state.assessment_pending = None
                st.rerun()

    # ── First-session onboarding hint (shown once, before the first message) ──
    if user["total_interactions"] == 0 and not get_session_history(sid, n=1):
        if len(get_quiz_history(user["user_id"])) == 1:
            hint_body = (
                f'Diagnostic complete. Ask me anything about {topic} — '
                f'ALPS is calibrated to your level and will adapt as we go.'
            )
        else:
            hint_body = (
                f'Ask me anything about {topic}. I will adapt to your level as we go.<br>'
                f'Try: explain {topic} to me, or give me a beginner example of {topic}.'
            )
        st.markdown(
            f'<div style="background:var(--accent-dim);border:1px solid var(--accent);'
            f'border-radius:8px;padding:12px 16px;font-size:13px;'
            f'color:var(--text-secondary);margin-bottom:12px;">'
            f'{hint_body}'
            f'</div>',
            unsafe_allow_html=True)

    # ── Dynamic input ─────────────────────────────────────────────────────
    placeholder_text = "Generating response..." if st.session_state.generating else f"Ask anything about {topic}..."
    user_query = st.chat_input(placeholder_text, disabled=st.session_state.generating)

    if user_query and not st.session_state.generating:
        query = user_query.strip()

        st.session_state.generating = True

        prev_socratic  = st.session_state.socratic_mode
        classification = classify_query(query)
        is_direct      = classification == "DIRECT_ANSWER_REQUEST"
        st.session_state.direct_history.append(is_direct)
        socratic = check_socratic_mode(st.session_state.direct_history)
        st.session_state.socratic_mode          = socratic
        st.session_state.pending_classification  = classification
        st.session_state.pending_socratic_exit   = prev_socratic and not socratic
        set_socratic(sid, socratic)

        add_message(user["user_id"], sid, "user", query,
                    user["current_level"], socratic, classification)
        st.session_state.messages.append({"role": "user", "content": query, "socratic": False})

        st.rerun()

    # ── Generation pipeline ───────────────────────────────────────────────
    if st.session_state.generating and st.session_state.messages:
        if st.session_state.messages[-1]["role"] == "user":
            query = st.session_state.messages[-1]["content"]

            with chat_block:
                sc_cls   = " msg-socratic" if st.session_state.socratic_mode else ""
                label    = "SOCRATIC" if st.session_state.socratic_mode else "TUTOR"
                history  = get_session_history(sid, n=20)

                stream_placeholder = st.empty()

                stream_gen = generate_response(
                    query         = query,
                    level         = user["current_level"],
                    socratic_mode = st.session_state.socratic_mode,
                    strictness    = user.get("strictness", 3),
                    topic         = topic,
                    subject_area  = user["subject_area"],
                    chat_history  = history,
                    stream        = True,
                )

                full_response = ""
                for chunk in stream_gen:
                    piece = chunk["message"]["content"]
                    full_response += piece
                    stream_placeholder.markdown(
                        f'<div class="msg-assistant{sc_cls}">'
                        f'<div class="msg-label">{label}</div>'
                        f'{full_response}▌</div>',
                        unsafe_allow_html=True,
                    )

                stream_placeholder.markdown(
                    f'<div class="msg-assistant{sc_cls}">'
                    f'<div class="msg-label">{label}</div>'
                    f'{full_response}</div>',
                    unsafe_allow_html=True,
                )

                has_scaffolding, gives_away = check_pedagogical_output(full_response)
                if gives_away and not has_scaffolding:
                    full_response = generate_response(
                        query        = (query + "\n\nRevise your response to include "
                                        "scaffolding. Do not give the complete answer "
                                        "directly. Ask the learner a guiding question."),
                        level        = user["current_level"],
                        socratic_mode= st.session_state.socratic_mode,
                        strictness   = user.get("strictness", 3),
                        topic        = topic,
                        subject_area = user["subject_area"],
                        chat_history = history,
                        stream       = False,
                    )
                    stream_placeholder.markdown(
                        f'<div class="msg-assistant{sc_cls}">'
                        f'<div class="msg-label">{label}</div>'
                        f'{full_response}</div>',
                        unsafe_allow_html=True,
                    )

                ppl = compute_perplexity(full_response)
                if ppl is not None and ppl > 200:
                    full_response = generate_response(
                        query        = (query + "\n\nYour previous response was unclear "
                                        "or incoherent. Please rewrite it clearly and "
                                        "concisely for a learner studying DSA."),
                        level        = user["current_level"],
                        socratic_mode= st.session_state.socratic_mode,
                        strictness   = user.get("strictness", 3),
                        topic        = topic,
                        subject_area = user["subject_area"],
                        chat_history = history,
                        stream       = False,
                    )
                    stream_placeholder.markdown(
                        f'<div class="msg-assistant{sc_cls}">'
                        f'<div class="msg-label">{label}</div>'
                        f'{full_response}</div>',
                        unsafe_allow_html=True,
                    )

                ref_entry = REFERENCE_BY_TOPIC.get(topic.lower())
                if ref_entry:
                    bert = compute_bertscore(full_response, ref_entry["reference_answer"])
                    if bert["f1"] is not None and not passes_tier2(bert["f1"]):
                        full_response = generate_response(
                            query        = (query + "\n\nYour previous response did not "
                                            "sufficiently cover the key concepts of this "
                                            "topic. Please rewrite it to better address "
                                            "the core ideas a learner needs to understand."),
                            level        = user["current_level"],
                            socratic_mode= st.session_state.socratic_mode,
                            strictness   = user.get("strictness", 3),
                            topic        = topic,
                            subject_area = user["subject_area"],
                            chat_history = history,
                            stream       = False,
                        )
                        stream_placeholder.markdown(
                            f'<div class="msg-assistant{sc_cls}">'
                            f'<div class="msg-label">{label}</div>'
                            f'{full_response}</div>',
                            unsafe_allow_html=True,
                        )

            add_message(user["user_id"], sid, "assistant", full_response,
                        user["current_level"], st.session_state.socratic_mode, None)

            eval_thread = threading.Thread(
                target=run_evaluation_pipeline,
                args=(
                    full_response,
                    topic,
                    user["current_level"],
                    user["user_id"],
                    sid,
                ),
                daemon=True
            )
            eval_thread.start()
            # do NOT call eval_thread.join() — must be non-blocking

            st.session_state.messages.append({"role": "assistant", "content": full_response, "socratic": st.session_state.socratic_mode})

            bump_interactions(user["user_id"])
            st.session_state.interaction_count += 1
            st.session_state.user["total_interactions"] += 1

            award_xp(user["user_id"],
                     st.session_state.pending_classification or "LEARNING_QUERY")
            st.session_state.xp_just_changed = True
            if st.session_state.pending_socratic_exit:
                award_xp(user["user_id"], "SOCRATIC_EXIT")
                st.session_state.pending_socratic_exit = False

            if should_assess(st.session_state.interaction_count):
                result = run_assessment(user["user_id"], user["current_level"])
                if result:
                    st.session_state.assessment_pending = result

            st.session_state.generating = False
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────
# QUIZ PAGE
# ─────────────────────────────────────────────────────────────────────────

def page_quiz():
    user = st.session_state.user

    if st.session_state.level_up_animation:
        st.balloons()
        st.success("Level up! You've advanced to a new difficulty — keep it up!")
        st.session_state.level_just_changed = True
        st.session_state.level_up_animation = False

    st.markdown("## Quiz")

    diagnostic_mode = st.session_state.get("diagnostic_mode") is True and user["current_level"] > 1

    if diagnostic_mode and not st.session_state.quiz_result:
        st.markdown(
            '<div style="background:var(--accent-dim);border-left:3px solid var(--accent);'
            'border-radius:8px;padding:12px 16px;font-size:13px;color:var(--text-secondary);'
            'margin-bottom:12px;">Diagnostic Quiz — 3 questions to confirm your starting level.</div>',
            unsafe_allow_html=True)

    # ── Show result ───────────────────────────────────────────────────────
    if st.session_state.quiz_result:
        r = st.session_state.quiz_result

        if r.get("diagnostic_decision"):
            decision    = r["diagnostic_decision"]
            new_level   = r["diagnostic_new_level"]
            level_label = LEVEL_LABELS[new_level]
            if decision == "DECREASE":
                diag_msg = (f"Starting you at Level {new_level} — {level_label} based on "
                            f"your diagnostic. ALPS will adjust as you progress.")
            else:
                diag_msg = f"Level {new_level} — {level_label} confirmed. Let us get started."
            st.markdown(
                f'<div style="background:var(--accent-dim);border-left:3px solid var(--accent);'
                f'border-radius:8px;padding:12px 16px;font-size:13px;color:var(--text-secondary);'
                f'margin-bottom:12px;">{diag_msg}</div>',
                unsafe_allow_html=True)
            st.session_state.diagnostic_mode = False
            if st.button("Start studying", type="primary", use_container_width=True,
                         key="diagnostic_start_studying"):
                st.session_state.quiz_result = None
                st.session_state.page = "chat"
                st.rerun()
            return

        pct = r["score"] * 100
        st.markdown(f"### {r['topic']}")
        st.markdown(
            f'<div style="font-size:48px;font-weight:600;color:var(--text-primary);line-height:1;">'
            f'{pct:.0f}</div>'
            f'<div style="font-size:14px;color:var(--text-secondary);margin:6px 0 16px;">'
            f'% &mdash; {r["correct"]} of {r["total"]} correct &nbsp;'
            f'{render_level_badge(r["level"])}</div>',
            unsafe_allow_html=True)

        if pct == 100:  st.success("Perfect score!")
        elif pct >= 70: st.success("Great job!")
        elif pct >= 50: st.warning("Review and try again.")
        else:           st.error("More study needed.")

        zpd = r.get("zpd")
        if zpd and zpd["recommendation"] != "MAINTAIN":
            st.markdown("---")
            st.markdown(
                f'<div class="assessment-banner">'
                f'<strong style="color:var(--text-primary)">Adaptive Suggestion</strong><br>'
                f'{level_change_message(zpd["recommendation"], user["current_level"])}'
                f'</div>',
                unsafe_allow_html=True)
            st.caption(f"Why: {zpd['reasoning']}")
            cy, cn = st.columns(2)
            with cy:
                if st.button("Yes, update level", key="zpd_inline_yes"):
                    new = apply_recommendation(user["user_id"],
                                               user["current_level"],
                                               zpd["recommendation"])
                    if zpd["recommendation"] == "INCREASE":
                        award_xp(user["user_id"], "LEVEL_UP")
                        st.session_state.level_up_animation = True
                    st.session_state.user["current_level"] = new
                    st.session_state.quiz_result = None
                    st.session_state.quiz_data   = None
                    st.rerun()
            with cn:
                if st.button("Stay at current level", key="zpd_inline_no"):
                    st.session_state.quiz_result = None
                    st.session_state.quiz_data   = None
                    st.rerun()

        st.markdown("---")
        for i, q in enumerate(r["breakdown"], 1):
            correct_label = "Correct" if q["is_correct"] else "Incorrect"
            with st.expander(f"Q{i} of {len(r['breakdown'])}: {q['question']} — {correct_label}"):
                for j, opt in enumerate(q["options"]):
                    tag = ""
                    if j == q["correct_answer"]:
                        tag = "  (correct answer)"
                    if j == q["user_answer"] and not q["is_correct"]:
                        tag = "  (your answer)"
                    prefix = "> " if j == q["user_answer"] else "  "
                    st.markdown(f"{prefix}{opt}{tag}")
                st.info(q["explanation"])

        if st.button("Take another quiz", type="primary"):
            st.session_state.quiz_result = None
            st.session_state.quiz_data   = None
            st.rerun()
        return

    # ── Active quiz ───────────────────────────────────────────────────────
    if st.session_state.quiz_data:
        qdata     = st.session_state.quiz_data
        questions = qdata["questions"]
        st.markdown(
            f"### {qdata['topic']} &nbsp; {render_level_badge(qdata['level'])}",
            unsafe_allow_html=True)
        st.markdown(f"{len(questions)} questions — answer all to submit")
        st.markdown("---")

        answers      = []
        all_answered = True
        for i, q in enumerate(questions):
            st.markdown(
                f'<div style="font-size:11px;font-family:var(--font-mono);'
                f'color:var(--text-muted);margin-bottom:4px;">Q{i+1} of {len(questions)}</div>'
                f'<div style="font-size:15px;font-weight:500;color:var(--text-primary);'
                f'margin-bottom:8px;">{q["question"]}</div>',
                unsafe_allow_html=True)
            choice = st.radio(
                f"q{i}", options=list(range(4)),
                format_func=lambda x, q=q: str(q["options"][x]),
                index=None, label_visibility="collapsed",
                key=f"qq_{i}",
            )
            if choice is None: all_answered = False
            answers.append(choice)
            st.markdown("")

        st.markdown("---")
        if st.button("Submit", type="primary",
                     disabled=not all_answered, use_container_width=True):
            result = submit_quiz(qdata, answers)
            award_xp(user["user_id"], "QUIZ_CORRECT", result["correct"])
            award_xp(user["user_id"], "QUIZ_COMPLETED")
            st.session_state.xp_just_changed = True
            if diagnostic_mode:
                pct = result["score"] * 100
                decision = "DECREASE" if pct < 40 else "MAINTAIN"
                new_level = apply_recommendation(user["user_id"], user["current_level"], decision)
                if new_level != user["current_level"]:
                    st.session_state.user["current_level"] = new_level
                result["diagnostic_decision"]  = decision
                result["diagnostic_new_level"] = new_level
            else:
                result["zpd"] = run_assessment(user["user_id"], user["current_level"])
            st.session_state.quiz_result = result
            st.session_state.quiz_data   = None
            st.rerun()
        if not all_answered:
            st.caption("Answer all questions to submit.")
        return

    # ── Quiz setup ────────────────────────────────────────────────────────
    if diagnostic_mode:
        diag_topic_by_level = {2: "Arrays", 3: "Binary Search", 4: "Dynamic Programming"}
        topic_q = diag_topic_by_level.get(user["current_level"], "Arrays")
        n_q     = 3
        level_q = user["current_level"]

        sid = st.session_state.session_id or str(uuid.uuid4())
        if not st.session_state.session_id:
            create_session(sid, user["user_id"], topic_q)
            st.session_state.session_id = sid
        with st.spinner(f"Generating {n_q} diagnostic questions on {topic_q}..."):
            quiz = run_quiz_session(user["user_id"], sid, topic_q, level_q, n_q)
        if quiz:
            st.session_state.quiz_data = quiz
            st.rerun()
        else:
            st.error("Could not generate diagnostic quiz. Check your API key is set.")
        return

    st.markdown("Generate an AI quiz on any DSA topic.")
    topic_q = st.text_input("Topic", placeholder="e.g. Merge Sort",
                             label_visibility="collapsed")
    n_q     = st.slider("Number of questions", 3, 10, 5)
    level_q = st.selectbox("Difficulty",
                           options=list(LEVEL_LABELS.keys()),
                           format_func=lambda x: f"{x} — {LEVEL_LABELS[x]}",
                           index=user["current_level"] - 1)

    if st.button("Generate Quiz", type="primary", use_container_width=True):
        if topic_q.strip():
            sid = st.session_state.session_id or str(uuid.uuid4())
            if not st.session_state.session_id:
                create_session(sid, user["user_id"], topic_q.strip())
                st.session_state.session_id = sid
            with st.spinner(f"Generating {n_q} questions on {topic_q}..."):
                quiz = run_quiz_session(user["user_id"], sid,
                                        topic_q.strip(), level_q, n_q)
            if quiz:
                st.session_state.quiz_data = quiz
                st.rerun()
            else:
                st.error("Could not generate quiz. Check your API key is set.")
        else:
            st.warning("Please enter a topic.")

# ─────────────────────────────────────────────────────────────────────────
# STATS PAGE
# ─────────────────────────────────────────────────────────────────────────

def page_stats():
    user    = st.session_state.user
    scores  = get_quiz_scores(user["user_id"], n=20)
    history = get_quiz_history(user["user_id"])
    acc     = (sum(scores) / len(scores) * 100) if scores else 0

    st.markdown(f"## Stats — {user['user_id']}")

    new_user = user["total_interactions"] == 0 or not history
    if new_user:
        st.info("No stats yet. Complete a few topics and quizzes to see your progress.")
        return

    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl, is_badge in [
        (c1, str(user["total_interactions"]),           "Total Chats",  False),
        (c2, f"{acc:.0f}%",                             "Quiz Accuracy", False),
        (c3, str(len(history)),                         "Quizzes Taken", False),
        (c4, render_level_badge(user["current_level"]), "Level",         True),
    ]:
        with col:
            inner = (f'<div style="margin-top:8px;">{val}</div>' if is_badge
                     else f'<div class="metric-value">{val}</div>')
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">{lbl}</div>'
                f'{inner}</div>',
                unsafe_allow_html=True)

    st.markdown("---")

    if history:
        import pandas as pd
        df       = pd.DataFrame(history)
        df["pct"] = df["score"] * 100

        st.markdown('<div class="section-header">Score Trend</div>',
                    unsafe_allow_html=True)
        st.line_chart(df[["pct"]].rename(columns={"pct": "Score (%)"}),
                      color="#6366f1")

        st.markdown('<div class="section-header">By Topic</div>',
                    unsafe_allow_html=True)
        topic_avg = (df.groupby("topic")["pct"]
                     .mean().round(1)
                     .sort_values(ascending=False)
                     .reset_index())
        topic_avg.columns = ["Topic", "Avg Score (%)"]
        st.dataframe(topic_avg, use_container_width=True, hide_index=True)

        st.markdown('<div class="section-header">Full History</div>',
                    unsafe_allow_html=True)
        disp = df[["topic","correct_q","total_q","pct","level_at_time","timestamp"]].copy()
        disp.columns = ["Topic","Correct","Total","Score (%)","Level","Date"]
        disp["Level"] = disp["Level"].map(LEVEL_LABELS)
        disp["Date"]  = disp["Date"].str[:10]
        st.dataframe(disp, use_container_width=True, hide_index=True)
    else:
        st.info("No quizzes yet. Take your first quiz to see stats here.")

    st.markdown("---")
    st.markdown(f"**Username:** {user['user_id']}")
    st.markdown(f"**Subject:** {user['subject_area']}")
    st.markdown(f"**Member since:** {user['created_at'][:10]}")

# ─────────────────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────────────────

if not st.session_state.user or st.session_state.page == "login":
    page_login()
else:
    render_sidebar()
    p = st.session_state.page
    if   p == "home":  page_home()
    elif p == "chat":  page_chat()
    elif p == "quiz":  page_quiz()
    elif p == "stats": page_stats()
    else:
        st.session_state.page = "home"
        st.rerun()
