import uuid
import streamlit as st

from database.db import (
    init_db, upsert_user, get_user, all_users,
    create_session, add_message, get_history,
    get_quiz_history, get_quiz_scores,
    bump_interactions, update_strictness,
    update_level, set_socratic,
)
from llm.engine import generate_response
from adaptivity.quiz_manager import run_quiz_session, submit_quiz
from adaptivity.level_assessor import (
    should_assess, run_assessment,
    apply_recommendation, level_change_message,
)
from config import LEVEL_LABELS, LEVEL_COLORS, DEFAULT_SUBJECT

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
    if len(history) < 3: return False
    ratio = sum(history[-10:]) / min(len(history), 10)
    return ratio >= 0.30

# ── Page config ───────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ALPS",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0f1117; color: #e8eaf0; }

[data-testid="stSidebar"] {
    background: #161b27 !important;
    border-right: 1px solid #2a2f3e;
}

.alps-brand {
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem; font-weight: 700;
    color: #7dd3fc; letter-spacing: 0.12em;
}
.alps-sub {
    font-size: 0.72rem; color: #64748b;
    letter-spacing: 0.08em; text-transform: uppercase;
    margin-bottom: 1.5rem;
}

.level-badge {
    display: inline-block; padding: 2px 10px;
    border-radius: 999px; font-size: 0.75rem;
    font-weight: 600; letter-spacing: 0.05em;
    font-family: 'Space Mono', monospace;
}

/* Chat Bubbles Optimization */
.chat-container {
    padding-bottom: 160px !important; /* Premium cushion space so input doesn't block text */
}
.msg-user {
    background: #1e3a5f; border: 1px solid #2563eb33;
    border-radius: 12px 12px 2px 12px;
    padding: 0.75rem 1rem; margin: 0.5rem 0; margin-left: 15%;
    color: #bfdbfe; font-size: 0.92rem; line-height: 1.6;
}
.msg-assistant {
    background: #1a1f2e; border: 1px solid #334155;
    border-radius: 2px 12px 12px 12px;
    padding: 0.75rem 1rem; margin: 0.5rem 0; margin-right: 15%;
    color: #e2e8f0; font-size: 0.92rem; line-height: 1.6;
}
.msg-socratic { border-color: #f59e0b55 !important; background: #1c1a10 !important; }
.msg-label {
    font-size: 0.68rem; font-family: 'Space Mono', monospace;
    letter-spacing: 0.08em; margin-bottom: 4px; opacity: 0.6;
}

.metric-card {
    background: #161b27; border: 1px solid #2a2f3e;
    border-radius: 10px; padding: 1rem 1.2rem; text-align: center;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem; font-weight: 700; color: #7dd3fc;
}
.metric-label {
    font-size: 0.72rem; color: #64748b;
    text-transform: uppercase; letter-spacing: 0.08em; margin-top: 2px;
}

.section-header {
    font-family: 'Space Mono', monospace; font-size: 0.8rem;
    letter-spacing: 0.12em; text-transform: uppercase; color: #7dd3fc;
    border-bottom: 1px solid #2a2f3e; padding-bottom: 0.4rem;
    margin-bottom: 1rem; margin-top: 1.5rem;
}


.stButton > button {
    background: #1e3a5f !important; color: #7dd3fc !important;
    border: 1px solid #2563eb55 !important; border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important; letter-spacing: 0.05em !important;
}
.stButton > button:hover {
    background: #2563eb !important; color: #fff !important;
}
.stButton > button[kind="primary"] {
    background: #2563eb !important; color: #fff !important;
    border-color: #2563eb !important;
}
#MainMenu, footer, header { visibility: hidden; }

/* ── FLOATING GLASS PILL INPUT BOX ── */
div[data-testid="stChatInput"] {
    position: fixed !important;
    bottom: 30px !important;
    left: 55% !important; /* Offset slightly to account for the sidebar layout space */
    transform: translateX(-50%) !important;
    max-width: 850px !important;
    width: 75% !important;
    background: rgba(22, 27, 39, 0.85) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(125, 211, 252, 0.2) !important;
    border-radius: 999px !important;
    padding: 4px 16px !important;
    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.6) !important;
    z-index: 99999 !important;
}

div[data-testid="stChatInput"] textarea {
    background: transparent !important;
    border: none !important;
    color: #e2e8f0 !important;
    font-size: 0.95rem !important;
}

div[data-testid="stChatInputContainer"] {
    background-color: transparent !important;
    border: none !important;
}


/* Force sidebar always open, remove collapse button entirely */
[data-testid="stSidebar"] {
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
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()
init_db()

# ── Helpers ───────────────────────────────────────────────────────────────

def level_badge(level):
    color = LEVEL_COLORS.get(level, "#94a3b8")
    label = LEVEL_LABELS.get(level, "?")
    return (f'<span class="level-badge" style="background:{color}22;'
            f'color:{color};border:1px solid {color}55">{label}</span>')

def render_message(msg):
    sc = " msg-socratic" if msg.get("socratic") else ""
    if msg["role"] == "user":
        st.markdown(
            f'<div class="msg-user"><div class="msg-label">YOU</div>'
            f'{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        label = "🔄 SOCRATIC MODE" if msg.get("socratic") else "TUTOR"
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
        st.markdown('<div class="alps-sub">Adaptive Learning Pathway System · DSA</div>',
                    unsafe_allow_html=True)
        st.markdown("---")

        existing = all_users()
        if existing:
            st.markdown("#### Returning user")
            names = [u["user_id"] for u in existing]
            pick  = st.selectbox("Select username", ["— new user —"] + names,
                                 label_visibility="collapsed")
            if pick != "— new user —":
                if st.button("Continue →", type="primary", use_container_width=True):
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

        if st.button("Create & Start →", use_container_width=True, type="primary"):
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
        st.markdown(f"**{user['user_id']}**")
        st.markdown(level_badge(user["current_level"]), unsafe_allow_html=True)
        st.caption(f"📚 {user['subject_area']}")
        st.markdown("---")

        st.markdown('<div class="section-header">Navigate</div>', unsafe_allow_html=True)
        for label, pg in [("🏠  Home","home"),("💬  Chat","chat"),
                           ("🧠  Quiz","quiz"),("📊  Stats","stats")]:
            kind = "primary" if st.session_state.page == pg else "secondary"
            if st.button(label, use_container_width=True, type=kind):
                st.session_state.page = pg
                st.rerun()

        st.markdown("---")
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
            st.rerun()

        st.markdown("---")
        if st.session_state.socratic_mode:
            st.warning("🔄 Socratic mode active")

        if st.button("🚪 Log out", use_container_width=True):
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

    st.markdown(f"## Welcome back, {user['user_id']} 👋")
    st.markdown(
        f"Level: {level_badge(user['current_level'])} · {user['subject_area']}",
        unsafe_allow_html=True)
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in [
        (c1, user["total_interactions"], "Total Chats"),
        (c2, f"{acc:.0f}%",             "Quiz Accuracy"),
        (c3, len(recent),               "Quizzes Taken"),
        (c4, LEVEL_LABELS[user["current_level"]], "Level"),
    ]:
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value">{val}</div>'
                f'<div class="metric-label">{lbl}</div></div>',
                unsafe_allow_html=True)

    st.markdown("---")
    ca, cb = st.columns(2)
    with ca:
        if st.button("💬 Start studying", use_container_width=True, type="primary"):
            st.session_state.page = "chat"; st.rerun()
    with cb:
        if st.button("🧠 Take a quiz", use_container_width=True):
            st.session_state.page = "quiz"; st.rerun()

    if recent:
        st.markdown('<div class="section-header">Recent Quizzes</div>',
                    unsafe_allow_html=True)
        for r in recent[:5]:
            pct  = r["score"] * 100
            icon = "🏆" if pct==100 else "✅" if pct>=70 else "📚"
            st.markdown(
                f"{icon} **{r['topic']}** — {r['correct_q']}/{r['total_q']} "
                f"({pct:.0f}%) · {level_badge(r['level_at_time'])} · "
                f"<small style='color:#64748b'>{r['timestamp'][:10]}</small>",
                unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# CHAT PAGE (Instant Lock & Secure State Flow)
# ─────────────────────────────────────────────────────────────────────────

def page_chat():
    user = st.session_state.user

    # ── Topic selection ───────────────────────────────────────────────────
    if not st.session_state.topic:
        st.markdown("## 💬 Study Chat")
        st.markdown("---")
        st.markdown("### 📌 What topic do you want to study today?")
        st.markdown(
            "Enter a **DSA topic** you want to learn or get help with. "
            "Your entire chat session will be focused on this topic."
        )
        st.markdown("")
        st.markdown("**Quick picks:**")
        cols = st.columns(4)
        quick = ["Arrays", "Linked Lists", "Binary Search",
                 "Sorting Algorithms", "Binary Trees", "Dynamic Programming",
                 "Graphs", "Hash Tables"]
        for i, suggestion in enumerate(quick):
            with cols[i % 4]:
                if st.button(suggestion, use_container_width=True):
                    sid = str(uuid.uuid4())
                    create_session(sid, user["user_id"], suggestion)
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
            start = st.button("Start →", type="primary", use_container_width=True)
        if start and topic_input.strip():
            sid = str(uuid.uuid4())
            create_session(sid, user["user_id"], topic_input.strip())
            st.session_state.session_id = sid
            st.session_state.topic      = topic_input.strip()
            st.session_state.messages   = []
            st.rerun()
        elif start:
            st.warning("Please enter a topic or pick one above.")
        return

    topic = st.session_state.topic
    sid   = st.session_state.session_id

    # Initialize a clean lock flag in state if it doesn't exist
    if "generating" not in st.session_state:
        st.session_state.generating = False

    # ── Header ────────────────────────────────────────────────────────────
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"## 💬 {topic} Studio")
        st.markdown(level_badge(user["current_level"]), unsafe_allow_html=True)
    with col2:
        if st.button("📝 New topic", disabled=st.session_state.generating):
            st.session_state.topic      = None
            st.session_state.session_id = None
            st.session_state.messages   = []
            st.session_state.generating = False
            st.rerun()

    st.markdown("---")

    # ── Assessment banner ─────────────────────────────────────────────────
    # if st.session_state.assessment_pending:
    # if st.session_state.assessment_pending and st.session_state.assessment_pending["recommendation"] != "MAINTAIN":
    #     result = st.session_state.assessment_pending
    #     st.info(level_change_message(result["recommendation"], user["current_level"]))
    #     cy, cn = st.columns(2)
    #     with cy:
    #         if st.button("✅ Yes, update level", disabled=st.session_state.generating):
    #             new = apply_recommendation(user["user_id"], user["current_level"],
    #                                        result["recommendation"])
    #             st.session_state.user["current_level"] = new
    #             st.session_state.assessment_pending    = None
    #             st.rerun()
    #     with cn:
    #         if st.button("❌ Stay at current level", disabled=st.session_state.generating):
    #             st.session_state.assessment_pending = None
    #             st.rerun()


    # ── Assessment banner ─────────────────────────────────────────────────
    if st.session_state.assessment_pending and st.session_state.assessment_pending["recommendation"] != "MAINTAIN":
        result = st.session_state.assessment_pending
        rec = result["recommendation"]
        accent = "#fb923c" if rec == "DECREASE" else "#4ade80"
        st.markdown(
            f'<div style="background:{accent}18;border:1px solid {accent}55;'
            f'border-left:4px solid {accent};border-radius:10px;'
            f'padding:0.9rem 1.2rem;margin-bottom:0.8rem;color:#e8eaf0;">'
            f'<b style="color:{accent}">ADAPTIVE SUGGESTION</b><br>'
            f'{level_change_message(rec, user["current_level"])}</div>',
            unsafe_allow_html=True)
        cy, cn = st.columns(2)
        with cy:
            if st.button("✅ Yes, update level", disabled=st.session_state.generating):
                new = apply_recommendation(user["user_id"], user["current_level"],
                                           result["recommendation"])
                st.session_state.user["current_level"] = new
                st.session_state.assessment_pending    = None
                st.rerun()
        with cn:
            if st.button("❌ Stay at current level", disabled=st.session_state.generating):
                st.session_state.assessment_pending = None
                st.rerun()

    # ── Message Container ───────────────────────────────────────────────
    chat_block = st.container()

    with chat_block:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for msg in st.session_state.messages:
            render_message(msg)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Dynamic Input Box ─────────────────────────────────────────────────
    # If generating is True, it turns gray and says "Generating..." completely blocking input.
    placeholder_text = "⏳ Generating response..." if st.session_state.generating else f"Ask anything about {topic}..."
    user_query = st.chat_input(placeholder_text, disabled=st.session_state.generating)

    # ── Process Execution on Submission ──────────────────────────────────
    if user_query and not st.session_state.generating:
        query = user_query.strip()

        # Turn on the lock immediately and trigger a re-render so the chat input locks up
        st.session_state.generating = True

        # Classify + socratic checks
        classification = classify_query(query)
        is_direct      = classification == "DIRECT_ANSWER_REQUEST"
        st.session_state.direct_history.append(is_direct)
        socratic = check_socratic_mode(st.session_state.direct_history)
        st.session_state.socratic_mode = socratic
        set_socratic(sid, socratic)

        # Log + store user message immediately to database and state
        add_message(user["user_id"], sid, "user", query,
                    user["current_level"], socratic, classification)
        st.session_state.messages.append({"role": "user", "content": query, "socratic": False})

        # Instantly lock the screen layout and draw user message
        st.rerun()

    # ── Generation Pipeline (Runs when lock is active) ────────────────────
    if st.session_state.generating and st.session_state.messages:
        # Check if the last item is from the user (ensures we need an assistant answer)
        if st.session_state.messages[-1]["role"] == "user":
            query = st.session_state.messages[-1]["content"]

            with chat_block:
                sc_cls   = " msg-socratic" if st.session_state.socratic_mode else ""
                label    = "🔄 SOCRATIC MODE" if st.session_state.socratic_mode else "TUTOR"
                history  = get_history(user["user_id"], n=20)

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

                # Final clean render replacing the streaming cursor chunk
                stream_placeholder.markdown(
                    f'<div class="msg-assistant{sc_cls}">'
                    f'<div class="msg-label">{label}</div>'
                    f'{full_response}</div>',
                    unsafe_allow_html=True,
                )

            # Append structured assistant response payload into database and state
            add_message(user["user_id"], sid, "assistant", full_response,
                        user["current_level"], st.session_state.socratic_mode, None)
            st.session_state.messages.append({"role": "assistant", "content": full_response, "socratic": st.session_state.socratic_mode})

            # Metric progression tracking updates
            bump_interactions(user["user_id"])
            st.session_state.interaction_count += 1
            st.session_state.user["total_interactions"] += 1

            # Periodic optimization checking for adaptation thresholds
            if should_assess(st.session_state.interaction_count):
                result = run_assessment(user["user_id"], user["current_level"])
                st.write("DEBUG chat assessment:", result)   # ← temporary
                if result:
                    st.session_state.assessment_pending = result

            # Unlock the input container and refresh layout smoothly
            st.session_state.generating = False
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────
# QUIZ PAGE
# ─────────────────────────────────────────────────────────────────────────

def page_quiz():
    user = st.session_state.user
    st.markdown("## 🧠 Quiz")

    # ── Show result ───────────────────────────────────────────────────────
    if st.session_state.quiz_result:
        r   = st.session_state.quiz_result
        pct = r["score"] * 100
        st.markdown(f"### {'🏆' if pct==100 else '✅' if pct>=70 else '📚'} {r['topic']}")
        st.markdown(
            f"**{r['correct']}/{r['total']} ({pct:.0f}%)** · "
            + level_badge(r["level"]), unsafe_allow_html=True)

        if pct == 100:  st.success("Perfect score!")
        elif pct >= 70: st.success("Great job!")
        elif pct >= 50: st.warning("Review and try again.")
        else:           st.error("More study needed.")

        # ── ZPD recommendation (computed ONCE at submit, read here) ────────
        zpd = r.get("zpd")
        if zpd and zpd["recommendation"] != "MAINTAIN":
            st.markdown("---")
            st.info(level_change_message(zpd["recommendation"], user["current_level"]))
            st.caption(f"Why: {zpd['reasoning']}")
            cy, cn = st.columns(2)
            with cy:
                if st.button("✅ Yes, update level", key="zpd_inline_yes"):
                    new = apply_recommendation(user["user_id"],
                                               user["current_level"],
                                               zpd["recommendation"])
                    st.session_state.user["current_level"] = new
                    st.session_state.quiz_result = None
                    st.session_state.quiz_data   = None
                    st.rerun()
            with cn:
                if st.button("❌ Stay at current level", key="zpd_inline_no"):
                    st.session_state.quiz_result = None
                    st.session_state.quiz_data   = None
                    st.rerun()

        st.markdown("---")
        for i, q in enumerate(r["breakdown"], 1):
            icon = "✅" if q["is_correct"] else "❌"
            with st.expander(f"{icon} Q{i}: {q['question']}"):
                for j, opt in enumerate(q["options"]):
                    tag = ""
                    if j == q["correct_answer"]:              tag = " ✅"
                    if j == q["user_answer"] and not q["is_correct"]: tag = " ❌ your answer"
                    st.markdown(f"{'→ ' if j==q['user_answer'] else '  '}{opt}{tag}")
                st.info(f"💡 {q['explanation']}")

        if st.button("← Take another quiz", type="primary"):
            st.session_state.quiz_result = None
            st.session_state.quiz_data   = None
            st.rerun()
        return

    # ── Active quiz ───────────────────────────────────────────────────────
    if st.session_state.quiz_data:
        qdata     = st.session_state.quiz_data
        questions = qdata["questions"]
        st.markdown(f"### {qdata['topic']} · {level_badge(qdata['level'])}",
                    unsafe_allow_html=True)
        st.markdown(f"{len(questions)} questions · Answer all to submit")
        st.markdown("---")

        answers      = []
        all_answered = True
        for i, q in enumerate(questions):
            st.markdown(f"**Q{i+1}: {q['question']}**")
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
        if st.button("Submit →", type="primary",
                     disabled=not all_answered, use_container_width=True):
            result = submit_quiz(qdata, answers)
            # ZPD: assess once, now that a fresh score exists, and stash it on
            # the result dict so the result screen can show it without recomputing.
            result["zpd"] = run_assessment(user["user_id"], user["current_level"])
            st.session_state.quiz_result = result
            st.session_state.quiz_data   = None
            st.rerun()
        if not all_answered:
            st.caption("Answer all questions to submit.")
        return

    # ── Quiz setup ────────────────────────────────────────────────────────
    st.markdown("Generate an AI quiz on any DSA topic.")
    topic_q = st.text_input("Topic", placeholder="e.g. Merge Sort",
                             label_visibility="collapsed")
    n_q     = st.slider("Number of questions", 3, 10, 5)
    level_q = st.selectbox("Difficulty",
                           options=list(LEVEL_LABELS.keys()),
                           format_func=lambda x: f"{x} — {LEVEL_LABELS[x]}",
                           index=user["current_level"] - 1)

    if st.button("Generate Quiz →", type="primary", use_container_width=True):
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
                st.error("Could not generate quiz. Check Ollama is running.")
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

    st.markdown(f"## 📊 Stats — {user['user_id']}")

    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in [
        (c1, user["total_interactions"],           "Total Chats"),
        (c2, f"{acc:.0f}%",                        "Quiz Accuracy"),
        (c3, len(history),                         "Quizzes Taken"),
        (c4, LEVEL_LABELS[user["current_level"]], "Level"),
    ]:
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value">{val}</div>'
                f'<div class="metric-label">{lbl}</div></div>',
                unsafe_allow_html=True)

    st.markdown("---")

    if history:
        import pandas as pd
        df       = pd.DataFrame(history)
        df["pct"] = df["score"] * 100

        st.markdown('<div class="section-header">Score Trend</div>',
                    unsafe_allow_html=True)
        st.line_chart(df[["pct"]].rename(columns={"pct": "Score (%)"}))

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