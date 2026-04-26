import streamlit as st
import joblib
import numpy as np
import re
import json
import hashlib
import os
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from textblob import TextBlob
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# ── Page Config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="TruthLens — AI Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Professional CSS ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root Theme ── */
:root {
    --bg:        #0a0e1a;
    --surface:   #111827;
    --card:      #1a2236;
    --border:    #1e2d45;
    --accent:    #00d4ff;
    --accent2:   #7c3aed;
    --danger:    #ef4444;
    --success:   #10b981;
    --warning:   #f59e0b;
    --text:      #f1f5f9;
    --muted:     #64748b;
    --font-head: 'Syne', sans-serif;
    --font-body: 'DM Sans', sans-serif;
}

/* ── Global ── */
html, body, [class*="css"] {
    font-family: var(--font-body);
    background-color: var(--bg);
    color: var(--text);
}
.stApp { background: var(--bg); }
.block-container { padding: 1.5rem 2rem !important; max-width: 1200px; }

/* ── Hide Streamlit branding ── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent2), var(--accent)) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.6rem !important;
    font-family: var(--font-head) !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.03em !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(0, 212, 255, 0.2) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(0, 212, 255, 0.35) !important;
}

/* ── Inputs ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: var(--font-body) !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 1rem !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(0,212,255,0.15) !important;
}

/* ── Metric Cards ── */
[data-testid="stMetric"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1rem 1.2rem !important;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 0.8rem !important; }
[data-testid="stMetricValue"] { color: var(--text) !important; font-family: var(--font-head) !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-radius: 12px !important;
    padding: 0.3rem !important;
    gap: 0.3rem !important;
    border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 8px !important;
    color: var(--muted) !important;
    font-family: var(--font-head) !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.2rem !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--accent2), var(--accent)) !important;
    color: white !important;
}

/* ── Alerts ── */
.stAlert { border-radius: 12px !important; border: none !important; }

/* ── Progress bar ── */
.stProgress > div > div { background: linear-gradient(90deg, var(--accent2), var(--accent)) !important; border-radius: 10px !important; }
.stProgress > div { background: var(--card) !important; border-radius: 10px !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* ── Custom Components ── */
.hero-banner {
    background: linear-gradient(135deg, #0d1b3e 0%, #1a0533 50%, #0a1628 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(0,212,255,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: var(--font-head);
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #fff 30%, var(--accent));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.5rem 0;
    line-height: 1.2;
}
.hero-sub {
    color: var(--muted);
    font-size: 1rem;
    margin: 0;
    max-width: 500px;
}
.badge {
    display: inline-block;
    background: rgba(0,212,255,0.1);
    border: 1px solid rgba(0,212,255,0.3);
    color: var(--accent);
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.75rem;
    font-family: var(--font-head);
    font-weight: 600;
    letter-spacing: 0.05em;
    margin-bottom: 1rem;
}
.stat-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    transition: border-color 0.2s;
}
.stat-card:hover { border-color: var(--accent); }
.stat-number {
    font-family: var(--font-head);
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--accent2), var(--accent));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.stat-label { color: var(--muted); font-size: 0.85rem; margin-top: 0.2rem; }

.verdict-fake {
    background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(239,68,68,0.05));
    border: 1px solid rgba(239,68,68,0.4);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
}
.verdict-real {
    background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(16,185,129,0.05));
    border: 1px solid rgba(16,185,129,0.4);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
}
.verdict-uncertain {
    background: linear-gradient(135deg, rgba(245,158,11,0.15), rgba(245,158,11,0.05));
    border: 1px solid rgba(245,158,11,0.4);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
}
.verdict-title {
    font-family: var(--font-head);
    font-size: 1.8rem;
    font-weight: 800;
    margin: 0;
}
.verdict-sub { color: var(--muted); font-size: 0.9rem; margin-top: 0.3rem; }

.section-title {
    font-family: var(--font-head);
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text);
    margin: 1.5rem 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.signal-tag-fake {
    display: inline-block;
    background: rgba(239,68,68,0.15);
    border: 1px solid rgba(239,68,68,0.3);
    color: #f87171;
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.8rem;
    margin: 0.2rem;
}
.signal-tag-real {
    display: inline-block;
    background: rgba(16,185,129,0.15);
    border: 1px solid rgba(16,185,129,0.3);
    color: #34d399;
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.8rem;
    margin: 0.2rem;
}
.feature-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem;
}
.login-container {
    max-width: 420px;
    margin: 4rem auto;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2.5rem;
}
.login-title {
    font-family: var(--font-head);
    font-size: 1.8rem;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(135deg, #fff, var(--accent));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.login-sub {
    text-align: center;
    color: var(--muted);
    font-size: 0.9rem;
    margin-bottom: 2rem;
}
.sidebar-user {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1rem;
    text-align: center;
}
.sidebar-username {
    font-family: var(--font-head);
    font-weight: 700;
    font-size: 1rem;
    color: var(--text);
}
.sidebar-role {
    color: var(--accent);
    font-size: 0.75rem;
    margin-top: 0.2rem;
}
.nav-item {
    padding: 0.6rem 1rem;
    border-radius: 8px;
    cursor: pointer;
    font-family: var(--font-head);
    font-weight: 600;
    font-size: 0.9rem;
    color: var(--muted);
    transition: all 0.2s;
}
.info-box {
    background: rgba(0,212,255,0.07);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin: 1rem 0;
    font-size: 0.85rem;
    color: #94a3b8;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# USER AUTH (JSON file-based — simple & works locally)
# ══════════════════════════════════════════════════════════════════
USERS_FILE = "users.json"

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password, email):
    users = load_users()
    if username in users:
        return False, "Username already exists."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    users[username] = {
        "password": hash_password(password),
        "email": email,
        "analyses": 0
    }
    save_users(users)
    return True, "Account created successfully!"

def login_user(username, password):
    users = load_users()
    if username not in users:
        return False, "Username not found."
    if users[username]["password"] != hash_password(password):
        return False, "Incorrect password."
    return True, users[username]

def increment_analyses(username):
    users = load_users()
    if username in users:
        users[username]["analyses"] = users[username].get("analyses", 0) + 1
        save_users(users)

# ══════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "user_data" not in st.session_state:
    st.session_state.user_data = {}
if "history" not in st.session_state:
    st.session_state.history = []
if "page" not in st.session_state:
    st.session_state.page = "login"

# ══════════════════════════════════════════════════════════════════
# LOAD ML MODELS
# ══════════════════════════════════════════════════════════════════
@st.cache_resource
def load_models():
    try:
        model  = joblib.load('ensemble_model.pkl')
        tfidf  = joblib.load('tfidf_vectorizer.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, tfidf, scaler, True
    except Exception as e:
        return None, None, None, False

model, tfidf, scaler, model_loaded = load_models()

# ══════════════════════════════════════════════════════════════════
# ML HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════
CLICKBAIT_WORDS = [
    "breaking", "shocking", "secret", "exposed", "alert",
    "unbelievable", "bombshell", "urgent", "miracle", "conspiracy",
    "hidden", "banned", "leaked", "viral", "hoax",
    "must see", "you wont believe", "they dont want", "wake up", "truth revealed"
]

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_features(raw_text):
    text       = str(raw_text)
    text_lower = text.lower()
    words      = text_lower.split()
    total_chars = max(len(text), 1)
    total_words = max(len(words), 1)

    word_count            = len(words)
    char_count            = len(text)
    avg_word_length       = char_count / (total_words + 1)
    exclamation_count     = text.count("!")
    question_marks        = text.count("?")
    uppercase_ratio       = sum(1 for c in text if c.isupper()) / total_chars
    clickbait_score       = sum(p in text_lower for p in CLICKBAIT_WORDS)
    blob                  = TextBlob(text_lower)
    sentiment_polarity    = blob.sentiment.polarity
    sentiment_subjectivity= blob.sentiment.subjectivity
    unique_word_ratio     = len(set(words)) / (total_words + 1)
    digit_ratio           = sum(1 for c in text if c.isdigit()) / total_chars

    return [word_count, char_count, avg_word_length, exclamation_count,
            question_marks, uppercase_ratio, clickbait_score,
            sentiment_polarity, sentiment_subjectivity,
            unique_word_ratio, digit_ratio]

def run_prediction(text):
    cleaned      = clean_text(text)
    text_vec     = tfidf.transform([cleaned])
    raw_feats    = extract_features(text)
    scaled_feats = scaler.transform([raw_feats])
    final_input  = hstack([text_vec, csr_matrix(scaled_feats)])
    pred         = model.predict(final_input)[0]
    prob         = model.predict_proba(final_input)[0]
    return pred, prob, raw_feats

def get_top_words():
    lr_model = model.named_estimators_['lr']
    vocab    = tfidf.get_feature_names_out()
    coefs    = lr_model.coef_[0][:len(vocab)]
    top_fake = [vocab[i] for i in coefs.argsort()[-20:]]
    top_real = [vocab[i] for i in coefs.argsort()[:20]]
    return top_fake, top_real

# ══════════════════════════════════════════════════════════════════
# LOGIN / SIGNUP PAGE
# ══════════════════════════════════════════════════════════════════
def show_auth_page():
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("""
        <div style='text-align:center; margin-top: 2rem; margin-bottom: 2rem;'>
            <div style='font-size: 2.5rem; margin-bottom: 0.5rem;'>🔍</div>
            <div style='font-family: Syne, sans-serif; font-size: 1.8rem; font-weight: 800;
                        background: linear-gradient(135deg, #fff 30%, #00d4ff);
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                TruthLens
            </div>
            <div style='color: #64748b; font-size: 0.9rem; margin-top: 0.3rem;'>
                AI-Powered Fake News Detector
            </div>
        </div>
        """, unsafe_allow_html=True)

        tab_login, tab_signup = st.tabs(["🔐 Login", "✨ Create Account"])

        with tab_login:
            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
            username = st.text_input("Username", key="login_user", placeholder="Enter your username")
            password = st.text_input("Password", type="password", key="login_pass", placeholder="Enter your password")

            if st.button("Login →", key="btn_login", use_container_width=True):
                if not username or not password:
                    st.error("Please fill in all fields.")
                else:
                    success, result = login_user(username, password)
                    if success:
                        st.session_state.logged_in = True
                        st.session_state.username  = username
                        st.session_state.user_data = result
                        st.success("Welcome back! Redirecting...")
                        st.rerun()
                    else:
                        st.error(result)

            st.markdown("""
            <div class='info-box'>
                💡 <b>Demo account:</b> username <code>demo</code> / password <code>demo123</code>
            </div>
            """, unsafe_allow_html=True)

            # Auto-create demo account if not exists
            users = load_users()
            if "demo" not in users:
                register_user("demo", "demo123", "demo@truthlens.ai")

        with tab_signup:
            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
            new_user  = st.text_input("Choose a Username", key="reg_user", placeholder="e.g. jaspreet123")
            new_email = st.text_input("Email Address", key="reg_email", placeholder="you@email.com")
            new_pass  = st.text_input("Password", type="password", key="reg_pass", placeholder="Min 6 characters")
            new_pass2 = st.text_input("Confirm Password", type="password", key="reg_pass2", placeholder="Repeat password")

            if st.button("Create Account →", key="btn_signup", use_container_width=True):
                if not all([new_user, new_email, new_pass, new_pass2]):
                    st.error("Please fill in all fields.")
                elif new_pass != new_pass2:
                    st.error("Passwords do not match.")
                else:
                    success, msg = register_user(new_user, new_pass, new_email)
                    if success:
                        st.success(f"✅ {msg} Please login.")
                    else:
                        st.error(msg)

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
def show_sidebar():
    with st.sidebar:
        # Logo
        st.markdown("""
        <div style='text-align:center; padding: 1rem 0 1.5rem 0;'>
            <div style='font-size: 2rem;'>🔍</div>
            <div style='font-family: Syne, sans-serif; font-size: 1.3rem; font-weight: 800;
                        background: linear-gradient(135deg, #fff, #00d4ff);
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                TruthLens
            </div>
        </div>
        """, unsafe_allow_html=True)

        # User card
        analyses = load_users().get(st.session_state.username, {}).get("analyses", 0)
        st.markdown(f"""
        <div class='sidebar-user'>
            <div style='font-size:2rem;'>👤</div>
            <div class='sidebar-username'>{st.session_state.username}</div>
            <div class='sidebar-role'>✨ Verified Analyst</div>
            <div style='margin-top:0.6rem; font-size:0.8rem; color:#64748b;'>
                {analyses} articles analyzed
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("<div style='font-family:Syne,sans-serif; font-size:0.75rem; font-weight:700; color:#64748b; letter-spacing:0.08em; padding: 0 0.5rem 0.5rem 0.5rem;'>NAVIGATION</div>", unsafe_allow_html=True)

        pages = {
            "🏠  Dashboard":       "dashboard",
            "📝  Analyze Text":    "analyze",
            "🌐  URL Checker":     "url",
            "📰  Batch Analyze":   "batch",
            "📊  History":         "history",
            "☁️  Word Cloud":      "wordcloud",
        }

        for label, page_key in pages.items():
            is_active = st.session_state.page == page_key
            btn_style = "background: linear-gradient(135deg,#7c3aed,#00d4ff); color:white;" if is_active else ""
            if st.button(label, key=f"nav_{page_key}", use_container_width=True):
                st.session_state.page = page_key
                st.rerun()

        st.markdown("---")
        if st.button("🚪  Logout", use_container_width=True, key="logout_btn"):
            st.session_state.logged_in = False
            st.session_state.username  = ""
            st.session_state.history   = []
            st.session_state.page      = "login"
            st.rerun()

        st.markdown("""
        <div style='text-align:center; color:#374151; font-size:0.72rem; margin-top:2rem;'>
            TruthLens v2.0<br>Built with ML + Streamlit
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# DASHBOARD PAGE
# ══════════════════════════════════════════════════════════════════
def show_dashboard():
    st.markdown("""
    <div class='hero-banner'>
        <div class='badge'>🤖 AI POWERED</div>
        <div class='hero-title'>Welcome to TruthLens</div>
        <p class='hero-sub'>Detect fake news instantly using our Ensemble ML model — combining Logistic Regression + XGBoost with 11 behavioral signals.</p>
    </div>
    """, unsafe_allow_html=True)

    # Stats row
    c1, c2, c3, c4 = st.columns(4)
    analyses = load_users().get(st.session_state.username, {}).get("analyses", 0)
    history_fake = sum(1 for h in st.session_state.history if h["verdict"] == "Fake")
    history_real = sum(1 for h in st.session_state.history if h["verdict"] == "Real")

    with c1:
        st.markdown(f"<div class='stat-card'><div class='stat-number'>{analyses}</div><div class='stat-label'>Total Analyses</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='stat-card'><div class='stat-number' style='background:linear-gradient(135deg,#ef4444,#f87171);-webkit-background-clip:text;-webkit-text-fill-color:transparent;'>{history_fake}</div><div class='stat-label'>Fake Detected</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='stat-card'><div class='stat-number' style='background:linear-gradient(135deg,#10b981,#34d399);-webkit-background-clip:text;-webkit-text-fill-color:transparent;'>{history_real}</div><div class='stat-label'>Real Verified</div></div>", unsafe_allow_html=True)
    with c4:
        accuracy = "~98%"
        st.markdown(f"<div class='stat-card'><div class='stat-number'>{accuracy}</div><div class='stat-label'>Model Accuracy</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Features grid
    st.markdown("<div class='section-title'>⚡ What You Can Do</div>", unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)

    with f1:
        st.markdown("""
        <div class='feature-card'>
            <div style='font-size:1.5rem;'>📝</div>
            <div style='font-family:Syne,sans-serif; font-weight:700; margin: 0.4rem 0 0.3rem 0;'>Analyze Text</div>
            <div style='color:#64748b; font-size:0.85rem;'>Paste any headline or article and get instant verdict with confidence score.</div>
        </div>
        """, unsafe_allow_html=True)
    with f2:
        st.markdown("""
        <div class='feature-card'>
            <div style='font-size:1.5rem;'>🌐</div>
            <div style='font-family:Syne,sans-serif; font-weight:700; margin: 0.4rem 0 0.3rem 0;'>URL Checker</div>
            <div style='color:#64748b; font-size:0.85rem;'>Paste a news link — we auto-scrape and analyze the full article.</div>
        </div>
        """, unsafe_allow_html=True)
    with f3:
        st.markdown("""
        <div class='feature-card'>
            <div style='font-size:1.5rem;'>📰</div>
            <div style='font-family:Syne,sans-serif; font-weight:700; margin: 0.4rem 0 0.3rem 0;'>Batch Analyze</div>
            <div style='color:#64748b; font-size:0.85rem;'>Upload a CSV with multiple headlines and analyze them all at once.</div>
        </div>
        """, unsafe_allow_html=True)

    f4, f5, f6 = st.columns(3)
    with f4:
        st.markdown("""
        <div class='feature-card'>
            <div style='font-size:1.5rem;'>📊</div>
            <div style='font-family:Syne,sans-serif; font-weight:700; margin: 0.4rem 0 0.3rem 0;'>History Chart</div>
            <div style='color:#64748b; font-size:0.85rem;'>Track and compare all your predictions in a visual dashboard.</div>
        </div>
        """, unsafe_allow_html=True)
    with f5:
        st.markdown("""
        <div class='feature-card'>
            <div style='font-size:1.5rem;'>☁️</div>
            <div style='font-family:Syne,sans-serif; font-weight:700; margin: 0.4rem 0 0.3rem 0;'>Word Cloud</div>
            <div style='color:#64748b; font-size:0.85rem;'>Visual map of words the model learned to associate with fake vs real news.</div>
        </div>
        """, unsafe_allow_html=True)
    with f6:
        st.markdown("""
        <div class='feature-card'>
            <div style='font-size:1.5rem;'>🎯</div>
            <div style='font-family:Syne,sans-serif; font-weight:700; margin: 0.4rem 0 0.3rem 0;'>Score Meter</div>
            <div style='color:#64748b; font-size:0.85rem;'>Get a 0–100% fake score — not just a binary Real/Fake label.</div>
        </div>
        """, unsafe_allow_html=True)

    # Recent history
    if st.session_state.history:
        st.markdown("<div class='section-title'>🕒 Recent Analyses</div>", unsafe_allow_html=True)
        for item in reversed(st.session_state.history[-5:]):
            icon  = "🔴" if item["verdict"] == "Fake" else "🟢"
            color = "#ef4444" if item["verdict"] == "Fake" else "#10b981"
            st.markdown(f"""
            <div class='feature-card' style='display:flex; justify-content:space-between; align-items:center;'>
                <div style='flex:1; font-size:0.88rem; color:#94a3b8;'>{item['text'][:80]}...</div>
                <div style='font-family:Syne,sans-serif; font-weight:700; color:{color}; margin-left:1rem; white-space:nowrap;'>
                    {icon} {item['verdict']} &nbsp; {item['fake_score']:.0f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# ANALYZE TEXT PAGE
# ══════════════════════════════════════════════════════════════════
def show_analyze_page():
    st.markdown("<div class='hero-title' style='font-size:1.8rem; margin-bottom:0.3rem;'>📝 Analyze Text</div>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b; margin-bottom:1.5rem;'>Paste a news headline or full article below and get an instant AI verdict.</p>", unsafe_allow_html=True)

    user_input = st.text_area("News Article / Headline", height=180,
                               placeholder="e.g. BREAKING: Scientists discover shocking miracle cure hidden by government...")

    col_btn, col_clear = st.columns([1, 5])
    with col_btn:
        analyze_clicked = st.button("🔍 Analyze Now", key="btn_analyze")

    if analyze_clicked:
        if not user_input.strip():
            st.warning("Please enter some text to analyze.")
            return
        if not model_loaded:
            st.error("Model files not found. Make sure .pkl files are in the same folder as app.py")
            return

        with st.spinner("Analyzing article..."):
            pred, prob, raw_feats = run_prediction(user_input)

        fake_score = prob[1] * 100
        real_score = prob[0] * 100

        # Log to history
        increment_analyses(st.session_state.username)
        st.session_state.history.append({
            "text": user_input,
            "verdict": "Fake" if pred == 1 else "Real",
            "fake_score": fake_score,
            "real_score": real_score
        })

        st.markdown("<br>", unsafe_allow_html=True)

        # Verdict
        if fake_score >= 70:
            verdict_class = "verdict-fake"
            verdict_text  = "🔴 FAKE NEWS DETECTED"
            verdict_sub   = f"Our model is {fake_score:.1f}% confident this is fake or misleading."
        elif fake_score >= 40:
            verdict_class = "verdict-uncertain"
            verdict_text  = "🟡 UNCERTAIN — VERIFY MANUALLY"
            verdict_sub   = f"Model confidence is low. Fake score: {fake_score:.1f}%. Please cross-check."
        else:
            verdict_class = "verdict-real"
            verdict_text  = "🟢 LIKELY REAL NEWS"
            verdict_sub   = f"Our model is {real_score:.1f}% confident this is real news."

        st.markdown(f"""
        <div class='{verdict_class}'>
            <div class='verdict-title'>{verdict_text}</div>
            <div class='verdict-sub'>{verdict_sub}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Confidence scores
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div style='font-family:Syne,sans-serif; font-size:0.85rem; color:#64748b; margin-bottom:0.3rem;'>FAKE CONFIDENCE</div>", unsafe_allow_html=True)
            st.progress(int(fake_score) / 100)
            st.markdown(f"<div style='font-family:Syne,sans-serif; font-size:1.4rem; font-weight:700; color:#ef4444;'>{fake_score:.1f}%</div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<div style='font-family:Syne,sans-serif; font-size:0.85rem; color:#64748b; margin-bottom:0.3rem;'>REAL CONFIDENCE</div>", unsafe_allow_html=True)
            st.progress(int(real_score) / 100)
            st.markdown(f"<div style='font-family:Syne,sans-serif; font-size:1.4rem; font-weight:700; color:#10b981;'>{real_score:.1f}%</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Metrics
        st.markdown("<div class='section-title'>📐 Article Behavioral Signals</div>", unsafe_allow_html=True)
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Word Count",     int(raw_feats[0]))
        m2.metric("Exclamations",   int(raw_feats[3]))
        m3.metric("Question Marks", int(raw_feats[4]))
        m4.metric("Uppercase %",    f"{raw_feats[5]*100:.1f}%")
        m5.metric("Clickbait Score",int(raw_feats[6]))
        m6.metric("Subjectivity",   f"{raw_feats[8]:.2f}")

        # Signal words
        top_fake_words, top_real_words = get_top_words()
        words = user_input.lower().split()
        fake_hits = list(dict.fromkeys(w for w in words if w in top_fake_words))[:8]
        real_hits = list(dict.fromkeys(w for w in words if w in top_real_words))[:8]

        col_f, col_r = st.columns(2)
        with col_f:
            st.markdown("<div class='section-title'>🔴 Fake Signal Words Found</div>", unsafe_allow_html=True)
            if fake_hits:
                tags = "".join([f"<span class='signal-tag-fake'>{w}</span>" for w in fake_hits])
                st.markdown(tags, unsafe_allow_html=True)
            else:
                st.markdown("<span style='color:#64748b; font-size:0.85rem;'>None detected</span>", unsafe_allow_html=True)
        with col_r:
            st.markdown("<div class='section-title'>🟢 Real Signal Words Found</div>", unsafe_allow_html=True)
            if real_hits:
                tags = "".join([f"<span class='signal-tag-real'>{w}</span>" for w in real_hits])
                st.markdown(tags, unsafe_allow_html=True)
            else:
                st.markdown("<span style='color:#64748b; font-size:0.85rem;'>None detected</span>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# URL CHECKER PAGE
# ══════════════════════════════════════════════════════════════════
def show_url_page():
    st.markdown("<div class='hero-title' style='font-size:1.8rem; margin-bottom:0.3rem;'>🌐 URL Checker</div>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b; margin-bottom:1.5rem;'>Paste any news article URL — we'll scrape and analyze it automatically.</p>", unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
        💡 Works best with: BBC, Reuters, CNN, The Guardian, Times of India, NDTV, and most major news sites.
    </div>
    """, unsafe_allow_html=True)

    url_input = st.text_input("News Article URL", placeholder="https://www.bbc.com/news/article-link-here")

    if st.button("🌐 Fetch & Analyze", key="btn_url"):
        if not url_input.strip():
            st.warning("Please enter a URL.")
            return
        try:
            from newspaper import Article
            with st.spinner("Fetching article from URL..."):
                article = Article(url_input)
                article.download()
                article.parse()
                full_text = (article.title or "") + " " + (article.text or "")

            if len(full_text.split()) < 30:
                st.error("Could not extract enough text. Try a different news link.")
                return

            st.markdown(f"""
            <div class='feature-card'>
                <div style='font-size:0.75rem; color:#64748b; font-family:Syne,sans-serif; font-weight:600; letter-spacing:0.05em;'>ARTICLE FETCHED</div>
                <div style='font-size:1rem; font-weight:600; margin-top:0.4rem;'>{article.title}</div>
                <div style='color:#64748b; font-size:0.82rem; margin-top:0.3rem;'>{len(article.text.split())} words extracted from {url_input[:50]}...</div>
            </div>
            """, unsafe_allow_html=True)

            with st.spinner("Running AI analysis..."):
                pred, prob, raw_feats = run_prediction(full_text)

            fake_score = prob[1] * 100
            real_score = prob[0] * 100
            increment_analyses(st.session_state.username)
            st.session_state.history.append({
                "text": article.title or url_input,
                "verdict": "Fake" if pred == 1 else "Real",
                "fake_score": fake_score,
                "real_score": real_score
            })

            st.markdown("<br>", unsafe_allow_html=True)
            if fake_score >= 70:
                st.markdown(f"<div class='verdict-fake'><div class='verdict-title'>🔴 FAKE NEWS DETECTED</div><div class='verdict-sub'>Fake Score: {fake_score:.1f}%</div></div>", unsafe_allow_html=True)
            elif fake_score >= 40:
                st.markdown(f"<div class='verdict-uncertain'><div class='verdict-title'>🟡 UNCERTAIN</div><div class='verdict-sub'>Fake Score: {fake_score:.1f}% — Verify manually</div></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='verdict-real'><div class='verdict-title'>🟢 LIKELY REAL NEWS</div><div class='verdict-sub'>Real Score: {real_score:.1f}%</div></div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Fake Score",    f"{fake_score:.1f}%")
            c2.metric("Real Score",    f"{real_score:.1f}%")
            c3.metric("Clickbait Words", int(raw_feats[6]))

        except ImportError:
            st.error("Please install newspaper3k: pip install newspaper3k lxml_html_clean")
        except Exception as e:
            st.error(f"Could not fetch article: {e}")

# ══════════════════════════════════════════════════════════════════
# BATCH ANALYZER PAGE
# ══════════════════════════════════════════════════════════════════
def show_batch_page():
    st.markdown("<div class='hero-title' style='font-size:1.8rem; margin-bottom:0.3rem;'>📰 Batch Analyzer</div>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b; margin-bottom:1.5rem;'>Upload a CSV file with multiple headlines and analyze them all at once.</p>", unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
        📋 <b>CSV Format:</b> Your file must have a column with news text. Default column name: <code>headline</code>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV File", type=["csv"])
    col_name = st.text_input("Column name with news text", value="headline")

    if uploaded and st.button("📰 Run Batch Analysis", key="btn_batch"):
        df = pd.read_csv(uploaded)
        if col_name not in df.columns:
            st.error(f"Column '{col_name}' not found. Available: {list(df.columns)}")
            return

        results = []
        bar = st.progress(0)
        status = st.empty()
        total = len(df)

        for i, text in enumerate(df[col_name]):
            try:
                pred, prob, _ = run_prediction(str(text))
                results.append({
                    "Headline"   : str(text)[:80],
                    "Verdict"    : "🔴 Fake" if pred == 1 else "🟢 Real",
                    "Fake Score" : f"{prob[1]*100:.1f}%",
                    "Real Score" : f"{prob[0]*100:.1f}%"
                })
            except:
                results.append({"Headline": str(text)[:80], "Verdict": "⚠️ Error",
                                 "Fake Score": "—", "Real Score": "—"})
            bar.progress((i+1)/total)
            status.markdown(f"<span style='color:#64748b;'>Analyzed {i+1} of {total}...</span>", unsafe_allow_html=True)

        status.empty()
        results_df = pd.DataFrame(results)

        n_fake = sum(1 for r in results if "Fake" in r["Verdict"])
        n_real = sum(1 for r in results if "Real" in r["Verdict"])

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Articles",  total)
        c2.metric("🔴 Fake Detected", n_fake)
        c3.metric("🟢 Real Verified", n_real)

        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(results_df, use_container_width=True, height=400)

        csv_out = results_df.to_csv(index=False)
        st.download_button("⬇️ Download Results as CSV", csv_out,
                           "batch_results.csv", "text/csv", use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# HISTORY PAGE
# ══════════════════════════════════════════════════════════════════
def show_history_page():
    st.markdown("<div class='hero-title' style='font-size:1.8rem; margin-bottom:0.3rem;'>📊 Analysis History</div>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b; margin-bottom:1.5rem;'>Visual breakdown of all predictions made in this session.</p>", unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown("""
        <div class='feature-card' style='text-align:center; padding:3rem;'>
            <div style='font-size:2rem; margin-bottom:0.5rem;'>📭</div>
            <div style='font-family:Syne,sans-serif; font-weight:700;'>No analyses yet</div>
            <div style='color:#64748b; margin-top:0.3rem;'>Go to Analyze Text or URL Checker to get started.</div>
        </div>
        """, unsafe_allow_html=True)
        return

    history = st.session_state.history
    labels      = [h["text"][:35] + "..." for h in history]
    fake_scores = [h["fake_score"] for h in history]
    verdicts    = [h["verdict"] for h in history]
    colors      = ["#ef4444" if v == "Fake" else "#10b981" for v in verdicts]

    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='#111827')

    # Chart 1: Fake score per article
    bars = axes[0].barh(range(len(labels)), fake_scores, color=colors, edgecolor='none', height=0.6)
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_yticklabels(labels, fontsize=8, color='#94a3b8')
    axes[0].set_xlabel("Fake Score (%)", color='#64748b', fontsize=9)
    axes[0].set_title("Fake Score Per Article", color='white', fontsize=11, fontweight='bold', pad=12)
    axes[0].axvline(50, color='#374151', linestyle='--', linewidth=1)
    axes[0].set_facecolor('#111827')
    axes[0].tick_params(colors='#64748b')
    axes[0].spines[:].set_color('#1e2d45')
    axes[0].set_xlim(0, 100)
    for bar, score in zip(bars, fake_scores):
        axes[0].text(min(score + 1, 95), bar.get_y() + bar.get_height()/2,
                     f'{score:.0f}%', va='center', fontsize=7, color='white')

    # Chart 2: Timeline
    x = range(len(labels))
    axes[1].plot(x, fake_scores, color='#7c3aed', linewidth=2, zorder=2)
    axes[1].scatter(x, fake_scores, color=colors, s=80, zorder=3, edgecolors='white', linewidths=0.5)
    axes[1].fill_between(x, fake_scores, 50,
                         where=[s >= 50 for s in fake_scores],
                         alpha=0.15, color='#ef4444')
    axes[1].fill_between(x, fake_scores, 50,
                         where=[s < 50 for s in fake_scores],
                         alpha=0.15, color='#10b981')
    axes[1].axhline(50, color='#374151', linestyle='--', linewidth=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(i+1) for i in x], color='#64748b', fontsize=8)
    axes[1].set_ylabel("Fake Score (%)", color='#64748b', fontsize=9)
    axes[1].set_title("Score Timeline", color='white', fontsize=11, fontweight='bold', pad=12)
    axes[1].set_facecolor('#111827')
    axes[1].tick_params(colors='#64748b')
    axes[1].spines[:].set_color('#1e2d45')
    axes[1].set_ylim(0, 105)

    fig.patch.set_facecolor('#111827')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Table
    st.markdown("<div class='section-title'>📋 Full History</div>", unsafe_allow_html=True)
    hist_df = pd.DataFrame([{
        "#": i+1,
        "Article": h["text"][:70] + "...",
        "Verdict": ("🔴 Fake" if h["verdict"]=="Fake" else "🟢 Real"),
        "Fake %": f"{h['fake_score']:.1f}%",
        "Real %": f"{h['real_score']:.1f}%"
    } for i, h in enumerate(history)])

    st.dataframe(hist_df, use_container_width=True, hide_index=True)

    if st.button("🗑️ Clear History", key="btn_clear_hist"):
        st.session_state.history = []
        st.rerun()

# ══════════════════════════════════════════════════════════════════
# WORD CLOUD PAGE
# ══════════════════════════════════════════════════════════════════
def show_wordcloud_page():
    st.markdown("<div class='hero-title' style='font-size:1.8rem; margin-bottom:0.3rem;'>☁️ Word Cloud</div>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b; margin-bottom:1.5rem;'>Visual map of what the model learned — which words signal Fake vs Real news.</p>", unsafe_allow_html=True)

    if not model_loaded:
        st.error("Model not loaded.")
        return

    try:
        from wordcloud import WordCloud
    except ImportError:
        st.error("Run: pip install wordcloud")
        return

    with st.spinner("Generating word clouds..."):
        lr_model = model.named_estimators_['lr']
        vocab    = tfidf.get_feature_names_out()
        coefs    = lr_model.coef_[0][:len(vocab)]

        fake_freq = {vocab[i]: float(coefs[i])  for i in coefs.argsort()[-150:]}
        real_freq = {vocab[i]: float(-coefs[i]) for i in coefs.argsort()[:150]}

        plt.style.use('dark_background')
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor='#0a0e1a')

        wc_fake = WordCloud(width=700, height=380, background_color='#111827',
                            colormap='RdYlOr', max_words=80).generate_from_frequencies(fake_freq)
        axes[0].imshow(wc_fake, interpolation='bilinear')
        axes[0].axis('off')
        axes[0].set_title('🔴 FAKE NEWS SIGNALS', color='#f87171', fontsize=13,
                           fontweight='bold', pad=15, fontfamily='monospace')

        wc_real = WordCloud(width=700, height=380, background_color='#111827',
                            colormap='YlGn', max_words=80).generate_from_frequencies(real_freq)
        axes[1].imshow(wc_real, interpolation='bilinear')
        axes[1].axis('off')
        axes[1].set_title('🟢 REAL NEWS SIGNALS', color='#34d399', fontsize=13,
                           fontweight='bold', pad=15, fontfamily='monospace')

        fig.patch.set_facecolor('#0a0e1a')
        plt.tight_layout(pad=2)
        st.pyplot(fig)
        plt.close()

    st.markdown("""
    <div class='info-box'>
        💡 <b>How to read this:</b> Bigger word = stronger signal for that category.
        Red/orange words are commonly found in fake news. Green words appear more in real, credible journalism.
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# MAIN ROUTER
# ══════════════════════════════════════════════════════════════════
if not st.session_state.logged_in:
    show_auth_page()
else:
    show_sidebar()
    page = st.session_state.page

    if page == "dashboard":
        show_dashboard()
    elif page == "analyze":
        show_analyze_page()
    elif page == "url":
        show_url_page()
    elif page == "batch":
        show_batch_page()
    elif page == "history":
        show_history_page()
    elif page == "wordcloud":
        show_wordcloud_page()
    else:
        show_dashboard()
