import streamlit as st
import joblib
import numpy as np
import re
from scipy.sparse import hstack, csr_matrix
from textblob import TextBlob

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Fake News Detector",
    page_icon="🕵️",
    layout="wide"
)

# ── Load model files ──────────────────────────────────────────────
@st.cache_resource   # Cache so it loads only once, not on every click
def load_models():
    model  = joblib.load('ensemble_model.pkl')
    tfidf  = joblib.load('tfidf_vectorizer.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, tfidf, scaler

model, tfidf, scaler = load_models()

# ── Helper functions (same as your notebook) ──────────────────────
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
    total_chars = len(text) if len(text) > 0 else 1
    total_words = len(words) if len(words) > 0 else 1

    word_count         = len(words)
    char_count         = len(text)
    avg_word_length    = char_count / (total_words + 1)
    exclamation_count  = text.count("!")
    question_marks     = text.count("?")
    uppercase_ratio    = sum(1 for c in text if c.isupper()) / total_chars
    clickbait_score    = sum(p in text_lower for p in CLICKBAIT_WORDS)
    blob               = TextBlob(text_lower)
    sentiment_polarity    = blob.sentiment.polarity
    sentiment_subjectivity = blob.sentiment.subjectivity
    unique_word_ratio  = len(set(words)) / (total_words + 1)
    digit_ratio        = sum(1 for c in text if c.isdigit()) / total_chars

    return [word_count, char_count, avg_word_length, exclamation_count,
            question_marks, uppercase_ratio, clickbait_score,
            sentiment_polarity, sentiment_subjectivity,
            unique_word_ratio, digit_ratio]

def predict(text):
    cleaned      = clean_text(text)
    text_vec     = tfidf.transform([cleaned])
    raw_feats    = extract_features(text)
    scaled_feats = scaler.transform([raw_feats])
    final_input  = hstack([text_vec, csr_matrix(scaled_feats)])
    pred         = model.predict(final_input)[0]
    prob         = model.predict_proba(final_input)[0]
    return pred, prob, raw_feats

# ── UI ────────────────────────────────────────────────────────────
st.title("🕵️ AI-Powered Fake News Detector")
st.markdown("Built with **Machine Learning** — Logistic Regression + XGBoost Ensemble")
st.markdown("---")

# Tabs for different features
tab1, tab2, tab3 = st.tabs(["📝 Analyze Text", "🌐 Analyze URL", "📰 Batch Analyze CSV"])

# ─────────────────────────────────────────────────────────────────
# TAB 1: Text Input
# ─────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Paste a news article or headline below")
    user_input = st.text_area("News Text", height=200,
                               placeholder="Paste your news article or headline here...")

    if st.button("🔍 Analyze", key="btn_text"):
        if user_input.strip() == "":
            st.warning("Please enter some text first.")
        else:
            pred, prob, raw_feats = predict(user_input)
            fake_score = prob[1] * 100
            real_score = prob[0] * 100

            st.markdown("---")
            col1, col2 = st.columns(2)

            with col1:
                if pred == 1:
                    st.error(f"### 🔴 FAKE NEWS DETECTED")
                else:
                    st.success(f"### 🟢 REAL NEWS")

            with col2:
                st.metric("Fake Score",  f"{fake_score:.1f}%")
                st.metric("Real Score",  f"{real_score:.1f}%")

            # Score progress bar
            st.markdown("**Fake News Confidence Meter:**")
            st.progress(int(fake_score))

            # Feature breakdown
            st.markdown("---")
            st.subheader("📐 Article Analysis")
            col3, col4, col5 = st.columns(3)
            col3.metric("Clickbait Words",    int(raw_feats[6]))
            col4.metric("Exclamation Marks",  int(raw_feats[3]))
            col5.metric("Uppercase Ratio",    f"{raw_feats[5]*100:.1f}%")

            col6, col7, col8 = st.columns(3)
            col6.metric("Sentiment Polarity",     round(raw_feats[7], 3))
            col7.metric("Sentiment Subjectivity", round(raw_feats[8], 3))
            col8.metric("Word Count",             int(raw_feats[0]))

# ─────────────────────────────────────────────────────────────────
# TAB 2: URL Input
# ─────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Paste a news article URL")
    url_input = st.text_input("News URL", placeholder="https://www.bbc.com/news/...")

    if st.button("🌐 Fetch & Analyze", key="btn_url"):
        if url_input.strip() == "":
            st.warning("Please enter a URL first.")
        else:
            try:
                from newspaper import Article
                with st.spinner("Fetching article..."):
                    article = Article(url_input)
                    article.download()
                    article.parse()
                    full_text = article.title + " " + article.text

                if len(full_text.split()) < 30:
                    st.error("Could not extract enough text from this URL. Try another link.")
                else:
                    st.info(f"**Title:** {article.title}")
                    st.info(f"**Words extracted:** {len(article.text.split())}")

                    pred, prob, raw_feats = predict(full_text)
                    fake_score = prob[1] * 100

                    if pred == 1:
                        st.error(f"### 🔴 FAKE NEWS — {fake_score:.1f}% confidence")
                    else:
                        st.success(f"### 🟢 REAL NEWS — {prob[0]*100:.1f}% confidence")

                    st.progress(int(fake_score))

            except Exception as e:
                st.error(f"Error fetching URL: {e}")

# ─────────────────────────────────────────────────────────────────
# TAB 3: Batch CSV
# ─────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Upload a CSV file with headlines")
    st.markdown("Your CSV must have a column with news text (e.g. `headline`)")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    col_name = st.text_input("Column name containing news text", value="headline")

    if uploaded_file and st.button("📰 Run Batch Analysis", key="btn_batch"):
        import pandas as pd

        df = pd.read_csv(uploaded_file)

        if col_name not in df.columns:
            st.error(f"Column '{col_name}' not found. Available: {list(df.columns)}")
        else:
            results = []
            progress = st.progress(0)
            total = len(df)

            for i, text in enumerate(df[col_name]):
                try:
                    pred, prob, _ = predict(str(text))
                    results.append({
                        "headline"   : str(text)[:80],
                        "prediction" : "Fake" if pred == 1 else "Real",
                        "fake_%"     : round(prob[1]*100, 1),
                        "real_%"     : round(prob[0]*100, 1)
                    })
                except:
                    results.append({"headline": str(text)[:80],
                                    "prediction": "Error", "fake_%": 0, "real_%": 0})
                progress.progress((i+1) / total)

            results_df = pd.DataFrame(results)

            # Summary
            n_fake = (results_df["prediction"] == "Fake").sum()
            n_real = (results_df["prediction"] == "Real").sum()

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Articles", total)
            col2.metric("🔴 Fake Detected", n_fake)
            col3.metric("🟢 Real Detected", n_real)

            st.dataframe(results_df, use_container_width=True)

            # Download button
            csv = results_df.to_csv(index=False)
            st.download_button("⬇️ Download Results CSV", csv,
                               "batch_results.csv", "text/csv")

# ── Footer ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center>Built with ❤️ using Python, Scikit-learn, XGBoost & Streamlit</center>",
    unsafe_allow_html=True
)
