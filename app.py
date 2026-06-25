import streamlit as st
import joblib
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    .main-title {
        font-size: 2.6rem; font-weight: 800;
        text-align: center; color: #ffffff; margin-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center; color: #888;
        font-size: 1rem; margin-bottom: 2rem;
    }
    .result-fake {
        background: linear-gradient(135deg, #ff4b4b22, #ff4b4b44);
        border: 1.5px solid #ff4b4b; border-radius: 12px;
        padding: 1.2rem 1.5rem; text-align: center; margin-top: 1.5rem;
    }
    .result-real {
        background: linear-gradient(135deg, #00c85322, #00c85344);
        border: 1.5px solid #00c853; border-radius: 12px;
        padding: 1.2rem 1.5rem; text-align: center; margin-top: 1.5rem;
    }
    .result-label { font-size: 2rem; font-weight: 800; margin-bottom: 0.3rem; }
    .result-confidence { font-size: 1rem; color: #cccccc; }
    .warning-box {
        background-color: #2a2000; border: 1.5px solid #f0a500;
        border-radius: 10px; padding: 0.8rem 1.2rem;
        font-size: 0.9rem; color: #f0c040; margin-top: 1rem; text-align: center;
    }
    .stTextArea textarea {
        background-color: #1c1e26 !important; color: #f0f0f0 !important;
        border: 1px solid #333 !important; border-radius: 10px !important;
        font-size: 0.95rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white; border: none; border-radius: 10px;
        padding: 0.6rem 2rem; font-size: 1rem;
        font-weight: 600; width: 100%;
    }
    .stButton > button:hover { opacity: 0.88; }
    .info-box {
        background-color: #1c1e26; border-radius: 10px;
        padding: 0.8rem 1.2rem; font-size: 0.85rem;
        color: #aaa; margin-top: 1.5rem; text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "fake_news_model.pkl"
    if not os.path.exists(model_path):
        st.error("❌ Model file not found. Make sure `fake_news_model.pkl` is in the same folder as `app.py`.")
        st.stop()
    return joblib.load(model_path)

model = load_model()

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🔍 Fake News Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Paste a full news article to check if it\'s real or fake</div>', unsafe_allow_html=True)

news_input = st.text_area(
    label="News Text",
    placeholder="Paste the full news article here (not just a headline)...",
    height=200,
    label_visibility="collapsed"
)

# ── Word count warning ────────────────────────────────────────────────────────
MIN_WORDS = 30
word_count = len(news_input.strip().split()) if news_input.strip() else 0

if news_input.strip() and word_count < MIN_WORDS:
    st.markdown(f"""
    <div class="warning-box">
        ⚠️ Too short to analyze accurately ({word_count} words). 
        Please paste the <strong>full article</strong> — at least {MIN_WORDS} words recommended.
        Short headlines will almost always show as FAKE.
    </div>
    """, unsafe_allow_html=True)

if st.button("Analyze"):
    if not news_input.strip():
        st.warning("⚠️ Please enter some news text first.")
    elif word_count < MIN_WORDS:
        st.error(f"❌ Text too short ({word_count} words). Paste the full article for accurate results.")
    else:
        with st.spinner("Analyzing..."):
            prediction = model.predict([news_input])[0]
            confidence = model.predict_proba([news_input])[0]
            confidence_pct = round(max(confidence) * 100, 2)

        if prediction == 1:
            st.markdown(f"""
            <div class="result-fake">
                <div class="result-label">🔴 FAKE NEWS</div>
                <div class="result-confidence">Model is <strong>{confidence_pct}%</strong> confident this is fake</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-real">
                <div class="result-label">🟢 REAL NEWS</div>
                <div class="result-confidence">Model is <strong>{confidence_pct}%</strong> confident this is real</div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    Trained on 44,000+ articles · TF-IDF + Logistic Regression · Accuracy: 99.46%<br>
    ⚠️ Works best with full articles, not short headlines
</div>
""", unsafe_allow_html=True)
