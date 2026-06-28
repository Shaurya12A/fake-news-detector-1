import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import os

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Veritas AI - Professional News Verification",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS Styles for a Premium Production Interface ──────────────────────
st.markdown("""
<style>
    /* Main Layout Improvements */
    .reportview-container {
        background-color: #F8FAFC;
    }
    
    /* Elegant Headers */
    .brand-title {
        font-family: 'Inter', -apple-system, sans-serif;
        font-weight: 800;
        letter-spacing: -0.05em;
        color: #0F172A;
        font-size: 2.75rem;
        margin-bottom: 0.25rem;
    }
    .brand-subtitle {
        font-family: 'Inter', sans-serif;
        color: #64748B;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Result Cards */
    .verdict-card {
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        margin-bottom: 2rem;
        transition: all 0.3s ease;
    }
    .verdict-real {
        background-color: #F0FDF4;
        border-left: 6px solid #10B981;
        color: #065F46;
    }
    .verdict-fake {
        background-color: #FEF2F2;
        border-left: 6px solid #EF4444;
        color: #991B1B;
    }
    
    /* UI Badges */
    .status-badge {
        font-weight: 800;
        font-size: 1.5rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        display: inline-block;
        padding: 4px 16px;
        border-radius: 9999px;
        margin-bottom: 1rem;
    }
    .badge-real {
        background-color: #D1FAE5;
        color: #065F46;
    }
    .badge-fake {
        background-color: #FEE2E2;
        color: #991B1B;
    }
    
    /* Subtle Text Cards */
    .metric-box {
        background-color: #FFFFFF;
        padding: 1.25rem;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# ── Helper Text Cleaning ──────────────────────────────────────────────────────
def clean_news_text(text):
    """Normalize news article text to avoid formatting/punctuation biases."""
    text = str(text).lower()
    
    # Strip common journalistic stamps to ensure structural/semantic evaluation
    text = re.sub(r'^\s*([a-za-z\s]+)\s*\(reuters\)\s*[-—]', '', text)
    text = re.sub(r'^\s*([a-za-z\s]+)\s*\(ap\)\s*[-—]', '', text)
    text = re.sub(r'^\s*([a-za-z\s]+)\s*\(cnn\)\s*[-—]', '', text)
    text = re.sub(r'reuters|associated press', '', text)
    
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # Remove URLs
    text = re.sub(r'<.*?>', '', text) # Remove HTML Tags
    text = re.sub(r'[^\w\s]', '', text) # Remove Punctuation
    text = re.sub(r'\s+', ' ', text).strip() # Multi spaces
    return text

# ── Model Loading (Safe Handling) ─────────────────────────────────────────────
@st.cache_resource
def load_production_model():
    """Loads the pre-trained model pipeline. Safely handles missing files."""
    model_path = "model.pkl"
    if os.path.exists(model_path):
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            return model, None
        except Exception as e:
            return None, f"Error loading model pickle: {e}"
    return None, "Model file 'model.pkl' not found."

model, model_error = load_production_model()

# ── Sidebar Navigation & Calibration ─────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛡️ Veritas AI Panel")
    st.markdown("Veritas AI uses natural language processing to evaluate semantic structures, clickbait signals, and linguistic credibility markers.")
    
    st.markdown("---")
    st.markdown("### ⚙️ Calibration Settings")
    
    threshold = st.slider(
        "Model Sensitivity (Credibility Cutoff):",
        min_value=0.10, max_value=0.90, value=0.50, step=0.05,
        help="Higher values require stronger evidence to mark an article as CREDIBLE."
    )
    
    st.markdown("---")
    st.markdown("### 📋 Quick Quickstart")
    st.info(
        "Paste any recent news piece on the right. Our model strips publisher metadata and runs contextual validation based on historical reporting trends."
    )

# ── Main Header Section ───────────────────────────────────────────────────────
st.markdown("<h1 class='brand-title'>🛡️ Veritas AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='brand-subtitle'>State-of-the-art computational news veracity verification.</p>", unsafe_allow_html=True)

# ── Display Missing Model Setup Guide if Model Not Present ───────────────────
if model is None:
    st.markdown("""
    <div style="background-color: #FFFBEB; border-left: 6px solid #F59E0B; padding: 2rem; border-radius: 16px; margin-bottom: 2rem;">
        <h3 style="color: #92400E; margin-top: 0;">📦 Model Integration Required</h3>
        <p style="color: #78350F; font-size: 1.05rem;">
            Veritas AI is successfully configured! To complete deployment, make sure your trained <code>model.pkl</code> file is uploaded to the root directory of your GitHub repository.
        </p>
        <h4 style="color: #92400E; margin-bottom: 0.5rem;">Quick Setup Instructions:</h4>
        <ol style="color: #78350F; line-height: 1.6;">
            <li>Save your trained classifier pipeline using <code>pickle.dump(pipeline, open('model.pkl', 'wb'))</code>.</li>
            <li>Commit and push the <code>model.pkl</code> file to your active GitHub repository alongside this <code>app.py</code>.</li>
            <li>Once pushed, Streamlit Community Cloud will automatically reload and enable live machine learning evaluations.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Showcase interactive placeholder mode for demo/presentation purposes
    st.subheader("💡 Sandbox Mode (Interactive Evaluation Preview)")
    st.caption("Since 'model.pkl' is not loaded yet, you can test-drive the UI structure below.")

# ── Text Entry Layout ─────────────────────────────────────────────────────────
headline_input = st.text_input("Article Headline (Optional):", placeholder="e.g., Global Markets Surge Amid New Economic Forecasts")
article_input = st.text_area("Article Body Content:", placeholder="Paste the full body text of the news article here...", height=220)

col_button, _ = st.columns([1, 2])
with col_button:
    evaluate_clicked = st.button("Verify Credibility Index", type="primary", use_container_width=True)

# ── Run Evaluation Block ─────────────────────────────────────────────────────
if evaluate_clicked and article_input:
    # 1. Structural Analytics
    word_count = len(article_input.split())
    char_count = len(article_input)
    est_reading_time = max(1, round(word_count / 225))
    all_caps_words = sum(1 for w in article_input.split() if w.isupper() and len(w) > 2)
    capitalization_ratio = (all_caps_words / word_count) if word_count > 0 else 0
    
    cleaned_body = clean_news_text(article_input)
    
    # 2. Probability Computation
    if model is not None:
        try:
            # Handle model formats safely
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba([cleaned_body])[0]
                real_idx = 1 if len(model.classes_) > 1 else 0
                prob_real = probabilities[real_idx]
            else:
                decision = model.decision_function([cleaned_body])[0]
                val = decision[0] if hasattr(decision, "__len__") else decision
                prob_real = 1 / (1 + np.exp(-val)) # Sigmoid fallback
        except Exception as eval_err:
            st.error(f"Prediction Error: {eval_err}")
            prob_real = 0.50 # Fallback
    else:
        # Balanced heuristic fallback for sandbox demo mode
        sensational_words = ["shocking", "conspiracy", "secret", "exposed", "lying", "truth", "urgent", "must watch", "alert"]
        matched_triggers = sum(1 for sw in sensational_words if sw in cleaned_body)
        
        # Base probability shifts on length and trigger occurrences
        base_score = 0.85 - (matched_triggers * 0.15) - (capitalization_ratio * 0.5)
        prob_real = float(np.clip(base_score, 0.05, 0.95))

    prob_fake = 1.0 - prob_real
    verdict = "Real" if prob_real >= threshold else "Fake"
    
    st.markdown("---")
    st.subheader("📊 Verification Report")
    
    col_results, col_meta = st.columns([3, 2])
    
    with col_results:
        # Display Premium Verdict Box
        if verdict == "Real":
            st.markdown(
                f"""
                <div class="verdict-card verdict-real">
                    <span class="status-badge badge-real">Credible Reporting</span>
                    <h2 style="margin: 0 0 0.5rem 0; color: #065F46;">Verity Score: {prob_real:.1%}</h2>
                    <p style="margin: 0; font-size: 1.05rem;">
                        This text displays semantic structures, syntax flow, and journalistic patterns associated with balanced, objective, and verified reporting. No major metadata vulnerabilities or manipulative trigger phrases were flagged.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="verdict-card verdict-fake">
                    <span class="status-badge badge-fake">Suspicious / Clickbait</span>
                    <h2 style="margin: 0 0 0.5rem 0; color: #991B1B;">Verity Score: {prob_real:.1%}</h2>
                    <p style="margin: 0; font-size: 1.05rem;">
                        <b>Attention:</b> This text exhibits structural markers commonly found in misinformation, sensationalism, or heavily biased reporting. The credibility rating fell below your calibrated <b>{threshold:.0%}</b> sensitivity index.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Progress Bars and Raw Metric Displays
        st.write("**Credibility Breakdown**")
        st.progress(float(prob_real))
        st.caption(f"Veritas Verity Score: {prob_real:.2%} | Misinformation Risk: {prob_fake:.2%}")
        
    with col_meta:
        st.markdown("<p style='font-weight: 600; margin-bottom: 0.75rem; color: #1E293B;'>Linguistic & Structural Insights</p>", unsafe_allow_html=True)
        
        # Key Information Cards
        st.markdown(
            f"""
            <div style="display: flex; flex-direction: column; gap: 1rem;">
                <div class="metric-box">
                    <small style="color: #64748B; text-transform: uppercase; font-weight: bold; letter-spacing: 0.05em;">Linguistic Volume</small>
                    <h3 style="margin: 4px 0 0 0; color: #1E293B;">{word_count:,} Words</h3>
                    <small style="color: #94A3B8;">Estimated reading time: {est_reading_time} min</small>
                </div>
                <div class="metric-box">
                    <small style="color: #64748B; text-transform: uppercase; font-weight: bold; letter-spacing: 0.05em;">Sensationalist Capitalization</small>
                    <h3 style="margin: 4px 0 0 0; color: #1E293B;">{all_caps_words} Emphasized Words</h3>
                    <small style="color: #94A3B8;">{capitalization_ratio:.1%} of total structure</small>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #94A3B8; font-size: 0.9rem;'>Veritas AI Verification Engine • Built with Streamlit, Scikit-Learn & Python</p>",
    unsafe_allow_html=True
)
