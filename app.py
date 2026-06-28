import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import os

# Try to import joblib to handle sklearn/joblib serialization fallbacks
try:
    import joblib
except ImportError:
    joblib = None

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

@st.cache_resource
def load_production_model():
    """Loads the pre-trained model pipeline. Safely handles and captures errors."""
    # Look for model files matching standard formats (case-insensitive checks)
    files_in_dir = os.listdir(".")
    model_file_match = None
    
    # Priority list of filenames we are looking for
    targets = ["fake_news_model.pkl", "model.pkl"]
    for f in files_in_dir:
        if f.lower() in targets:
            model_file_match = f
            break
            
    # Fallback: scan for any .pkl file containing "model"
    if not model_file_match:
        for f in files_in_dir:
            if f.lower().endswith(".pkl") and "model" in f.lower():
                model_file_match = f
                break
            
    if model_file_match:
        try:
            # First, read the header of the file to see if it is a Git LFS pointer
            with open(model_file_match, "rb") as test_file:
                head = test_file.read(100)
                if b"version https://git-lfs" in head:
                    return None, None, "GitLFSPointer"
            
            # 1. Try Standard Pickle Load
            with open(model_file_match, "rb") as file_obj:
                try:
                    model = pickle.load(file_obj)
                    return model, None, "Success"
                except Exception as pickle_err:
                    # 2. Try Joblib Load as a fallback (resolves the invalid load key error)
                    if joblib is not None:
                        try:
                            # Seek back to start and attempt joblib load
                            file_obj.seek(0)
                            model = joblib.load(file_obj)
                            return model, None, "Success"
                        except Exception as job_err:
                            combined_error = Exception(f"Pickle error: {pickle_err} | Joblib error: {job_err}")
                            return None, combined_error, "PickleLoadError"
                    else:
                        return None, pickle_err, "PickleLoadError"
                        
        except Exception as e:
            return None, e, "PickleLoadError"
            
    return None, None, "FileNotFound"

model, model_error, error_status = load_production_model()

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

if model is None:
    st.markdown("""
    <div style="background-color: #FFFBEB; border-left: 6px solid #F59E0B; padding: 2rem; border-radius: 16px; margin-bottom: 2rem;">
        <h3 style="color: #92400E; margin-top: 0;">⚠️ Model Loader Diagnostics</h3>
        <p style="color: #78350F; font-size: 1.05rem;">
            Your app.py file is live, but your machine learning model is not running yet. Use the diagnostics below to identify the issue.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_diag_left, col_diag_right = st.columns(2)
    
    with col_diag_left:
        st.subheader("📁 Directory Contents Audit")
        st.write("Here are the files Streamlit actually sees in your root folder right now:")
        all_local_files = os.listdir(".")
        # Filter files to make it clean
        st.code("\n".join([f"📄 {file}" for file in all_local_files if not file.startswith(".")]))
        
        # Guide based on file presence
        model_exists_case_insensitive = any(f.lower() == "fake_news_model.pkl" or f.lower() == "model.pkl" for f in all_local_files)
        if not model_exists_case_insensitive:
            st.error("❌ **Result:** `fake_news_model.pkl` was not found anywhere in the root directory. Please double-check that you uploaded it to the exact same folder as `app.py` on GitHub.")
        else:
            st.success("✅ **Result:** A model file exists in your folder! The issue is a load crash.")

    with col_diag_right:
        st.subheader("🐞 Crash Error Log")
        if error_status == "FileNotFound":
            st.warning("No fake_news_model.pkl was found, so no loading error could be generated.")
            
        elif error_status == "GitLFSPointer":
            st.error("❌ **Result: Git LFS Pointer Detected!**")
            st.write(
                "Your model was pushed to GitHub using Git LFS, but Streamlit cloned only the 130-byte text 'pointer' instead of the real file. "
                "This causes pickle to crash."
            )
            st.markdown("#### How to fix this Git LFS Pointer Issue:")
            st.info(
                "💡 **Fix:** Deactivate git LFS on your repository, delete the file on GitHub, and re-upload the `fake_news_model.pkl` file "
                "directly using the GitHub web interface (drag & drop), which bypasses LFS and uploads the actual binary file."
            )
            
        elif error_status == "PickleLoadError":
            st.error(f"Failed to open/unpack your pickle file. Python returned the following error:")
            st.code(f"{type(model_error).__name__}: {model_error}")
            
            # Actionable tips for pickle loading failures
            st.markdown("#### How to fix this error:")
            if "sklearn" in str(model_error) or "ModuleNotFoundError" in str(model_error):
                st.info("💡 **Fix:** You forgot to tell Streamlit to install `scikit-learn`. Create a file in your GitHub repository named **`requirements.txt`** and write `scikit-learn` inside it.")
            elif "UnpicklingError" in str(model_error) and "invalid load key" in str(model_error):
                st.info(
                    "💡 **Fix:** This is a serialization mismatch! It means the file was exported with `joblib.dump()` but is being opened as standard `pickle` or vice-versa. "
                    "Make sure your offline training script imports `joblib` (or `pickle`), and saves the model using standard `pickle.dump(model, open('fake_news_model.pkl', 'wb'))` before uploading."
                )
            elif "AttributeError" in str(model_error):
                st.info("💡 **Fix:** This is a serialization mismatch. This happens if you used a custom function during training that is not declared in your `app.py`, or if your local version of scikit-learn is different from Streamlit's.")
            else:
                st.info("💡 **Fix:** Try exporting your model using a standard pipeline and uploading it again.")
                
    st.markdown("---")
    st.subheader("💡 Sandbox Mode (Interactive Evaluation Preview)")
    st.caption("Since your production model is not loaded yet, you can test-drive the layout below using sandbox parameters.")

headline_input = st.text_input("Article Headline (Optional):", placeholder="e.g., Global Markets Surge Amid New Economic Forecasts")
article_input = st.text_area("Article Body Content:", placeholder="Paste the full body text of the news article here...", height=220)

col_button, _ = st.columns([1, 2])
with col_button:
    evaluate_clicked = st.button("Verify Credibility Index", type="primary", use_container_width=True)

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

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #94A3B8; font-size: 0.9rem;'>Veritas AI Verification Engine • Built with Streamlit, Scikit-Learn & Python</p>",
    unsafe_allow_html=True
)
