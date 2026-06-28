import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector (Diagnostic Mode)",
    page_icon="🔍",
    layout="wide"
)

# ── Custom CSS Styles for Premium Modern Interface ────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-family: 'Helvetica Neue', Arial, sans-serif;
        font-weight: 800;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 5px;
    }
    .sub-header {
        font-size: 1.15rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
        border-left: 5px solid #3B82F6;
    }
    .card-real {
        border-left-color: #10B981 !important;
        background-color: #F0FDF4;
    }
    .card-fake {
        border-left-color: #EF4444 !important;
        background-color: #FEF2F2;
    }
    .card-warning {
        border-left-color: #F59E0B !important;
        background-color: #FFFBEB;
    }
    .badge-real {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .badge-fake {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# ── Helper Clean Text Functions ───────────────────────────────────────────────
def clean_news_text(text, remove_sources=False):
    """Clean text to normalize casing, remove URLs, special characters and optionally sources."""
    text = str(text).lower()
    
    # Simulate stripping standard journalistic agency stamps/locations (e.g. "WASHINGTON (Reuters) - ...")
    if remove_sources:
        text = re.sub(r'^\s*([a-za-z\s]+)\s*\(reuters\)\s*[-—]', '', text)
        text = re.sub(r'^\s*([a-za-z\s]+)\s*\(ap\)\s*[-—]', '', text)
        text = re.sub(r'^\s*([a-za-z\s]+)\s*\(cnn\)\s*[-—]', '', text)
        text = re.sub(r'reuters|associated press', '', text)
        
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # Remove URLs
    text = re.sub(r'<.*?>', '', text) # Remove HTML Tags
    text = re.sub(r'[^\w\s]', '', text) # Remove Punctuation
    text = re.sub(r'\s+', ' ', text).strip() # Multi spaces
    return text

# ── Interactive Bias Simulator (In-Memory Model Generator) ───────────────────
@st.cache_resource
def train_mock_model(introduce_bias=True):
    """
    Trains a real interactive toy model to demonstrate the "Reuters Trap".
    If introduce_bias is True:
      - All Real news mentions "Reuters" (creates severe feature leakage).
      - All Fake news uses dramatic capitalization/no source.
    """
    if introduce_bias:
        real_data = [
            "WASHINGTON (Reuters) - Congress passed the bipartisan infrastructure package on Friday after days of debates.",
            "LONDON (Reuters) - The bank of England announced a small interest rate adjustment to control inflation.",
            "TOKYO (Reuters) - Japan's trade surplus increased by twelve percent due to high automotive shipments abroad.",
            "PARIS (Reuters) - European leaders met to discuss mutual renewable energy goals and climate targets.",
            "GENEVA (Reuters) - The World Health Organization launched a new health security framework today."
        ]
        fake_data = [
            "SHOCKING TRUTH: You won't believe what Congress just did behind closed doors!!! Click to watch!",
            "THEY ARE LYING TO YOU! Secret leaks show the global banks are planning to freeze all currency assets soon!",
            "SENSATIONAL: New experimental fuel engines run entirely on tap water and the government is hiding it!",
            "CONFIRMED: Mysterious lights above Paris are proof of hidden energy technologies being tested.",
            "ALERT! Hidden toxic chemicals found in public systems, independent researchers warn the community."
        ]
    else:
        # Cleaned balanced data without source signatures
        real_data = [
            "Congress passed the bipartisan infrastructure package on Friday after days of debates.",
            "The Bank of England announced a small interest rate adjustment to control inflation.",
            "Japan's trade surplus increased by twelve percent due to high automotive shipments abroad.",
            "European leaders met to discuss mutual renewable energy goals and climate targets.",
            "The World Health Organization launched a new health security framework today."
        ]
        fake_data = [
            "Congress just voted on a highly controversial bill without public consultation or review.",
            "Global monetary structures are facing unprecedented shifts due to experimental reserve policies.",
            "Alternative tech developers announce motor configurations capable of running on natural compounds.",
            "Unidentified sightings above European capital cities spark discussions among weather tracking researchers.",
            "Private laboratory claims to have identified trace anomalies in regional drinking supplies."
        ]

    X = real_data + fake_data
    y = np.array([1] * len(real_data) + [0] * len(fake_data)) # 1 = Real, 0 = Fake
    
    vectorizer = TfidfVectorizer(stop_words='english')
    X_vec = vectorizer.fit_transform(X)
    
    model = LogisticRegression()
    model.fit(X_vec, y)
    
    return model, vectorizer

# ── UI Elements & Title ───────────────────────────────────────────────────────
st.markdown("<h1 class='main-header'>🔍 Fake News Detector Diagnostic Center</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Is your NLP model biased? Run interactive checks, calibrate thresholds, and audit vocabulary matching.</p>", unsafe_allow_html=True)

# Create layout columns
col_sidebar, col_main = st.columns([1, 2])

with col_sidebar:
    st.header("⚙️ Model Setup")
    
    model_source = st.radio(
        "Choose Model Input Source:",
        ["1. Built-in Bias Simulator (Recommended to see the bug)", "2. Upload my custom .pkl files"]
    )
    
    model = None
    vectorizer = None
    is_custom_model = False
    is_pipeline = False
    
    if "1." in model_source:
        st.info("💡 Mode 1 trains a toy TF-IDF + Logistic Regression model directly in RAM. You can toggle 'Reuters bias' to see feature leakage.")
        bias_toggle = st.checkbox("Introduce 'Reuters Trap' Bias into data", value=True, 
                                  help="When active, the model will learn that any sentence without '(Reuters)' is Fake news.")
        model, vectorizer = train_mock_model(introduce_bias=bias_toggle)
        st.success("🤖 Simulator Model Trained Successfully!")
        
    else:
        is_custom_model = True
        st.info("📂 Upload your trained artifacts. If your model contains the vectorization pipeline internally, you do not need to upload a separate vectorizer file.")
        
        uploaded_model = st.file_uploader("Upload Model File (model.pkl)", type=["pkl", "bin", "sav"])
        uploaded_vec = st.file_uploader("Upload Vectorizer File (Optional - vectorizer.pkl)", type=["pkl", "bin", "sav"])
        
        if uploaded_model:
            try:
                model = pickle.load(uploaded_model)
                st.success("✅ Model loaded successfully!")
                
                # Check if loaded model is a Scikit-Learn Pipeline or wraps vectorization steps
                model_class_name = type(model).__name__
                if "Pipeline" in model_class_name or hasattr(model, "steps"):
                    is_pipeline = True
                    st.info("📦 **Pipeline Detected:** Your model includes built-in vectorization. A separate vectorizer file is NOT required.")
                    
                    # Try to extract the vectorizer from the pipeline steps for vocabulary diagnostics
                    for step_name, step_obj in model.steps:
                        if hasattr(step_obj, "get_feature_names_out") or hasattr(step_obj, "vocabulary_"):
                            vectorizer = step_obj
                            st.caption(f"ℹ️ Found internal vectorizer step: `{step_name}` ({type(step_obj).__name__})")
                            break
                else:
                    st.warning("⚠️ This model does not appear to be a Pipeline. You will likely need to upload a separate vectorizer file below.")
                
                # Load custom vectorizer if manually supplied
                if uploaded_vec:
                    vectorizer = pickle.load(uploaded_vec)
                    is_pipeline = False # Explicit vectorizer provided overrides internal pipeline extraction
                    st.success("✅ Custom vectorizer loaded successfully!")
                    
            except Exception as e:
                st.error(f"❌ Failed to load artifacts: {e}")
                st.warning("Please ensure your model was serialized properly in Python.")

    st.markdown("---")
    st.header("🧪 Diagnostic Audits")
    
    threshold = st.slider(
        "Decision Threshold for 'Real' news:",
        min_value=0.05, max_value=0.95, value=0.50, step=0.05,
        help="Default is 0.50. Raise the threshold to require stronger proof before a document is classified as Real."
    )
    
    strip_source = st.checkbox(
        "Apply Source Stripping (Simulated clean)", 
        value=False,
        help="Strips terms like 'Reuters' and 'Associated Press' before passing to the model."
    )

with col_main:
    st.header("📝 Run News Analysis")
    
    # Preset triggers for easy debugging
    preset_news = st.selectbox(
        "Quick Paste Standard Test Cases:",
        [
            "--- Select a Preset News Story ---",
            "WASHINGTON (Reuters) - The President signed an executive order regarding supply chain cybersecurity today.",
            "The President signed an executive order regarding supply chain cybersecurity today. (No source header)",
            "CRITICAL WARNING: The monetary system is collapsing! Move your money immediately!",
            "Researchers find unexpected variations in local ecosystem growth during summer monitoring."
        ]
    )
    
    input_text = st.text_area(
        "Enter News Article text to diagnose:", 
        value="" if preset_news == "--- Select a Preset News Story ---" else preset_news,
        height=180,
        placeholder="Paste an article here..."
    )
    
    run_diagnostics = st.button("Run Model Diagnostics", type="primary", use_container_width=True)
    
    if run_diagnostics and input_text:
        if model is None:
            st.error("⚠️ Please upload your model file first.")
        elif not is_pipeline and vectorizer is None:
            st.error("⚠️ This model requires an external vectorizer. Please upload a vectorizer.pkl file on the sidebar.")
        else:
            # Step 1: Pre-process
            cleaned_text_raw = clean_news_text(input_text, remove_sources=False)
            cleaned_text_sanitized = clean_news_text(input_text, remove_sources=strip_source)
            
            # Show original vs cleaned string comparison
            if strip_source:
                st.info(f"**Sanitization Enabled:** Evaluated text stripped of source headers.\n* **Before:** `'{cleaned_text_raw[:90]}...'` \n* **After:** `'{cleaned_text_sanitized[:90]}...'` ")
            
            # Step 2: Transform input vector
            try:
                if is_pipeline:
                    # Pipelines take raw strings directly as input
                    model_input = [cleaned_text_sanitized]
                else:
                    # Standalone models take vectorized input arrays
                    model_input = vectorizer.transform([cleaned_text_sanitized])
                
                # Calculate class probabilities
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(model_input)[0]
                    # Map classes. Usually Class 1 is Real, Class 0 is Fake.
                    real_class_idx = 1 if len(model.classes_) > 1 else 0
                    prob_real = probs[real_class_idx]
                    prob_fake = 1.0 - prob_real
                else:
                    # Fallback for models without predict_proba (like SVM without probability=True)
                    decision = model.decision_function(model_input)[0]
                    # Handle multi-class vs binary output shape in decision_function
                    val = decision[0] if hasattr(decision, "__len__") else decision
                    prob_real = 1 / (1 + np.exp(-val)) # Sigmoid mapping
                    prob_fake = 1.0 - prob_real
                
                # Apply custom threshold adjustment
                prediction = "Real" if prob_real >= threshold else "Fake"
                
                # ── SECTION A: Live Prediction Results ────────────────────────
                st.markdown("### 📊 Prediction Verdict")
                
                if prediction == "Real":
                    st.markdown(
                        f"""
                        <div class='card card-real'>
                            <h3>Verdict: <span class='badge-real'>REAL</span></h3>
                            <p>This text meets the <b>{threshold:.0%}</b> probability threshold to be categorized as credible news.</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div class='card card-fake'>
                            <h3>Verdict: <span class='badge-fake'>FAKE / SUSPICIOUS</span></h3>
                            <p>This text did not meet the <b>{threshold:.0%}</b> threshold required to be marked as credible. It was classified as Fake/Suspicious.</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                
                # Display probability visual breakdown
                col_prob1, col_prob2 = st.columns(2)
                with col_prob1:
                    st.metric("Credibility Rating (Real)", f"{prob_real:.2%}")
                    st.progress(float(prob_real))
                with col_prob2:
                    st.metric("Bias / Clickbait Rating (Fake)", f"{prob_fake:.2%}")
                    st.progress(float(prob_fake))
                    
                # ── SECTION B: Vocabulary Matcher (The Crucial Diagnosis) ─────
                st.markdown("### 🔍 Feature Vocabulary Overlap Analysis")
                
                if vectorizer is not None:
                    # Tokenize the user's input words
                    words = cleaned_text_sanitized.split()
                    
                    try:
                        feature_names = vectorizer.get_feature_names_out()
                    except AttributeError:
                        # Fallback for older vectorizer versions
                        feature_names = vectorizer.get_feature_names() if hasattr(vectorizer, "get_feature_names") else []
                        
                    vocab_set = set(feature_names)
                    
                    if len(vocab_set) > 0:
                        matching_words = [w for w in words if w in vocab_set]
                        ignored_words = [w for w in words if w not in vocab_set]
                        
                        overlap_percent = len(matching_words) / len(words) if len(words) > 0 else 0
                        
                        st.write(f"**Text Length:** {len(words)} words | **Matching Vocabulary Features:** {len(matching_words)} words")
                        
                        # Display warning if overlap is exceptionally low
                        if overlap_percent < 0.15:
                            st.markdown(
                                f"""
                                <div class='card card-warning'>
                                    ⚠️ <b>Critical Warning: Low Vocabulary Overlap ({overlap_percent:.1%})</b><br/>
                                    Your vectorizer only recognized <b>{len(matching_words)} out of {len(words)} words</b>. 
                                    Because the vocabulary doesn't match the training data, the vector is almost empty, 
                                    forcing the model to default to its baseline mathematical bias (which is often to predict 'Fake'). 
                                    Ensure you aren't resetting the vectorizer using <code>fit_transform()</code> in your deployment code!
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        else:
                            st.success(f"✅ Healthy Vocabulary Overlap: **{overlap_percent:.1%}** of the words matched words in the vectorizer vocabulary.")
                        
                        # Expandable lists for detailed debugging
                        with st.expander("Show detailed word-by-word vocabulary audit"):
                            col_word1, col_word2 = st.columns(2)
                            with col_word1:
                                st.subheader("Words Recognized:")
                                st.write(list(set(matching_words)) if matching_words else "None")
                            with col_word2:
                                st.subheader("Words Ignored/Unknown:")
                                st.write(list(set(ignored_words)) if ignored_words else "None")
                    else:
                        st.info("ℹ️ Vocabulary extracted, but vocabulary set is empty or could not be decoded.")
                else:
                    st.info("ℹ️ Vocabulary overlap diagnostics are unavailable because your custom model uses an encapsulated internal vectorizer that couldn't be extracted. Prediction features are still active!")
                        
                # ── SECTION C: Troubleshooting Recommendations ───────────────
                st.markdown("### 🛠️ Recommended Action Items")
                
                recommendations = []
                
                # Diagnose Reuters trap
                if "reuters" in cleaned_text_raw and not strip_source and prob_real > 0.80:
                    recommendations.append(
                        "🚨 **The Reuters Trap Detected:** Your text contains 'reuters' and scored very high as Real. Try ticking the **'Apply Source Stripping'** box on the sidebar. If the prediction drops to 'Fake', your model is cheating by looking at source headers instead of real grammar patterns. You need to strip source names from your dataset and retrain."
                    )
                
                # Class boundary troubleshooting
                if 0.40 <= prob_real <= 0.60:
                    recommendations.append(
                        "⚖️ **Close Classification Margin:** The model calculated a near-fifty-fifty probability. Adjusting the **Decision Threshold Slider** on the sidebar can help calibrate this app for strict vs relaxed criteria."
                    )
                
                if "1." in model_source and bias_toggle:
                    recommendations.append(
                        "💡 **Experience Bias:** Try toggling off 'Introduce Reuters Trap Bias' on the sidebar. This will retrain the internal model on cleaned data. Analyze the exact same text again and watch the score become balanced!"
                    )
                
                if is_custom_model:
                    if is_pipeline:
                        recommendations.append(
                            "📦 **Pipeline Verified:** Your model was detected as an active Scikit-Learn Pipeline. It is receiving raw text and transforming it internally. This eliminates the chance of the 'Vectorizer Re-Fitting Bug', meaning your 'All Fake' issue is likely caused by **Reuters Trap bias** or **Severe Class Imbalance** during your offline training stage."
                        )
                    else:
                        recommendations.append(
                            "📦 **Standalone Model Audit:** Since you are using a separate vectorizer, make sure your Streamlit file uses `vectorizer.transform()` and never `fit_transform()`. Calling `fit_transform()` resets the internal feature mapping and will make your custom model classify everything as Fake."
                        )
                
                if not recommendations:
                    recommendations.append("✅ **No obvious visual artifacts found.** Your model is responding confidently with consistent vocabulary overlap!")
                    
                for rec in recommendations:
                    st.write(rec)
                    
            except Exception as eval_err:
                st.error(f"⚠️ Prediction Error: {eval_err}")
                st.info("If your file is a pipeline, ensure that you do not pass a vectorized matrix to it. If it is a standalone model, make sure your vectorizer matches the input size expected by the model.")
