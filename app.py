"""
app.py
------
Production-ready Streamlit web application for Twitter Sentiment Analysis.
Features:
- Multi-model inference: Multinomial Naive Bayes, SimpleRNN, LSTM, GRU
- Resilient fallback: works with or without TensorFlow
- Real-time single tweet analysis with confidence gauge & probability distribution
- Batch tweet analysis (CSV upload & multi-line input) with CSV export
- Model benchmark leaderboard and visual accuracy comparison
- Polished modern UI with custom CSS and dark/light mode compatibility
"""

import os
import json
import time
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from preprocess import clean_text

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="PulseTweet AI · Twitter Sentiment Analysis",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_DIR = "models"
OUTPUTS_DIR = "outputs"
MAX_LEN = 40

SENTIMENT_THEMES = {
    "Positive": {
        "color": "#10B981",
        "bg_dark": "rgba(16, 185, 129, 0.15)",
        "border": "#10B981",
        "icon": "😊",
        "desc": "Expresses satisfaction, excitement, optimism, or approval."
    },
    "Negative": {
        "color": "#EF4444",
        "bg_dark": "rgba(239, 68, 68, 0.15)",
        "border": "#EF4444",
        "icon": "😡",
        "desc": "Expresses frustration, criticism, disappointment, or anger."
    },
    "Neutral": {
        "color": "#3B82F6",
        "bg_dark": "rgba(59, 130, 246, 0.15)",
        "border": "#3B82F6",
        "icon": "⚖️",
        "desc": "Factual, objective, news, or neutral inquiry without sentiment."
    },
    "Irrelevant": {
        "color": "#A855F7",
        "bg_dark": "rgba(168, 85, 247, 0.15)",
        "border": "#A855F7",
        "icon": "🎯",
        "desc": "Off-topic, non-sentiment conversational noise or unlinked text."
    },
}

SAMPLE_TWEETS = [
    {
        "label": "Positive",
        "icon": "😊",
        "text": "I absolutely love this new update! Everything runs so smoothly and the new design looks incredible. Huge thumbs up to the team! 👍✨"
    },
    {
        "label": "Negative",
        "icon": "😡",
        "text": "Worst customer service experience ever! Waited over an hour on hold and then the agent hung up on me. Completely unacceptable. 🤬"
    },
    {
        "label": "Neutral",
        "icon": "⚖️",
        "text": "Microsoft announced their quarterly financial earnings report this Tuesday morning at 10 AM EST."
    },
    {
        "label": "Irrelevant",
        "icon": "🎯",
        "text": "Just grabbed an iced caramel macchiato and a butter croissant from the cafe down the block."
    }
]

# -----------------------------------------------------------------------------
# Custom Styling
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
/* Font & Base layout */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Hero Header */
.hero-container {
    background: linear-gradient(135deg, #0b1426 0%, #1e293b 50%, #0f172a 100%);
    border: 1px solid rgba(29, 155, 240, 0.3);
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    border-radius: 16px;
    padding: 26px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    color: #FFFFFF;
    margin: 0 0 6px 0;
    display: flex;
    align-items: center;
    gap: 12px;
}
.hero-subtitle {
    color: #94A3B8;
    font-size: 1.02rem;
    margin: 0;
    font-weight: 400;
}
.hero-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 9999px;
    background: rgba(29, 155, 240, 0.2);
    border: 1px solid #1D9BF0;
    color: #38BDF8;
    font-size: 0.8rem;
    font-weight: 600;
    margin-top: 10px;
}

/* Result Card */
.result-box {
    border-radius: 16px;
    padding: 24px 28px;
    margin-top: 18px;
    margin-bottom: 20px;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
}
.result-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
}
.sentiment-title {
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0;
}
.conf-badge {
    text-align: right;
}
.conf-number {
    font-size: 2.4rem;
    font-weight: 800;
    line-height: 1;
}
.conf-label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94A3B8;
    font-weight: 700;
    margin-top: 4px;
}
.meta-chips {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-top: 16px;
    padding-top: 14px;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
}
.chip {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.12);
    padding: 4px 10px;
    border-radius: 8px;
    font-size: 0.82rem;
    color: #CBD5E1;
}

/* Probability Bars */
.prob-card {
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 18px 22px;
    margin-top: 12px;
}
.prob-item {
    margin-bottom: 12px;
}
.prob-header-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.92rem;
    font-weight: 600;
    margin-bottom: 6px;
}
.prob-bar-track {
    width: 100%;
    height: 10px;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 9999px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 9999px;
    transition: width 0.6s ease;
}

/* Metric Cards */
.kpi-card {
    background: rgba(30, 41, 59, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}
.kpi-value {
    font-size: 1.8rem;
    font-weight: 800;
}
.kpi-label {
    font-size: 0.8rem;
    color: #94A3B8;
    text-transform: uppercase;
    font-weight: 600;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Model & Asset Loaders (Cached)
# -----------------------------------------------------------------------------
@st.cache_resource
def check_tensorflow_support():
    """Detect if TensorFlow / Keras can be safely imported without crashing."""
    try:
        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        return True, None
    except Exception as e:
        return False, str(e)


@st.cache_resource
def load_label_encoder():
    path = os.path.join(MODEL_DIR, "label_encoder.joblib")
    if os.path.exists(path):
        return joblib.load(path)
    return None


@st.cache_resource
def load_naive_bayes():
    nb_path = os.path.join(MODEL_DIR, "naive_bayes_model.joblib")
    vec_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
    acc_path = os.path.join(MODEL_DIR, "nb_accuracy.json")

    model = joblib.load(nb_path)
    vectorizer = joblib.load(vec_path)
    acc = 0.8014
    if os.path.exists(acc_path):
        with open(acc_path) as f:
            acc = json.load(f).get("accuracy", 0.8014)
    return model, vectorizer, acc


@st.cache_resource
def load_dl_model(model_type: str):
    """
    Load RNN, LSTM, or GRU model + tokenizer.
    model_type in ['rnn', 'lstm', 'gru']
    """
    has_tf, _ = check_tensorflow_support()
    if not has_tf:
        raise RuntimeError("TensorFlow is not available in the current environment.")

    from tensorflow.keras.models import load_model

    model_path = os.path.join(MODEL_DIR, f"{model_type}_model.keras")
    tok_path = os.path.join(MODEL_DIR, f"{model_type}_tokenizer.joblib")
    acc_path = os.path.join(MODEL_DIR, f"{model_type}_accuracy.json")

    model = load_model(model_path)
    tokenizer = joblib.load(tok_path)

    acc = 0.93
    if os.path.exists(acc_path):
        with open(acc_path) as f:
            acc = json.load(f).get("accuracy", 0.93)

    return model, tokenizer, acc


# -----------------------------------------------------------------------------
# Prediction Functions
# -----------------------------------------------------------------------------
def predict_with_nb(text: str, label_encoder):
    start = time.perf_counter()
    model, vectorizer, accuracy = load_naive_bayes()
    cleaned = clean_text(text)
    if not cleaned:
        cleaned = "empty"
    features = vectorizer.transform([cleaned])
    probs = model.predict_proba(features)[0]
    idx = np.argmax(probs)
    label = label_encoder.inverse_transform([idx])[0]
    latency_ms = (time.perf_counter() - start) * 1000
    return label, probs, accuracy, latency_ms, cleaned


def predict_with_dl(text: str, model_type: str, label_encoder):
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    start = time.perf_counter()
    model, tokenizer, accuracy = load_dl_model(model_type)
    cleaned = clean_text(text)
    if not cleaned:
        cleaned = "empty"
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    probs = model.predict(padded, verbose=0)[0]
    idx = np.argmax(probs)
    label = label_encoder.inverse_transform([idx])[0]
    latency_ms = (time.perf_counter() - start) * 1000
    return label, probs, accuracy, latency_ms, cleaned


def run_prediction(text: str, chosen_model: str, label_encoder):
    if chosen_model == "Multinomial Naive Bayes":
        return predict_with_nb(text, label_encoder)
    elif chosen_model == "SimpleRNN":
        return predict_with_dl(text, "rnn", label_encoder)
    elif chosen_model == "LSTM":
        return predict_with_dl(text, "lstm", label_encoder)
    elif chosen_model == "GRU":
        return predict_with_dl(text, "gru", label_encoder)
    else:
        return predict_with_nb(text, label_encoder)


# -----------------------------------------------------------------------------
# Main Application
# -----------------------------------------------------------------------------
def main():
    has_tf, tf_err = check_tensorflow_support()
    label_encoder = load_label_encoder()

    if label_encoder is None:
        st.error("Error: label_encoder.joblib not found in models/ directory.")
        st.stop()

    classes = list(label_encoder.classes_)

    # Available models list based on TensorFlow presence
    if has_tf:
        model_options = [
            "Multinomial Naive Bayes",
            "GRU",
            "LSTM",
            "SimpleRNN",
        ]
    else:
        model_options = ["Multinomial Naive Bayes"]

    # -------------------------------------------------------------------------
    # Sidebar
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.markdown("## ⚙️ Model Settings")
        chosen_model = st.selectbox(
            "Select Classification Model:",
            options=model_options,
            index=0,
            help="Choose the model used for sentiment inference."
        )

        # Model details chip
        if chosen_model == "Multinomial Naive Bayes":
            acc_val = "80.14%"
            desc = "Fast TF-IDF n-gram probabilistic classifier. Sub-millisecond latency."
            arch = "Classical ML (Scikit-Learn)"
        elif chosen_model == "GRU":
            acc_val = "94.48%"
            desc = "Gated Recurrent Unit neural network with word embeddings."
            arch = "Deep Learning (Keras/TensorFlow)"
        elif chosen_model == "LSTM":
            acc_val = "94.18%"
            desc = "Long Short-Term Memory network capturing long-range sequence context."
            arch = "Deep Learning (Keras/TensorFlow)"
        else:
            acc_val = "93.68%"
            desc = "Simple Recurrent Neural Network for sequential tweet processing."
            arch = "Deep Learning (Keras/TensorFlow)"

        st.info(f"**{chosen_model}**\n\n🎯 **Accuracy:** `{acc_val}`\n\n🏗️ **Type:** {arch}\n\n💡 {desc}")

        if not has_tf:
            st.warning("⚠️ **Note:** Deep learning models (GRU, LSTM, RNN) require TensorFlow. Multinomial Naive Bayes is active and running at peak performance.")

        st.markdown("---")
        st.markdown("### 🏷️ Sentiment Classes")
        for sentiment, theme in SENTIMENT_THEMES.items():
            st.markdown(
                f"<span style='color:{theme['color']}; font-weight:700;'>{theme['icon']} {sentiment}:</span> "
                f"<span style='font-size:0.85rem; color:#94A3B8;'>{theme['desc']}</span>",
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.markdown(
            """
            <div style='font-size:0.8rem; color:#64748B; text-align:center;'>
                PulseTweet AI v2.0 · Ready for Streamlit Cloud<br>
                Trained on 74k+ Twitter Entities
            </div>
            """,
            unsafe_allow_html=True
        )

    # -------------------------------------------------------------------------
    # Hero Section
    # -------------------------------------------------------------------------
    st.markdown(
        """
        <div class="hero-container">
            <h1 class="hero-title">🐦 PulseTweet AI · Twitter Sentiment Analyzer</h1>
            <p class="hero-subtitle">
                Accurately classify tweet sentiments into Positive, Negative, Neutral, and Irrelevant using machine learning & deep neural networks.
            </p>
            <div style="display:flex; gap:8px; flex-wrap:wrap;">
                <span class="hero-badge">⚡ Real-time Inference</span>
                <span class="hero-badge">🧠 4 Trained Models</span>
                <span class="hero-badge">📊 Bulk CSV Analysis</span>
                <span class="hero-badge">🚀 Cloud Ready</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # -------------------------------------------------------------------------
    # Main Tabs
    # -------------------------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "⚡ Single Tweet Analysis",
        "📊 Batch CSV Analysis",
        "🏆 Model Comparison & Benchmark",
        "🚀 Cloud Deployment Guide"
    ])

    # =========================================================================
    # TAB 1: Single Tweet Analysis
    # =========================================================================
    with tab1:
        st.markdown("#### 1. Choose or type a Tweet")

        # Sample quick buttons
        st.markdown("<div style='font-size:0.88rem; color:#94A3B8; margin-bottom:8px;'>Try a quick sample:</div>", unsafe_allow_html=True)
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)

        if "tweet_input" not in st.session_state:
            st.session_state.tweet_input = SAMPLE_TWEETS[0]["text"]

        if col_s1.button("😊 Positive Sample", use_container_width=True):
            st.session_state.tweet_input = SAMPLE_TWEETS[0]["text"]
        if col_s2.button("😡 Negative Sample", use_container_width=True):
            st.session_state.tweet_input = SAMPLE_TWEETS[1]["text"]
        if col_s3.button("⚖️ Neutral Sample", use_container_width=True):
            st.session_state.tweet_input = SAMPLE_TWEETS[2]["text"]
        if col_s4.button("🎯 Irrelevant Sample", use_container_width=True):
            st.session_state.tweet_input = SAMPLE_TWEETS[3]["text"]

        tweet_text = st.text_area(
            "Tweet content:",
            value=st.session_state.tweet_input,
            height=120,
            placeholder="Type or paste any tweet or social media text here...",
            help="Input raw text including URLs, emojis, and mentions. The preprocessor cleans it automatically."
        )

        col_btn1, col_btn2, col_spacer = st.columns([1.5, 1.2, 5])
        with col_btn1:
            analyze_clicked = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
        with col_btn2:
            if st.button("🧹 Clear Text", use_container_width=True):
                st.session_state.tweet_input = ""
                st.rerun()

        # Run Analysis
        if analyze_clicked or (tweet_text and "has_run_init" not in st.session_state):
            st.session_state.has_run_init = True
            if not tweet_text.strip():
                st.warning("⚠️ Please enter or paste some text to analyze.")
            else:
                with st.spinner(f"Analyzing with {chosen_model}..."):
                    try:
                        label, probs, accuracy, latency, cleaned = run_prediction(
                            tweet_text, chosen_model, label_encoder
                        )
                        conf = float(np.max(probs))
                        theme = SENTIMENT_THEMES[label]

                        # Result Card
                        st.markdown(
                            f"""
                            <div class="result-box" style="background:{theme['bg_dark']}; border:2px solid {theme['border']};">
                                <div class="result-header">
                                    <div>
                                        <div style="font-size:0.85rem; text-transform:uppercase; letter-spacing:0.08em; color:{theme['color']}; font-weight:700;">
                                            Predicted Sentiment
                                        </div>
                                        <h2 class="sentiment-title" style="color:{theme['color']};">
                                            {theme['icon']} {label}
                                        </h2>
                                        <div style="color:#CBD5E1; font-size:0.92rem; margin-top:4px;">
                                            {theme['desc']}
                                        </div>
                                    </div>
                                    <div class="conf-badge">
                                        <div class="conf-number" style="color:{theme['color']};">
                                            {conf * 100:.1f}%
                                        </div>
                                        <div class="conf-label">Confidence Score</div>
                                    </div>
                                </div>
                                <div class="meta-chips">
                                    <span class="chip">🤖 <b>Model:</b> {chosen_model}</span>
                                    <span class="chip">🎯 <b>Validation Accuracy:</b> {accuracy * 100:.2f}%</span>
                                    <span class="chip">⚡ <b>Inference Latency:</b> {latency:.2f} ms</span>
                                    <span class="chip">📏 <b>Tokens:</b> {len(cleaned.split())} words</span>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        # Probability Distribution Breakdown
                        st.markdown("#### 📊 Confidence Breakdown Across All Classes")
                        prob_cols = st.columns(len(classes))
                        for i, cls in enumerate(classes):
                            cls_prob = float(probs[i])
                            cls_theme = SENTIMENT_THEMES[cls]
                            with prob_cols[i]:
                                st.markdown(
                                    f"""
                                    <div class="kpi-card" style="border-color:{cls_theme['color'] if cls == label else 'rgba(255,255,255,0.08)'};">
                                        <div style="font-size:1.4rem;">{cls_theme['icon']}</div>
                                        <div class="kpi-value" style="color:{cls_theme['color']}; font-size:1.5rem;">
                                            {cls_prob * 100:.1f}%
                                        </div>
                                        <div class="kpi-label">{cls}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                        # Progress breakdown bar
                        st.markdown(
                            f"""
                            <div class="prob-card">
                                <div style="font-weight:700; font-size:0.92rem; margin-bottom:12px; color:#E2E8F0;">
                                    Class Probability Distribution
                                </div>
                            """,
                            unsafe_allow_html=True
                        )
                        for cls, p in zip(classes, probs):
                            t = SENTIMENT_THEMES[cls]
                            pct = float(p) * 100
                            st.markdown(
                                f"""
                                <div class="prob-item">
                                    <div class="prob-header-row">
                                        <span>{t['icon']} {cls}</span>
                                        <span style="color:{t['color']};">{pct:.2f}%</span>
                                    </div>
                                    <div class="prob-bar-track">
                                        <div class="prob-bar-fill" style="width:{max(pct, 1.5):.2f}%; background:{t['color']};"></div>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        st.markdown("</div>", unsafe_allow_html=True)

                        # Preprocessing Inspection
                        with st.expander("🔍 Inspect Text Preprocessing (What the Model Saw)"):
                            c_raw, c_clean = st.columns(2)
                            with c_raw:
                                st.markdown("**Original Raw Tweet:**")
                                st.code(tweet_text, language="markdown")
                            with c_clean:
                                st.markdown("**Cleaned & Normalized Text:**")
                                st.code(cleaned if cleaned != "empty" else "(no alphanumeric tokens left)", language="markdown")
                            st.caption("Preprocessing pipeline: URLs removed, user handles stripped, non-letter characters filtered, lowercased, and whitespace trimmed.")

                    except Exception as e:
                        st.error(f"Error during inference: {str(e)}")

    # =========================================================================
    # TAB 2: Batch CSV Analysis
    # =========================================================================
    with tab2:
        st.markdown("#### 📂 Bulk Sentiment Classification")
        st.markdown("Analyze multiple tweets at once using uploaded CSV data or multi-line text input.")

        batch_mode = st.radio("Choose Input Method:", ["Paste Multiple Tweets", "Upload CSV File"], horizontal=True)

        batch_texts = []

        if batch_mode == "Paste Multiple Tweets":
            sample_batch = (
                "This new phone battery life is simply fantastic!\n"
                "Flight delayed by 5 hours with zero explanation. Awful airline.\n"
                "The stock market closed 15 points higher on Wednesday.\n"
                "Making coffee and getting ready for the morning commute.\n"
                "Customer support helped me resolve my account issue in 2 minutes!"
            )
            raw_batch = st.text_area(
                "Enter tweets (one tweet per line):",
                value=sample_batch,
                height=150
            )
            if raw_batch.strip():
                batch_texts = [line.strip() for line in raw_batch.strip().split("\n") if line.strip()]

        else:
            uploaded_file = st.file_uploader("Upload CSV containing tweets", type=["csv"])
            if uploaded_file is not None:
                try:
                    df_upload = pd.read_csv(uploaded_file)
                    st.write(f"Uploaded preview ({len(df_upload)} rows):", df_upload.head(3))
                    # Guess or choose column
                    text_cols = df_upload.columns.tolist()
                    chosen_col = st.selectbox("Select the column containing Tweet text:", text_cols)
                    batch_texts = df_upload[chosen_col].dropna().astype(str).tolist()
                except Exception as ex:
                    st.error(f"Error reading CSV: {ex}")

        if st.button("🚀 Process Batch Tweets", type="primary"):
            if not batch_texts:
                st.warning("Please provide at least one tweet to analyze.")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()

                results = []
                total = len(batch_texts)

                for idx, text in enumerate(batch_texts):
                    label, probs, _, latency, cleaned = run_prediction(text, chosen_model, label_encoder)
                    conf = float(np.max(probs))
                    results.append({
                        "Tweet": text,
                        "Predicted Sentiment": label,
                        "Confidence": f"{conf * 100:.1f}%",
                        "Confidence_Score": conf,
                        "Cleaned_Text": cleaned
                    })
                    if idx % max(1, total // 20) == 0 or idx == total - 1:
                        progress_bar.progress((idx + 1) / total)
                        status_text.text(f"Processed {idx + 1} of {total} tweets...")

                status_text.success(f"Successfully processed {total} tweets with {chosen_model}!")
                res_df = pd.DataFrame(results)

                # Metric Cards
                counts = res_df["Predicted Sentiment"].value_counts()
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    pos_c = counts.get("Positive", 0)
                    st.metric("😊 Positive", f"{pos_c} ({pos_c/total*100:.1f}%)")
                with c2:
                    neg_c = counts.get("Negative", 0)
                    st.metric("😡 Negative", f"{neg_c} ({neg_c/total*100:.1f}%)")
                with c3:
                    neu_c = counts.get("Neutral", 0)
                    st.metric("⚖️ Neutral", f"{neu_c} ({neu_c/total*100:.1f}%)")
                with c4:
                    irr_c = counts.get("Irrelevant", 0)
                    st.metric("🎯 Irrelevant", f"{irr_c} ({irr_c/total*100:.1f}%)")

                # Distribution Chart
                st.markdown("##### Sentiment Distribution")
                chart_data = pd.DataFrame({
                    "Sentiment": list(counts.index),
                    "Count": list(counts.values)
                }).set_index("Sentiment")
                st.bar_chart(chart_data)

                # Table & Download
                st.markdown("##### Detailed Predictions")
                st.dataframe(res_df[["Tweet", "Predicted Sentiment", "Confidence"]], use_container_width=True)

                csv_data = res_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv_data,
                    file_name="twitter_sentiment_predictions.csv",
                    mime="text/csv"
                )

    # =========================================================================
    # TAB 3: Model Benchmark & Comparison
    # =========================================================================
    with tab3:
        st.markdown("#### 🏆 Model Performance Benchmark & Architecture Comparison")
        st.markdown("Comparison across all 4 models trained on the Twitter Sentiment dataset:")

        # Benchmark Table
        benchmark_data = {
            "Model": ["GRU", "LSTM", "SimpleRNN", "Multinomial Naive Bayes"],
            "Validation Accuracy": ["94.48%", "94.18%", "93.68%", "80.14%"],
            "Architecture": [
                "Gated Recurrent Unit (Keras)",
                "2-layer LSTM + Dropout (Keras)",
                "SimpleRNN + Dense Softmax (Keras)",
                "TF-IDF Vectorizer (10k n-grams) + MultinomialNB"
            ],
            "Avg Latency": ["~12 ms", "~15 ms", "~8 ms", "< 1 ms"],
            "Best Use Case": [
                "Highest accuracy & sequence comprehension",
                "Deep contextual dependencies",
                "Fast recurrent sequence modeling",
                "Ultra-low latency, CPU-only & lightweight deployments"
            ]
        }
        st.table(pd.DataFrame(benchmark_data))

        # Show generated plots if available
        col_img1, col_img2 = st.columns(2)
        p1 = os.path.join(OUTPUTS_DIR, "accuracy_comparison.png")
        p2 = os.path.join(OUTPUTS_DIR, "dl_accuracy_comparison.png")

        with col_img1:
            if os.path.exists(p1):
                st.image(p1, caption="Classical ML vs SimpleRNN Accuracy", use_container_width=True)
        with col_img2:
            if os.path.exists(p2):
                st.image(p2, caption="Deep Learning Architectures Comparison (RNN vs LSTM vs GRU)", use_container_width=True)

    # =========================================================================
    # TAB 4: Cloud Deployment Guide
    # =========================================================================
    with tab4:
        st.markdown("#### 🚀 Deploy to Streamlit Community Cloud in 1 Minute")
        st.markdown(
            """
            This project is **100% pre-configured and ready to deploy**. Follow these simple steps:

            ### Step 1: Push your code to GitHub
            1. Create a new repository on [GitHub](https://github.com/new) (e.g. `twitter-sentiment-analysis`).
            2. Run these commands in your project folder:
            ```bash
            git init
            git add .
            git commit -m "Deploy Twitter Sentiment Analysis App"
            git branch -M main
            git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
            git push -u origin main
            ```

            ---

            ### Step 2: Connect to Streamlit Community Cloud
            1. Visit **[share.streamlit.io](https://share.streamlit.io/)** and sign in with your GitHub account.
            2. Click **"New app"**.
            3. Fill in the deployment form:
               - **Repository**: `YOUR_USERNAME/YOUR_REPO_NAME`
               - **Branch**: `main`
               - **Main file path**: `app.py`
            4. Click **"Deploy!"** 🚀

            ---

            ### Step 3: Your Live Website URL
            Streamlit Cloud will assign you a live, public URL:
            👉 `https://<your-app-name>.streamlit.app`

            Share this link with anyone! It features instant automatic HTTPS, fast cloud hosting, and automatic redeployment on git push.
            """
        )


if __name__ == "__main__":
    main()
