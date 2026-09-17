# 🐦 PulseTweet AI · Twitter Sentiment Analysis

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red.svg)](https://keras.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A state-of-the-art **Twitter Sentiment Analysis Web Application** built with Streamlit, Scikit-Learn, and Deep Learning (GRU, LSTM, SimpleRNN). Pre-configured and fully production-ready for deployment on **Streamlit Community Cloud**.

---

## 🌟 Key Features

- **⚡ Real-Time Single Tweet Classification**: Instant sentiment prediction with confidence score, class probability breakdown, and regex text normalization preview.
- **🏷️ 4-Class Sentiment Taxonomy**:
  - 😊 **Positive**: Excitement, appreciation, optimism, praise.
  - 😡 **Negative**: Criticism, anger, disappointment, frustration.
  - ⚖️ **Neutral**: Factual statements, news, product updates.
  - 🎯 **Irrelevant**: Conversational noise, off-topic chat.
- **🧠 Multi-Model AI Engine**:
  - **Multinomial Naive Bayes** (TF-IDF 10k n-grams) · Ultra-fast `< 1ms` inference, 80.14% accuracy.
  - **GRU (Gated Recurrent Unit)** · Sequence-aware deep neural network, **94.48% accuracy**.
  - **LSTM (Long Short-Term Memory)** · 2-layer recurrent network with dropout, **94.18% accuracy**.
  - **SimpleRNN** · Classic recurrent neural model, **93.68% accuracy**.
  - *Resilient design*: Automatically switches to lightweight Naive Bayes if TensorFlow is unavailable on resource-constrained platforms.
- **📂 Bulk CSV & Multi-line Tweet Analysis**: Upload a CSV or paste multiple tweets to classify in batch, with sentiment distribution metrics and 1-click CSV export.
- **📊 Interactive Visualizations**: Probability distributions, sentiment metric cards, and validation comparison charts.
- **🎨 Modern Twitter UI**: Dark-mode optimized, responsive layout with custom badges and feedback.

---

## 🏆 Model Performance Benchmark

All models were evaluated on the Twitter Sentiment dataset:

| Model | Architecture | Validation Accuracy | Inference Latency | Primary Strength |
| :--- | :--- | :--- | :--- | :--- |
| **GRU** | Gated Recurrent Unit | **94.48%** | ~12 ms | Highest accuracy & complex context |
| **LSTM** | 2-Layer LSTM + Dropout | **94.18%** | ~15 ms | Strong long-term sequence memory |
| **SimpleRNN** | Simple Recurrent Layer | **93.68%** | ~8 ms | Fast recurrent sequence modeling |
| **Multinomial NB** | TF-IDF (10,000 features) | **80.14%** | **< 1 ms** | Ultra-lightweight, zero memory overhead |

---

## 🚀 Instant Deployment to Streamlit Cloud (1 Minute)

This repository includes all required configuration files (`.streamlit/config.toml`, `requirements.txt`, and trained model weights) so you can deploy immediately.

### Step 1: Push to your GitHub
```bash
git init
git add .
git commit -m "Deploy PulseTweet AI"
git branch -M main
git remote add origin https://github.com/<YOUR_USERNAME>/<YOUR_REPOSITORY_NAME>.git
git push -u origin main
```

### Step 2: Deploy on Streamlit Community Cloud
1. Go to **[share.streamlit.io](https://share.streamlit.io/)** and log in with GitHub.
2. Click **"New app"** (top right).
3. Select:
   - **Repository:** `<YOUR_USERNAME>/<YOUR_REPOSITORY_NAME>`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **Deploy!** 🚀

Your app will be live with a permanent URL:  
👉 `https://<your-app-name>.streamlit.app`

---

## 💻 Running Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPOSITORY_NAME>.git
   cd <YOUR_REPOSITORY_NAME>
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the web application:**
   ```bash
   streamlit run app.py
   ```
   Open your browser at `http://localhost:8501`.

---

## 📁 Repository Structure

```
├── .streamlit/
│   └── config.toml               # Streamlit theme & server configuration
├── models/                       # Trained model binaries & metadata
│   ├── naive_bayes_model.joblib  # Trained Multinomial NB model
│   ├── tfidf_vectorizer.joblib   # Fitted TF-IDF Vectorizer
│   ├── label_encoder.joblib      # Sentiment class encoder
│   ├── rnn_model.keras           # Trained SimpleRNN weights
│   ├── rnn_tokenizer.joblib      # RNN text tokenizer
│   ├── lstm_model.keras          # Trained LSTM weights
│   ├── lstm_tokenizer.joblib     # LSTM text tokenizer
│   ├── gru_model.keras           # Trained GRU weights
│   ├── gru_tokenizer.joblib      # GRU text tokenizer
│   └── *_accuracy.json           # Model validation metrics
├── outputs/                      # Model comparison charts
│   ├── accuracy_comparison.png
│   └── dl_accuracy_comparison.png
├── app.py                        # Streamlit web application
├── preprocess.py                 # Text cleaning & regex normalization
├── train_naive_bayes.py          # Naive Bayes training script
├── train_rnn.py                  # SimpleRNN training script
├── train_lstm.py                 # LSTM training script
├── train_gru.py                  # GRU training script
├── compare_models.py             # Classical vs RNN benchmark script
├── compare_dl_models.py          # Deep learning benchmark script
├── requirements.txt              # Cloud-optimized dependencies
├── .gitignore                    # Git tracking rules
└── README.md                     # Documentation
```

---

## 🧪 Sample Tweets to Test

| Sentiment | Example Tweet |
| :--- | :--- |
| **😊 Positive** | *"I absolutely love this new update! Everything runs so smoothly and looks amazing."* |
| **😡 Negative** | *"Worst customer service ever! Waited over an hour and they just hung up on me."* |
| **⚖️ Neutral** | *"Microsoft announced their quarterly financial earnings report this Tuesday morning."* |
| **🎯 Irrelevant** | *"Just got a cup of iced coffee and a muffin from the bakery down the street."* |
