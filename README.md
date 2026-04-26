# AI-Powered-Fake-News-Detector
# 🕵️ AI-Powered Fake News Detector

A machine learning web app that detects whether a news article is **Real or Fake**
using an ensemble of Logistic Regression + XGBoost.

🔗 **Live Demo:** [Click here](https://YOUR-APP-NAME.streamlit.app)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎯 Score Meter | Visual 0–100% fake confidence gauge |
| 🌐 Live URL Checker | Paste any news link — auto-scrapes & analyzes |
| 📰 Batch Analyzer | Upload a CSV of headlines for bulk predictions |
| 🔦 LIME Explainability | See exactly which words drove the prediction |
| ☁️ Word Cloud | Visual map of top fake vs real signal words |
| 📊 History Chart | Compare predictions across multiple articles |

---

## 🧠 How It Works
News Text → Clean → TF-IDF (unigrams + bigrams) + 11 Custom Features
↓
Voting Ensemble (LR + XGBoost)
↓
Verdict + Confidence Score + Word Explanation

## 🧪 Model Performance
- ✅ Accuracy: ~98%
- ✅ ROC-AUC: ~0.99
- ✅ Cross-validated (5-fold)

---

## 🛠️ Tech Stack
`Python` `Scikit-learn` `XGBoost` `LIME` `Streamlit` `TextBlob` `Newspaper3k`

---

## 🚀 Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/fake-news-detector.git
cd fake-news-detector
pip install -r requirements.txt
streamlit run app.py
```

---

## 📂 Dataset
[Fake and Real News Dataset — Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
