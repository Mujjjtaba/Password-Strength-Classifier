# 🔐 Password Strength Classifier ML App

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-98.85%25_Accuracy-189AB4?style=for-the-badge)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A full-stack ML-powered web application that classifies any password as Weak, Moderate, or Strong — trained on a balanced dataset using character-level TF-IDF vectorization and 3 competing models.**

[🌐 Live App](https://mujjjtaba-password-strength-classification-app-hrvln6.streamlit.app/) · [📄 Project Report](https://drive.google.com/drive/folders/1FVaqAgvUkd-wUiAzFODG6szq7dO685OA?usp=sharing) · [🔗 LinkedIn](https://www.linkedin.com/in/mujtaba11)

</div>

---

## ✨ What This App Does

Enter any password → select a model → get an instant **Weak / Moderate / Strong** classification with full model performance metrics, confusion matrices, and comparative visualizations — all inside a clean Streamlit interface.

> Built as part of the **Summer Internship / Training Project** at Lovely Professional University  
> Course: *Machine Learning Made Easy: From Basics to AI Applications*  
> Student: **Mujtaba Kamal** · Reg. No: 12319576 · June–July 2025

---

## 🚀 Live Demo

🌐 **[Try the App →](https://mujjjtaba-password-strength-classification-app-hrvln6.streamlit.app/)**

App features available live:
- 🔍 Classify any password with your choice of model
- 🔐 Generate cryptographically strong random passwords
- 📊 View confusion matrices for all 3 models
- 📈 Compare training time, accuracy, precision, recall & F1-score
- 💡 Read project insights and model selection rationale

---

## 🏆 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|----------|-----------|--------|----------|---------------|
| 🥇 **XGBoost** | **98.85%** | **98.86%** | **98.85%** | **98.85%** | 11.35s |
| 🥈 **Logistic Regression** | 80.34% | 80.17% | 80.34% | 80.20% | 14.72s |
| 🥉 **Naive Bayes** | 58.87% | 58.42% | 58.87% | 58.47% | 0.17s |

> **XGBoost dominates** — nearly 99% accuracy with strong generalization across all three password strength classes.

---

## 🧠 How It Works

```
User Input Password
        │
        ▼
Character-Level Tokenizer
        │
        ▼
TF-IDF Vectorizer (char n-grams)
        │
        ├──▶ Logistic Regression ──▶ Weak / Moderate / Strong
        ├──▶ XGBoost Classifier  ──▶ Weak / Moderate / Strong
        └──▶ Naive Bayes         ──▶ Weak / Moderate / Strong
```

**Why character-level TF-IDF?** Passwords don't have "words" — they have character patterns. This approach captures entropy, repetition, special character usage, and structural complexity at the character level, which directly determines password strength.

---

## 📊 App Sections

| Section | Description |
|---------|-------------|
| 🔍 **Classify Password** | Enter any password, select a model, get instant strength prediction |
| 🔐 **Generate Password** | Slider-based strong password generator (8–24 chars) |
| 📊 **Model Evaluation** | Live confusion matrices for all 3 models on test data |
| 📈 **Model Comparison** | Bar charts: accuracy & training time side-by-side |
| 📉 **Extended Metrics** | Precision, Recall, F1-Score visualized per model |
| 💡 **Insights** | Model selection rationale and key project takeaways |

---

## 🗂️ Project Structure

```
Password-Strength-Classifier/
│
├── 📄 app.py                        # Streamlit app — UI, prediction, visualization
├── 📄 train_and_save_models.py      # Full training pipeline
├── 📄 requirements.txt              # Python dependencies
├── 📄 SummerInternship.pdf          # Full academic project report
│
├── 📁 models/
│   ├── logistic_regression.pkl      # Trained Logistic Regression model
│   ├── xgboost.pkl                  # Trained XGBoost model (98.85% accuracy)
│   ├── naive_bayes.pkl              # Trained Naive Bayes model
│   ├── tfidf.pkl                    # Fitted TF-IDF vectorizer
│   └── metrics.json                 # All model metrics (accuracy, F1, time, etc.)
│
└── 📁 data/
    └── data.csv                     # Labeled password dataset (Weak/Moderate/Strong)
```

---

## ⚙️ Getting Started (Run Locally)

### 1. Clone the repo
```bash
git clone https://github.com/Mujjjtaba/Password-Strength-Classifier.git
cd Password-Strength-Classifier
```

### 2. Create virtual environment & install dependencies
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

### 3. (Optional) Retrain models
```bash
python train_and_save_models.py
```

### 4. Launch the app
```bash
streamlit run app.py
```

Then open → **`http://localhost:8501`**

---

## 📦 Dependencies

```txt
streamlit
pandas
scikit-learn
xgboost
seaborn
matplotlib
joblib
```

---

## 🔬 ML Pipeline Details

| Step | Details |
|------|---------|
| **Dataset** | CSV with passwords labeled 0 (Weak), 1 (Moderate), 2 (Strong) |
| **Balancing** | Upsampling minority classes to match majority class size |
| **Tokenization** | Character-level (not word-level) — captures password structure |
| **Vectorization** | TF-IDF on character tokens — weights rare/unique patterns higher |
| **Split** | 80% train / 20% test, stratified |
| **Serialization** | `joblib` for models + vectorizer; `metrics.json` for results |
| **Deployment** | Streamlit Cloud with `@st.cache_data` for performance |

---

## 🔮 Future Scope

- [ ] Deep learning (LSTM/RNN) for sequence-aware password modeling
- [ ] Real-time feedback: "Add a special character to strengthen this password"
- [ ] Leaked password database integration (HaveIBeenPwned API)
- [ ] SHAP/LIME explainability for transparent predictions
- [ ] Chrome extension / mobile SDK for signup form integration
- [ ] Multilingual password support

---

## 📚 References

1. [Password Strength Detection via ML — arXiv](https://arxiv.org/)
2. [ML-Based Password Strength Estimation — MDPI](https://www.mdpi.com/)
3. [A Password Strength Classifier Using Supervised ML — ResearchGate](https://www.researchgate.net/)
4. [Upsampling & Downsampling Imbalanced Data — GeeksForGeeks](https://www.geeksforgeeks.org/)
5. [Scikit-learn Documentation](https://scikit-learn.org/)
6. [XGBoost Documentation](https://xgboost.readthedocs.io/)

---

## 👤 Author

**Mujtaba Kamal**  
B.Tech CSE · Lovely Professional University · Reg. No: 12319576  
Summer Internship — Centre for Professional Enhancement · Certificate ID: 407935

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat&logo=linkedin)](https://www.linkedin.com/in/mujtaba11)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat&logo=github)](https://github.com/Mujjjtaba)
[![Live App](https://img.shields.io/badge/Streamlit-Live_App-FF4B4B?style=flat&logo=streamlit)](https://mujjjtaba-password-strength-classification-app-hrvln6.streamlit.app/)

---

## 📄 License

Licensed under the **MIT License** — free to use, modify, and distribute with attribution.

---

<div align="center">
  <sub>Built with Python, Streamlit & ❤️ · Lovely Professional University · Summer 2025</sub>
</div>
