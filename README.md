# Password Strength Classifier

A Streamlit application that uses machine learning to classify passwords into **Weak**, **Moderate**, and **Strong** categories.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-189AB4)](https://xgboost.readthedocs.io/)

## Overview

The project compares three supervised machine-learning models:

- Logistic Regression
- XGBoost
- Multinomial Naive Bayes

Passwords are represented using **character-level TF-IDF**. The training pipeline uses a stratified 80:20 train-test split. The vectorizer is fitted only on the training partition, and class balancing is applied only during model fitting. The held-out test partition is not upsampled or resampled.

The application rejects whitespace characters in entered passwords. Its password generator uses Python's `secrets` module.

## Latest evaluation

These results are from the corrected training run on the whitespace-filtered dataset (November 15, 2026).

| Model | Accuracy | Weighted precision | Weighted recall | Weighted F1 | Macro F1 | Training time |
|---|---:|---:|---:|---:|---:|---:|
| **XGBoost** | **95.81%** | **96.38%** | **95.81%** | **95.93%** | **94.15%** | 9.38 s |
| Logistic Regression | 81.05% | 88.06% | 81.05% | 82.95% | 78.57% | 8.04 s |
| Multinomial Naive Bayes | 60.86% | 78.67% | 60.86% | 65.94% | 61.72% | 0.10 s |

**Important:** These are measured results for this dataset and this particular split. They are not a guarantee that a password is secure, and they should not be interpreted as real-world attack resistance.

### Evaluation dataset

| Item | Count |
|---|---:|
| Original rows | 669,640 |
| Rows removed because password contained whitespace | 851 |
| Rows after cleaning | 668,788 |
| Training rows (80%) | 535,030 |
| Test rows (20%) | 133,758 |

Class labels: `0 = Weak`, `1 = Moderate`, `2 = Strong`.

## How the pipeline works

1. Load the labeled CSV dataset.
2. Remove rows with missing labels, empty passwords, unsupported labels, and passwords containing whitespace.
3. Split the data into stratified training and test partitions (80:20).
4. Fit character-level TF-IDF on the training passwords only.
5. Transform the test passwords using the already-fitted vectorizer.
6. Calculate balanced sample weights from the training labels.
7. Train Logistic Regression with `class_weight="balanced"`, and train XGBoost and Multinomial Naive Bayes with balanced sample weights.
8. Evaluate each model on the same untouched test partition.
9. Save models, vectorizer, metrics, classification reports, confusion matrices, and test data.

## Application features

- Password strength prediction with a selectable model
- Whitespace validation (spaces, tabs, newlines and other whitespace are rejected)
- Password input masking
- Secure random password generation with `secrets`
- Confusion matrices and classification reports on the saved test set
- Model comparison and extended weighted/macro metrics

## Project structure

```text
Password-Strength-Classifier/
├── app.py
├── train_and_save_models.py
├── requirements.txt
├── README.md
├── data/
│   └── data.csv
└── models/
    ├── logistic_regression.pkl
    ├── xgboost.pkl
    ├── naive_bayes.pkl
    ├── tfidf.pkl
    ├── metrics.json
    ├── classification_reports.json
    ├── test_data.pkl
    └── *_confusion_matrix.npy
```

Generated model artifacts are created by the training script and should be kept in sync with the code and metrics. Do not mix model files from an older training run with newly generated metrics or test data.

## Run locally

### 1. Clone the repository

```bash
git clone https://github.com/Mujjjtaba/Password-Strength-Classifier.git
cd Password-Strength-Classifier
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the models

```bash
python train_and_save_models.py
```

The script reads `data/data.csv` (and also supports a root-level `data.csv`), then writes all generated artifacts to `models/`.

### 5. Launch Streamlit

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit (usually `http://localhost:8501`).

## Notes on security and limitations

- The classifier is a machine-learning demonstration, not a password-policy engine or a password-auditing service.
- Passwords entered in the app are processed for local model inference by the Streamlit application. Do not enter passwords that you actually use for important accounts into a demonstration app.
- The generator uses the operating system-backed randomness exposed by Python's `secrets` module.
- This project applies a no-whitespace rule as an explicit application policy. Some real-world services allow spaces in passphrases; password policies vary.
- The three models have different performance characteristics. Naive Bayes is retained for comparison even though its measured results are lower on this dataset.
- Training time depends on hardware, software versions, and system load.

## Author

**Mujtaba Kamal**  
B.Tech CSE, Lovely Professional University

[GitHub](https://github.com/Mujjjtaba) · [LinkedIn](https://www.linkedin.com/in/mujtaba11) · [Live Streamlit app](https://mujjjtaba-password-strength-classification-app-hrvln6.streamlit.app/)
