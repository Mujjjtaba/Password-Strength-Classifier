import json
import secrets
import string
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
CLASS_NAMES = ["Weak", "Moderate", "Strong"]
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "XGBoost": "xgboost.pkl",
    "Naive Bayes": "naive_bayes.pkl",
}

st.set_page_config(
    page_title="PasswordIQ | Password Strength Analyzer",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)


def char_tokenizer(text):
    """Must match the tokenizer used to train the saved TF-IDF vectorizer."""
    return list(text)


@st.cache_resource
def load_models():
    missing = [
        str(MODEL_DIR / filename)
        for filename in [*MODEL_FILES.values(), "tfidf.pkl"]
        if not (MODEL_DIR / filename).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Required trained model artifacts are missing. Run "
            "`python train_and_save_models.py` and add the generated files "
            "to the repository. Missing: " + ", ".join(missing)
        )

    loaded = {
        name: joblib.load(MODEL_DIR / filename)
        for name, filename in MODEL_FILES.items()
    }
    vectorizer = joblib.load(MODEL_DIR / "tfidf.pkl")
    return loaded, vectorizer


@st.cache_data
def load_metrics():
    path = MODEL_DIR / "metrics.json"
    if not path.exists():
        raise FileNotFoundError("models/metrics.json is missing.")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_test_data():
    path = MODEL_DIR / "test_data.pkl"
    if not path.exists():
        raise FileNotFoundError(
            "models/test_data.pkl is missing. Run train_and_save_models.py "
            "and upload the generated test_data.pkl artifact."
        )
    data = joblib.load(path)
    return data["X_test"], data["y_test"]


def validate_password(password):
    if not password:
        return False, "Please enter a password."
    if any(character.isspace() for character in password):
        return False, "Whitespace is not allowed. Remove spaces, tabs or line breaks."
    return True, ""


def predict_password_strength(password, model, vectorizer):
    features = vectorizer.transform([password])
    prediction = int(model.predict(features)[0])
    return CLASS_NAMES[prediction]


def generate_strong_password(length):
    alphabet = string.ascii_lowercase + string.ascii_uppercase + string.digits + string.punctuation
    while True:
        password = "".join(secrets.choice(alphabet) for _ in range(length))
        if (
            any(char.islower() for char in password)
            and any(char.isupper() for char in password)
            and any(char.isdigit() for char in password)
            and any(char in string.punctuation for char in password)
            and not any(char.isspace() for char in password)
        ):
            return password


def show_evaluation(models, vectorizer):
    st.title("Model Evaluation")
    st.caption("Confusion matrices and classification reports on the saved, untouched test partition.")

    try:
        X_test, y_test = load_test_data()
    except (FileNotFoundError, KeyError) as error:
        st.error(str(error))
        return

    X_test_features = vectorizer.transform(X_test)

    for name, model in models.items():
        with st.expander(f"{name} — confusion matrix", expanded=(name == "XGBoost")):
            predictions = model.predict(X_test_features)
            matrix = confusion_matrix(y_test, predictions, labels=[0, 1, 2])

            fig, ax = plt.subplots(figsize=(7, 5))
            sns.heatmap(
                matrix,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES,
                ax=ax,
            )
            ax.set_xlabel("Predicted class")
            ax.set_ylabel("Actual class")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            report = classification_report(
                y_test,
                predictions,
                labels=[0, 1, 2],
                target_names=CLASS_NAMES,
                output_dict=True,
                zero_division=0,
            )
            st.dataframe(
                pd.DataFrame(report).transpose(),
                use_container_width=True,
            )


def show_comparison():
    st.title("Model Comparison")
    try:
        metrics = load_metrics()
    except (FileNotFoundError, json.JSONDecodeError) as error:
        st.error(str(error))
        return

    frame = pd.DataFrame(metrics).T.rename_axis("Model").reset_index()
    display_columns = [
        "Model",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "macro_f1_score",
        "training_time_sec",
    ]
    frame = frame[[column for column in display_columns if column in frame.columns]]
    st.dataframe(frame, use_container_width=True, hide_index=True)

    best = max(metrics, key=lambda model: metrics[model]["accuracy"])
    left, middle, right = st.columns(3)
    left.metric("Top accuracy model", best.replace("_", " ").title())
    middle.metric("Best accuracy", f"{metrics[best]['accuracy']:.2f}%")
    right.metric("Best macro F1", f"{metrics[best]['macro_f1_score']:.2f}%")

    chart_data = frame.rename(
        columns={
            "accuracy": "Accuracy (%)",
            "training_time_sec": "Training time (sec)",
            "Model": "Model",
        }
    )
    st.subheader("Accuracy")
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(data=chart_data, x="Model", y="Accuracy (%)", ax=ax)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Training time")
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(data=chart_data, x="Model", y="Training time (sec)", ax=ax)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def show_extended_metrics():
    st.title("Extended Metrics")
    try:
        metrics = load_metrics()
    except (FileNotFoundError, json.JSONDecodeError) as error:
        st.error(str(error))
        return

    available = [
        ("precision", "Weighted precision"),
        ("recall", "Weighted recall"),
        ("f1_score", "Weighted F1"),
        ("macro_precision", "Macro precision"),
        ("macro_recall", "Macro recall"),
        ("macro_f1_score", "Macro F1"),
    ]
    for key, label in available:
        if not all(key in values for values in metrics.values()):
            continue
        st.subheader(label)
        chart = pd.DataFrame(
            {
                "Model": [name.replace("_", " ").title() for name in metrics],
                "Score (%)": [values[key] for values in metrics.values()],
            }
        )
        fig, ax = plt.subplots(figsize=(8, 3.5))
        sns.barplot(data=chart, x="Model", y="Score (%)", ax=ax)
        ax.set_ylim(0, 100)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


def main():
    try:
        models, vectorizer = load_models()
    except (FileNotFoundError, ModuleNotFoundError, AttributeError, ValueError) as error:
        st.error(f"Could not load model artifacts: {error}")
        st.info("Train the models with train_and_save_models.py and upload the resulting artifacts.")
        st.stop()

    with st.sidebar:
        st.title("🔐 PasswordIQ")
        st.caption("Password strength analysis")
        page = st.radio(
            "NAVIGATION",
            [
                "Password Analyzer",
                "Password Generator",
                "Model Evaluation",
                "Model Comparison",
                "Extended Metrics",
                "About",
            ],
        )
        st.divider()
        st.caption("Models: Logistic Regression · XGBoost · Naive Bayes")

    if page == "Password Analyzer":
        st.title("Password Strength Analyzer")
        st.caption("Classify a password using one of the trained machine-learning models.")

        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter a password",
            help="Whitespace characters such as spaces, tabs and line breaks are not accepted.",
        )
        selected_model = st.selectbox("Machine-learning model", list(models))

        if st.button("Analyze password", type="primary", use_container_width=True):
            valid, message = validate_password(password)
            if not valid:
                st.error(message)
            else:
                result = predict_password_strength(
                    password,
                    models[selected_model],
                    vectorizer,
                )
                if result == "Weak":
                    st.error("Prediction: Weak")
                elif result == "Moderate":
                    st.warning("Prediction: Moderate")
                else:
                    st.success("Prediction: Strong")
                st.caption(
                    "This is a dataset-based ML prediction, not a guarantee of resistance "
                    "to guessing, phishing, credential stuffing or data breaches."
                )

    elif page == "Password Generator":
        st.title("Secure Password Generator")
        st.caption("Generates passwords using Python's cryptographically secure secrets module.")
        length = st.slider("Password length", min_value=8, max_value=32, value=16)
        if st.button("Generate password", type="primary", use_container_width=True):
            st.code(generate_strong_password(length), language=None)
            st.success("Generated. Store it in a reputable password manager.")

    elif page == "Model Evaluation":
        show_evaluation(models, vectorizer)

    elif page == "Model Comparison":
        show_comparison()

    elif page == "Extended Metrics":
        show_extended_metrics()

    elif page == "About":
        st.title("About PasswordIQ")
        st.write(
            "PasswordIQ is a machine-learning project that classifies passwords "
            "into Weak, Moderate and Strong labels using character-level TF-IDF."
        )
        st.markdown(
            """
            **Models**
            - Logistic Regression
            - XGBoost
            - Multinomial Naive Bayes

            **Methodology**
            - Stratified 80:20 train-test split
            - TF-IDF fitted on training data only
            - Class balancing applied during training only
            - Evaluation on the same untouched test set for all models

            **Input policy:** whitespace is rejected by this application. This is a
            project policy; some real-world services support spaces in passphrases.
            """
        )

    st.caption("PasswordIQ · Machine-learning demonstration")


if __name__ == "__main__":
    main()
