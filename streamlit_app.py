import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/raw/diabetes_health_indicators.csv"
TARGET_COL = "Diabetes_binary"


@st.cache_resource
def train_models():
    df = pd.read_csv(DATA_PATH)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found.")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    lr_pred = lr_model.predict(X_test_scaled)
    lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

    rf_pred = rf_model.predict(X_test)
    rf_proba = rf_model.predict_proba(X_test)[:, 1]

    metrics = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
            "Logistic Regression": [
                accuracy_score(y_test, lr_pred),
                precision_score(y_test, lr_pred, zero_division=0),
                recall_score(y_test, lr_pred, zero_division=0),
                f1_score(y_test, lr_pred, zero_division=0),
                roc_auc_score(y_test, lr_proba),
            ],
            "Random Forest": [
                accuracy_score(y_test, rf_pred),
                precision_score(y_test, rf_pred, zero_division=0),
                recall_score(y_test, rf_pred, zero_division=0),
                f1_score(y_test, rf_pred, zero_division=0),
                roc_auc_score(y_test, rf_proba),
            ],
        }
    )

    return df, X.columns.tolist(), scaler, lr_model, rf_model, metrics


def input_widget(col_name: str, series: pd.Series):
    uniques = sorted(series.dropna().unique())

    # Binary and low-cardinality fields become select boxes for easier input.
    if len(uniques) <= 10:
        options = [float(v) for v in uniques]
        default = options[0]
        return st.selectbox(col_name, options=options, index=options.index(default))

    min_v = float(series.min())
    max_v = float(series.max())
    mean_v = float(series.mean())
    return st.number_input(col_name, min_value=min_v, max_value=max_v, value=mean_v)


def main():
    st.set_page_config(page_title="Diabetes Health Indicator", page_icon="🩺", layout="wide")

    st.title("Diabetes Health Indicator Predictor")
    st.caption("Supervised classification using Logistic Regression and Random Forest")

    try:
        df, feature_cols, scaler, lr_model, rf_model, metrics = train_models()
    except FileNotFoundError:
        st.error("Dataset not found. Place diabetes_health_indicators.csv in data/raw/.")
        st.stop()
    except Exception as exc:
        st.error(f"Failed to train models: {exc}")
        st.stop()

    left, right = st.columns([1.1, 1])

    with left:
        st.subheader("Enter Health Indicators")
        input_data = {}
        for col in feature_cols:
            input_data[col] = input_widget(col, df[col])

        input_df = pd.DataFrame([input_data])

        if st.button("Predict Diabetes Risk", type="primary"):
            lr_prob = float(lr_model.predict_proba(scaler.transform(input_df))[0][1])
            rf_prob = float(rf_model.predict_proba(input_df)[0][1])

            lr_pred = int(lr_prob >= 0.5)
            rf_pred = int(rf_prob >= 0.5)

            st.markdown("### Prediction Results")
            p1, p2 = st.columns(2)

            with p1:
                st.metric("Logistic Regression", "Diabetes" if lr_pred == 1 else "No Diabetes")
                st.progress(lr_prob)
                st.write(f"Probability: **{lr_prob:.2%}**")

            with p2:
                st.metric("Random Forest", "Diabetes" if rf_pred == 1 else "No Diabetes")
                st.progress(rf_prob)
                st.write(f"Probability: **{rf_prob:.2%}**")

            avg_prob = (lr_prob + rf_prob) / 2
            st.info(f"Average model risk score: **{avg_prob:.2%}**")

    with right:
        st.subheader("Model Comparison on Test Set")
        st.dataframe(
            metrics.style.format({"Logistic Regression": "{:.4f}", "Random Forest": "{:.4f}"}),
            width="stretch",
        )

        plot_df = metrics.set_index("Metric")
        st.bar_chart(plot_df)

        st.markdown("### Dataset Snapshot")
        st.write(f"Rows: **{len(df):,}**")
        st.write(f"Features: **{len(feature_cols)}**")
        class_dist = df[TARGET_COL].value_counts(normalize=True) * 100
        st.write(f"Class 0: **{class_dist.get(0.0, 0):.2f}%**")
        st.write(f"Class 1: **{class_dist.get(1.0, 0):.2f}%**")


if __name__ == "__main__":
    main()
