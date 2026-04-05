import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/raw/diabetes_health_indicators.csv"
TARGET_COL = "Diabetes_binary"

AGE_GROUP_LABELS = {
    1.0: "18-24",
    2.0: "25-29",
    3.0: "30-34",
    4.0: "35-39",
    5.0: "40-44",
    6.0: "45-49",
    7.0: "50-54",
    8.0: "55-59",
    9.0: "60-64",
    10.0: "65-69",
    11.0: "70-74",
    12.0: "75-79",
    13.0: "80+",
}

FEATURE_GROUPS = {
    "Core Health": ["BMI", "GenHlth", "PhysHlth", "MentHlth", "DiffWalk", "Age", "Sex"],
    "Cardio & Medical History": [
        "HighBP",
        "HighChol",
        "CholCheck",
        "Stroke",
        "HeartDiseaseorAttack",
    ],
    "Lifestyle": ["Smoker", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump"],
    "Healthcare & Socioeconomic": ["AnyHealthcare", "NoDocbcCost", "Education", "Income"],
}

INDICATOR_LABELS = {
    "HighBP": "High Blood Pressure",
    "HighChol": "High Cholesterol",
    "CholCheck": "Cholesterol Check",
    "BMI": "Body Mass Index (BMI)",
    "Smoker": "Smoker",
    "Stroke": "Stroke History",
    "HeartDiseaseorAttack": "Heart Disease or Heart Attack",
    "PhysActivity": "Physical Activity",
    "Fruits": "Consumes Fruits Regularly",
    "Veggies": "Consumes Vegetables Regularly",
    "HvyAlcoholConsump": "Heavy Alcohol Consumption",
    "AnyHealthcare": "Has Any Healthcare Coverage",
    "NoDocbcCost": "Could Not See Doctor Due to Cost",
    "GenHlth": "General Health Rating",
    "MentHlth": "Poor Mental Health Days (Last 30)",
    "PhysHlth": "Poor Physical Health Days (Last 30)",
    "DiffWalk": "Difficulty Walking",
    "Sex": "Sex",
    "Age": "Age Group",
    "Education": "Education Level",
    "Income": "Income Level",
}

BINARY_VALUE_LABELS = {
    0.0: "No",
    1.0: "Yes",
}

MOST_RELEVANT_INDICATORS = ["GenHlth", "HighBP", "DiffWalk", "BMI", "HighChol"]
PINNED_REVIEW_INDICATORS = ["BMI", "MentHlth", "PhysHlth"]


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
    display_label = INDICATOR_LABELS.get(col_name, col_name)
    uniques = sorted(series.dropna().unique())

    # In this dataset, Age is an encoded category (1-13), not raw years.
    if col_name == "Age":
        options = [float(v) for v in uniques]
        default = 6.0 if 6.0 in options else options[0]
        return st.selectbox(
            display_label,
            options=options,
            index=options.index(default),
            format_func=lambda v: AGE_GROUP_LABELS.get(v, "Unknown"),
            help="Age is encoded as a group code in this dataset.",
        )

    # Binary and low-cardinality fields become select boxes for easier input.
    if len(uniques) <= 10:
        options = [float(v) for v in uniques]
        default = options[0]
        if set(options).issubset(BINARY_VALUE_LABELS.keys()):
            return st.selectbox(
                display_label,
                options=options,
                index=options.index(default),
                format_func=lambda v: BINARY_VALUE_LABELS.get(v, str(v)),
            )
        return st.selectbox(display_label, options=options, index=options.index(default))

    min_v = float(series.min())
    max_v = float(series.max())
    mean_v = float(series.mean())
    return st.number_input(display_label, min_value=min_v, max_value=max_v, value=mean_v)


def build_feature_groups(feature_cols: list[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    assigned = set()

    for group_name, group_cols in FEATURE_GROUPS.items():
        present_cols = [col for col in group_cols if col in feature_cols]
        if present_cols:
            grouped[group_name] = present_cols
            assigned.update(present_cols)

    remaining_cols = [col for col in feature_cols if col not in assigned]
    if remaining_cols:
        grouped["Other Indicators"] = remaining_cols

    return grouped


def default_value_for_feature(col_name: str, series: pd.Series) -> float:
    uniques = sorted(series.dropna().unique())

    if col_name == "Age":
        options = [float(v) for v in uniques]
        return 6.0 if 6.0 in options else options[0]

    if len(uniques) <= 10:
        return float(uniques[0])

    return float(series.mean())


def main():
    st.set_page_config(page_title="Diabetes Health Indicator", page_icon="🩺", layout="wide")

    # Hide Streamlit deploy controls for local/demo usage.
    st.markdown(
        """
        <style>
        .stDeployButton {
            display: none !important;
        }
        [data-testid="stAppDeployButton"] {
            display: none !important;
        }
        button[aria-label*="Deploy"],
        button[title*="Deploy"],
        a[aria-label*="Deploy"],
        a[href*="share.streamlit.io"] {
            display: none !important;
        }
        /* Force pointer cursor for interactive controls and indicator widgets. */
        button,
        [role="button"],
        a,
        input,
        select,
        textarea,
        [data-baseweb="select"] * {
            cursor: pointer !important;
        }
        div[data-testid="stSelectbox"] *,
        div[data-testid="stNumberInput"] *,
        div[data-testid="stSlider"] *,
        div[data-testid="stCheckbox"] *,
        div[data-testid="stRadio"] *,
        div[data-testid="stMultiSelect"] *,
        div[data-testid="stDateInput"] *,
        div[data-testid="stTimeInput"] *,
        div[data-testid="stTextInput"] *,
        div[data-testid="stTextArea"] *,
        div[data-testid="stToggle"] *,
        [data-testid="stWidgetLabel"] * {
            cursor: pointer !important;
        }
        input[type="text"],
        input[type="number"],
        textarea {
            cursor: text !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

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

    predict_tab, performance_tab = st.tabs(["User Prediction", "Model Performance"])

    with predict_tab:
        st.subheader("Check Your Diabetes Risk")
        st.write("Fill in your health indicators and click Predict to see risk estimates.")
        st.caption("Indicators are grouped below. Expand a section to edit values.")

        form_col, result_col = st.columns([1.2, 1])

        with form_col:
            input_data = {}
            grouped_features = build_feature_groups(feature_cols)
            default_values = {
                col: default_value_for_feature(col, df[col]) for col in feature_cols if col in df.columns
            }

            for group_name, group_cols in grouped_features.items():
                with st.expander(f"{group_name} ({len(group_cols)})", expanded=group_name == "Core Health"):
                    left_col, right_col = st.columns(2)
                    for idx, col in enumerate(group_cols):
                        target_col = left_col if idx % 2 == 0 else right_col
                        with target_col:
                            input_data[col] = input_widget(col, df[col])

            # Keep prediction input exactly aligned with training feature order.
            input_df = pd.DataFrame([input_data]).reindex(columns=feature_cols)

        with result_col:
            st.markdown("### Review Selected Indicators")
            changed_rows = []
            for col in feature_cols:
                current_value = float(input_df.at[0, col])
                default_value = float(default_values[col])
                if abs(current_value - default_value) > 1e-9:
                    changed_rows.append(
                        {
                            "Indicator": INDICATOR_LABELS.get(col, col),
                            "Selected Value": current_value,
                        }
                    )

            changed_lookup = {row["Indicator"]: row["Selected Value"] for row in changed_rows}
            pinned_rows = [
                {
                    "Indicator": INDICATOR_LABELS.get(col, col),
                    "Selected Value": changed_lookup.get(INDICATOR_LABELS.get(col, col), ""),
                }
                for col in PINNED_REVIEW_INDICATORS
                if col in feature_cols
            ]

            other_changed_rows = [
                row for row in changed_rows if row["Indicator"] not in {p["Indicator"] for p in pinned_rows}
            ]

            review_rows = pinned_rows + other_changed_rows

            if review_rows:
                review_df = pd.DataFrame(review_rows)
                st.dataframe(review_df, width="stretch", height=360)
            else:
                st.dataframe(
                    pd.DataFrame(columns=["Indicator", "Selected Value"]),
                    width="stretch",
                    height=180,
                )
                st.caption("No indicators changed yet. Update any field and it will appear here.")
                relevant_labels = [INDICATOR_LABELS.get(col, col) for col in MOST_RELEVANT_INDICATORS]
                st.caption(f"Most relevant indicators: {', '.join(relevant_labels)}")

            predict_clicked = st.button("Predict Diabetes Risk", type="primary", use_container_width=True)
            st.markdown("### Results")
            if predict_clicked:
                lr_prob = float(lr_model.predict_proba(scaler.transform(input_df))[0][1])
                rf_prob = float(rf_model.predict_proba(input_df)[0][1])

                lr_pred = int(lr_prob >= 0.5)
                rf_pred = int(rf_prob >= 0.5)

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
                st.caption("This tool is for educational use and not a medical diagnosis.")
            else:
                st.caption("Prediction results will appear here after you click the button.")

    with performance_tab:
        st.subheader("Model Comparison on Test Set")
        st.dataframe(
            metrics.style.format({"Logistic Regression": "{:.4f}", "Random Forest": "{:.4f}"}),
            width="stretch",
        )

        st.bar_chart(metrics.set_index("Metric"))

        with st.expander("Dataset Snapshot"):
            st.write(f"Rows: **{len(df):,}**")
            st.write(f"Features: **{len(feature_cols)}**")
            class_dist = df[TARGET_COL].value_counts(normalize=True) * 100
            st.write(f"Class 0: **{class_dist.get(0.0, 0):.2f}%**")
            st.write(f"Class 1: **{class_dist.get(1.0, 0):.2f}%**")


if __name__ == "__main__":
    main()
