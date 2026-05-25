import json
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from config import (
    MODEL_PATH,
    PREPROCESSED_DATA_PATH,
    FEATURE_NAMES_PATH,
    RAW_FEATURE_COLUMNS_PATH,
    MODEL_REPORT_DIR,
)

from predict import (
    load_model,
    load_preprocessor,
    load_selected_threshold,
    load_raw_feature_columns,
    prepare_input_for_prediction,
    assign_risk_level,
    create_recommendation,
)


# =========================================================
# Artifact loading
# =========================================================

def load_feature_names() -> List[str]:
    if not FEATURE_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"Feature names file not found: {FEATURE_NAMES_PATH}. "
            "Run preprocessing.py first."
        )

    with open(FEATURE_NAMES_PATH, "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    return feature_names


def load_preprocessed_data():
    if not PREPROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessed data not found: {PREPROCESSED_DATA_PATH}. "
            "Run preprocessing.py first."
        )

    return joblib.load(PREPROCESSED_DATA_PATH)


def build_shap_explainer(model):
    """
    Build SHAP TreeExplainer for XGBoost model.
    """

    return shap.TreeExplainer(model)


# =========================================================
# SHAP compatibility helpers
# =========================================================

def extract_shap_values(shap_values):
    """
    Handle different SHAP output formats.

    For binary classification, SHAP can return:
    - numpy array with shape (n_samples, n_features)
    - list of arrays
    - 3D array in some versions
    """

    if isinstance(shap_values, list):
        if len(shap_values) == 2:
            return shap_values[1]
        return shap_values[0]

    shap_values = np.array(shap_values)

    if shap_values.ndim == 3:
        return shap_values[:, :, 1]

    return shap_values


def align_feature_names(feature_names: List[str], n_features: int) -> List[str]:
    """
    Make sure feature names match processed feature count.
    """

    if len(feature_names) == n_features:
        return feature_names

    return [f"feature_{i}" for i in range(n_features)]


def infer_raw_feature_name(
    processed_feature_name: str,
    raw_feature_columns: List[str]
) -> Dict[str, Optional[str]]:
    """
    Try to map processed one-hot feature name back to original raw feature.

    Example:
    BasePolicy_All Perils -> raw_column=BasePolicy, category_value=All Perils
    """

    sorted_columns = sorted(raw_feature_columns, key=len, reverse=True)

    for col in sorted_columns:
        if processed_feature_name == col:
            return {
                "raw_column": col,
                "category_value": None
            }

        prefix = f"{col}_"

        if processed_feature_name.startswith(prefix):
            category_value = processed_feature_name.replace(prefix, "", 1)

            return {
                "raw_column": col,
                "category_value": category_value
            }

    return {
        "raw_column": None,
        "category_value": None
    }


def build_reason_text(
    processed_feature_name: str,
    raw_column: Optional[str],
    category_value: Optional[str],
    shap_value: float
) -> str:
    """
    Create simple business-friendly explanation text.
    """

    direction = "increased" if shap_value > 0 else "decreased"

    if raw_column and category_value:
        return (
            f"{raw_column} being '{category_value}' {direction} the fraud risk."
        )

    if raw_column:
        return f"{raw_column} {direction} the fraud risk."

    return f"{processed_feature_name} {direction} the fraud risk."


# =========================================================
# Global SHAP analysis
# =========================================================

def create_global_shap_summary(
    sample_size: int = 1000
) -> Dict[str, Any]:
    """
    Create global SHAP summary plot and feature importance table.
    """

    MODEL_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    model = load_model()
    data = load_preprocessed_data()
    feature_names = load_feature_names()

    X_train = data["X_train_processed"]

    n_rows = X_train.shape[0]
    n_features = X_train.shape[1]

    feature_names = align_feature_names(feature_names, n_features)

    if n_rows > sample_size:
        rng = np.random.default_rng(42)
        sample_indices = rng.choice(n_rows, size=sample_size, replace=False)
        X_sample = X_train[sample_indices]
    else:
        X_sample = X_train

    explainer = build_shap_explainer(model)
    shap_values_raw = explainer.shap_values(X_sample)
    shap_values = extract_shap_values(shap_values_raw)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False)

    importance_output_path = MODEL_REPORT_DIR / "shap_feature_importance.csv"
    importance_df.to_csv(importance_output_path, index=False, encoding="utf-8-sig")

    shap_summary_path = MODEL_REPORT_DIR / "shap_summary.png"

    plt.figure()
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        show=False,
        max_display=25
    )
    plt.tight_layout()
    plt.savefig(shap_summary_path, dpi=140, bbox_inches="tight")
    plt.close()

    bar_plot_path = MODEL_REPORT_DIR / "shap_feature_importance_bar.png"

    top_importance = importance_df.head(25).sort_values("mean_abs_shap")

    plt.figure(figsize=(9, 8))
    plt.barh(top_importance["feature"], top_importance["mean_abs_shap"])
    plt.title("Top 25 SHAP Feature Importances")
    plt.xlabel("Mean absolute SHAP value")
    plt.tight_layout()
    plt.savefig(bar_plot_path, dpi=140, bbox_inches="tight")
    plt.close()

    create_global_shap_html_report(importance_df)

    return {
        "sample_size_used": int(X_sample.shape[0]),
        "feature_count": int(n_features),
        "shap_summary_path": str(shap_summary_path),
        "shap_feature_importance_path": str(importance_output_path),
        "shap_bar_plot_path": str(bar_plot_path),
        "top_features": importance_df.head(20).to_dict(orient="records")
    }


def create_global_shap_html_report(importance_df: pd.DataFrame):
    """
    Create simple HTML report for global SHAP analysis.
    """

    top_table = importance_df.head(50).to_html(
        index=False,
        classes="styled-table"
    )

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Vehicle Insurance Fraud Detection - SHAP Report</title>

        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                background-color: #f5f6f8;
                color: #222;
            }}

            .container {{
                max-width: 1200px;
                margin: auto;
                padding: 36px;
            }}

            h1, h2, h3 {{
                color: #111827;
            }}

            .section {{
                background: white;
                padding: 24px;
                border-radius: 14px;
                margin-bottom: 28px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
            }}

            img {{
                max-width: 100%;
                height: auto;
                display: block;
                margin: auto;
            }}

            .table-wrapper {{
                overflow-x: auto;
            }}

            .styled-table {{
                border-collapse: collapse;
                width: 100%;
                font-size: 13px;
                margin-top: 12px;
            }}

            .styled-table th, .styled-table td {{
                border: 1px solid #ddd;
                padding: 9px;
                text-align: left;
            }}

            .styled-table th {{
                background-color: #f3f4f6;
                font-weight: bold;
            }}

            .note {{
                background-color: #fff7ed;
                border-left: 5px solid #f97316;
                padding: 14px;
                margin-top: 16px;
                border-radius: 8px;
            }}

            code {{
                background: #f3f4f6;
                padding: 2px 6px;
                border-radius: 5px;
            }}
        </style>
    </head>

    <body>
        <div class="container">

            <h1>Vehicle Insurance Claim Fraud Detection - SHAP Explainability Report</h1>

            <div class="section">
                <h2>1. What is this report?</h2>
                <p>
                    This report explains the global behavior of the XGBoost fraud detection model using SHAP.
                    SHAP values show how much each feature contributes to increasing or decreasing the model output.
                </p>

                <div class="note">
                    SHAP explanations are especially useful in fraud detection because business users need to understand
                    why a claim is considered risky, not only whether it is classified as fraud.
                </div>
            </div>

            <div class="section">
                <h2>2. SHAP Summary Plot</h2>
                <img src="shap_summary.png" />
            </div>

            <div class="section">
                <h2>3. Top SHAP Feature Importance</h2>
                <img src="shap_feature_importance_bar.png" />
            </div>

            <div class="section">
                <h2>4. Feature Importance Table</h2>
                <div class="table-wrapper">
                    {top_table}
                </div>
            </div>

        </div>
    </body>
    </html>
    """

    output_path = MODEL_REPORT_DIR / "shap_report.html"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return output_path


# =========================================================
# Local SHAP explanation
# =========================================================

def explain_single_claim(
    input_data: Dict[str, Any],
    model=None,
    preprocessor=None,
    threshold: Optional[float] = None,
    raw_feature_columns: Optional[List[str]] = None,
    feature_names: Optional[List[str]] = None,
    explainer=None,
    top_n: int = 10
) -> Dict[str, Any]:
    """
    Explain a single claim prediction using SHAP.
    """

    if model is None:
        model = load_model()

    if preprocessor is None:
        preprocessor = load_preprocessor()

    if threshold is None:
        threshold = load_selected_threshold()

    if raw_feature_columns is None:
        raw_feature_columns = load_raw_feature_columns()

    if feature_names is None:
        feature_names = load_feature_names()

    if explainer is None:
        explainer = build_shap_explainer(model)

    X = prepare_input_for_prediction(
        input_data=input_data,
        raw_feature_columns=raw_feature_columns
    )

    X_processed = preprocessor.transform(X)

    n_features = X_processed.shape[1]
    feature_names = align_feature_names(feature_names, n_features)

    fraud_probability = float(model.predict_proba(X_processed)[:, 1][0])
    prediction = int(fraud_probability >= threshold)

    risk_level = assign_risk_level(fraud_probability)

    recommendation = create_recommendation(
        fraud_probability=fraud_probability,
        prediction=prediction,
        risk_level=risk_level
    )

    shap_values_raw = explainer.shap_values(X_processed)
    shap_values = extract_shap_values(shap_values_raw)

    row_shap_values = shap_values[0]
    row_feature_values = X_processed[0]

    explanation_rows = []

    for idx, shap_value in enumerate(row_shap_values):
        processed_feature_name = feature_names[idx]

        mapped_feature = infer_raw_feature_name(
            processed_feature_name=processed_feature_name,
            raw_feature_columns=raw_feature_columns
        )

        raw_column = mapped_feature["raw_column"]
        category_value = mapped_feature["category_value"]

        direction = "increases_fraud_risk" if shap_value > 0 else "decreases_fraud_risk"

        explanation_rows.append({
            "processed_feature": processed_feature_name,
            "raw_column": raw_column,
            "category_value": category_value,
            "feature_value": round(float(row_feature_values[idx]), 4),
            "shap_value": round(float(shap_value), 6),
            "absolute_shap_value": round(float(abs(shap_value)), 6),
            "direction": direction,
            "reason": build_reason_text(
                processed_feature_name=processed_feature_name,
                raw_column=raw_column,
                category_value=category_value,
                shap_value=float(shap_value)
            )
        })

    explanation_rows = sorted(
        explanation_rows,
        key=lambda x: x["absolute_shap_value"],
        reverse=True
    )

    top_reasons = explanation_rows[:top_n]

    return {
        "fraud_probability": round(fraud_probability, 4),
        "threshold": round(float(threshold), 4),
        "prediction": prediction,
        "prediction_label": "Fraud" if prediction == 1 else "Not Fraud",
        "risk_level": risk_level,
        "recommendation": recommendation,
        "top_reasons": top_reasons
    }


# =========================================================
# Standalone run
# =========================================================

def main():
    """
    Create global SHAP report and run one local explanation sample.
    """

    print("=" * 70)
    print("CREATING GLOBAL SHAP REPORT")
    print("=" * 70)

    summary = create_global_shap_summary(sample_size=1000)

    print("\nGlobal SHAP outputs:")
    for key, value in summary.items():
        if key != "top_features":
            print(f"{key}: {value}")

    print("\nTop 10 global SHAP features:")
    for item in summary["top_features"][:10]:
        print(item)

    data = load_preprocessed_data()

    if "X_test_raw" in data:
        sample_claim = data["X_test_raw"].iloc[0].to_dict()

        print("\n" + "=" * 70)
        print("SAMPLE LOCAL SHAP EXPLANATION")
        print("=" * 70)

        local_explanation = explain_single_claim(sample_claim, top_n=10)

        print(json.dumps(local_explanation, indent=4))


if __name__ == "__main__":
    main()