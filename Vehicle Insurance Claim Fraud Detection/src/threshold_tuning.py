import json
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from config import (
    PREPROCESSED_DATA_PATH,
    MODEL_PATH,
    THRESHOLD_PATH,
    MODEL_REPORT_DIR,
)


# =========================================================
# Settings
# =========================================================

MIN_ACCEPTABLE_RECALL = 0.70

THRESHOLDS = np.round(np.arange(0.10, 0.91, 0.05), 2)


# =========================================================
# Load model and data
# =========================================================

def load_model_and_data():
    """
    Load trained model and preprocessed test data.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. Run train.py first."
        )

    if not PREPROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessed data not found: {PREPROCESSED_DATA_PATH}. "
            "Run preprocessing.py first."
        )

    model = joblib.load(MODEL_PATH)
    data = joblib.load(PREPROCESSED_DATA_PATH)

    required_keys = [
        "X_test_processed",
        "y_test",
    ]

    for key in required_keys:
        if key not in data:
            raise KeyError(f"Missing key in preprocessed data: {key}")

    return model, data


# =========================================================
# Threshold evaluation
# =========================================================

def evaluate_thresholds(y_true, y_proba, thresholds) -> pd.DataFrame:
    """
    Evaluate multiple thresholds and return results as dataframe.
    """

    rows = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        predicted_fraud_count = int((y_pred == 1).sum())
        predicted_not_fraud_count = int((y_pred == 0).sum())

        rows.append({
            "threshold": float(threshold),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4),
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
            "predicted_fraud_count": predicted_fraud_count,
            "predicted_not_fraud_count": predicted_not_fraud_count,
        })

    return pd.DataFrame(rows)


def select_best_threshold(results_df: pd.DataFrame) -> dict:
    """
    Select threshold using two strategies:

    1. best_f1_threshold:
       Threshold with highest F1-score.

    2. business_selected_threshold:
       Among thresholds with recall >= MIN_ACCEPTABLE_RECALL,
       choose the one with the highest precision.
       If no threshold satisfies this, fallback to best F1 threshold.
    """

    best_f1_row = (
        results_df
        .sort_values(["f1_score", "precision", "recall"], ascending=False)
        .iloc[0]
    )

    eligible = results_df[results_df["recall"] >= MIN_ACCEPTABLE_RECALL]

    if not eligible.empty:
        business_row = (
            eligible
            .sort_values(["precision", "f1_score", "recall"], ascending=False)
            .iloc[0]
        )

        selection_reason = (
            f"Selected threshold maximizes precision while keeping recall >= "
            f"{MIN_ACCEPTABLE_RECALL}."
        )

    else:
        business_row = best_f1_row

        selection_reason = (
            f"No threshold satisfied recall >= {MIN_ACCEPTABLE_RECALL}. "
            f"Fallback selected best F1-score threshold."
        )

    selected_threshold_info = {
        "selected_threshold": float(business_row["threshold"]),
        "selection_strategy": "maximize_precision_with_minimum_recall",
        "minimum_acceptable_recall": MIN_ACCEPTABLE_RECALL,
        "selection_reason": selection_reason,

        "selected_precision": float(business_row["precision"]),
        "selected_recall": float(business_row["recall"]),
        "selected_f1_score": float(business_row["f1_score"]),
        "selected_false_positive": int(business_row["false_positive"]),
        "selected_false_negative": int(business_row["false_negative"]),
        "selected_true_positive": int(business_row["true_positive"]),
        "selected_true_negative": int(business_row["true_negative"]),

        "best_f1_threshold": float(best_f1_row["threshold"]),
        "best_f1_precision": float(best_f1_row["precision"]),
        "best_f1_recall": float(best_f1_row["recall"]),
        "best_f1_score": float(best_f1_row["f1_score"]),
        "best_f1_false_positive": int(best_f1_row["false_positive"]),
        "best_f1_false_negative": int(best_f1_row["false_negative"]),
    }

    return selected_threshold_info


# =========================================================
# Save outputs
# =========================================================

def save_threshold_artifacts(
    results_df: pd.DataFrame,
    selected_threshold_info: dict
) -> None:
    """
    Save threshold tuning outputs.
    """

    THRESHOLD_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(THRESHOLD_PATH, "w", encoding="utf-8") as f:
        json.dump(selected_threshold_info, f, indent=4)

    results_output_path = THRESHOLD_PATH.parent / "threshold_tuning_results.json"

    with open(results_output_path, "w", encoding="utf-8") as f:
        json.dump(
            results_df.to_dict(orient="records"),
            f,
            indent=4
        )


# =========================================================
# Plots
# =========================================================

def save_precision_recall_f1_plot(results_df: pd.DataFrame):
    """
    Save precision, recall and F1-score by threshold plot.
    """

    plt.figure(figsize=(9, 5))

    plt.plot(
        results_df["threshold"],
        results_df["precision"],
        marker="o",
        label="Precision"
    )

    plt.plot(
        results_df["threshold"],
        results_df["recall"],
        marker="o",
        label="Recall"
    )

    plt.plot(
        results_df["threshold"],
        results_df["f1_score"],
        marker="o",
        label="F1-score"
    )

    plt.axhline(
        y=MIN_ACCEPTABLE_RECALL,
        linestyle="--",
        label=f"Min Recall = {MIN_ACCEPTABLE_RECALL}"
    )

    plt.title("Precision, Recall and F1-score by Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = MODEL_REPORT_DIR / "threshold_precision_recall_f1.png"
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()

    return output_path


def save_error_counts_plot(results_df: pd.DataFrame):
    """
    Save false positive and false negative counts by threshold plot.
    """

    plt.figure(figsize=(9, 5))

    plt.plot(
        results_df["threshold"],
        results_df["false_positive"],
        marker="o",
        label="False Positive"
    )

    plt.plot(
        results_df["threshold"],
        results_df["false_negative"],
        marker="o",
        label="False Negative"
    )

    plt.title("False Positive and False Negative Counts by Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = MODEL_REPORT_DIR / "threshold_error_counts.png"
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()

    return output_path


def save_predicted_fraud_count_plot(results_df: pd.DataFrame):
    """
    Save predicted fraud count by threshold plot.
    """

    plt.figure(figsize=(9, 5))

    plt.plot(
        results_df["threshold"],
        results_df["predicted_fraud_count"],
        marker="o",
        label="Predicted Fraud Count"
    )

    plt.title("Predicted Fraud Count by Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Predicted Fraud Count")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = MODEL_REPORT_DIR / "threshold_predicted_fraud_count.png"
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()

    return output_path


# =========================================================
# HTML report
# =========================================================

def create_html_report(
    results_df: pd.DataFrame,
    selected_threshold_info: dict
):
    """
    Create threshold tuning HTML report.
    """

    selected_threshold = selected_threshold_info["selected_threshold"]
    best_f1_threshold = selected_threshold_info["best_f1_threshold"]

    table_html = results_df.to_html(
        index=False,
        classes="styled-table"
    )

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Vehicle Insurance Fraud Detection - Threshold Tuning</title>

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

            .summary-grid {{
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 16px;
                margin-bottom: 28px;
            }}

            .card {{
                background: white;
                padding: 20px;
                border-radius: 14px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
                text-align: center;
            }}

            .card .label {{
                font-size: 14px;
                color: #555;
            }}

            .card .value {{
                font-size: 28px;
                font-weight: bold;
                margin-top: 8px;
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

            .good {{
                background-color: #ecfdf5;
                border-left: 5px solid #10b981;
                padding: 14px;
                margin-top: 16px;
                border-radius: 8px;
            }}

            .danger {{
                background-color: #fef2f2;
                border-left: 5px solid #ef4444;
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

            <h1>Vehicle Insurance Claim Fraud Detection - Threshold Tuning Report</h1>

            <p>
                This report compares multiple decision thresholds for the fraud detection model.
                The goal is to choose a threshold that balances fraud detection recall and investigation workload.
            </p>

            <div class="summary-grid">
                <div class="card">
                    <div class="label">Selected Threshold</div>
                    <div class="value">{selected_threshold}</div>
                </div>

                <div class="card">
                    <div class="label">Precision</div>
                    <div class="value">{selected_threshold_info["selected_precision"]}</div>
                </div>

                <div class="card">
                    <div class="label">Recall</div>
                    <div class="value">{selected_threshold_info["selected_recall"]}</div>
                </div>

                <div class="card">
                    <div class="label">F1-score</div>
                    <div class="value">{selected_threshold_info["selected_f1_score"]}</div>
                </div>
            </div>

            <div class="section">
                <h2>1. Selected Threshold</h2>

                <div class="good">
                    <strong>Selected threshold:</strong> <code>{selected_threshold}</code>
                </div>

                <p>
                    <strong>Selection strategy:</strong>
                    {selected_threshold_info["selection_strategy"]}
                </p>

                <p>
                    <strong>Reason:</strong>
                    {selected_threshold_info["selection_reason"]}
                </p>

                <table class="styled-table">
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Selected Threshold</td>
                        <td>{selected_threshold}</td>
                    </tr>
                    <tr>
                        <td>Precision</td>
                        <td>{selected_threshold_info["selected_precision"]}</td>
                    </tr>
                    <tr>
                        <td>Recall</td>
                        <td>{selected_threshold_info["selected_recall"]}</td>
                    </tr>
                    <tr>
                        <td>F1-score</td>
                        <td>{selected_threshold_info["selected_f1_score"]}</td>
                    </tr>
                    <tr>
                        <td>False Positive</td>
                        <td>{selected_threshold_info["selected_false_positive"]}</td>
                    </tr>
                    <tr>
                        <td>False Negative</td>
                        <td>{selected_threshold_info["selected_false_negative"]}</td>
                    </tr>
                    <tr>
                        <td>True Positive</td>
                        <td>{selected_threshold_info["selected_true_positive"]}</td>
                    </tr>
                    <tr>
                        <td>True Negative</td>
                        <td>{selected_threshold_info["selected_true_negative"]}</td>
                    </tr>
                </table>
            </div>

            <div class="section">
                <h2>2. Best F1 Threshold</h2>

                <p>
                    The threshold with the highest F1-score is:
                    <code>{best_f1_threshold}</code>
                </p>

                <table class="styled-table">
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Best F1 Threshold</td>
                        <td>{best_f1_threshold}</td>
                    </tr>
                    <tr>
                        <td>Precision</td>
                        <td>{selected_threshold_info["best_f1_precision"]}</td>
                    </tr>
                    <tr>
                        <td>Recall</td>
                        <td>{selected_threshold_info["best_f1_recall"]}</td>
                    </tr>
                    <tr>
                        <td>F1-score</td>
                        <td>{selected_threshold_info["best_f1_score"]}</td>
                    </tr>
                    <tr>
                        <td>False Positive</td>
                        <td>{selected_threshold_info["best_f1_false_positive"]}</td>
                    </tr>
                    <tr>
                        <td>False Negative</td>
                        <td>{selected_threshold_info["best_f1_false_negative"]}</td>
                    </tr>
                </table>
            </div>

            <div class="section">
                <h2>3. Precision, Recall and F1 by Threshold</h2>
                <img src="threshold_precision_recall_f1.png" />

                <div class="note">
                    Lower thresholds usually increase recall but also increase false positives.
                    Higher thresholds usually improve precision but may miss more fraud cases.
                </div>
            </div>

            <div class="section">
                <h2>4. Error Counts by Threshold</h2>
                <img src="threshold_error_counts.png" />

                <div class="danger">
                    <strong>Business interpretation:</strong>
                    False negatives are actual fraud cases missed by the model.
                    False positives are normal claims sent to manual investigation unnecessarily.
                </div>
            </div>

            <div class="section">
                <h2>5. Investigation Workload by Threshold</h2>
                <img src="threshold_predicted_fraud_count.png" />

                <div class="note">
                    Predicted fraud count can be interpreted as approximate manual investigation workload.
                </div>
            </div>

            <div class="section">
                <h2>6. Full Threshold Comparison Table</h2>

                <div class="table-wrapper">
                    {table_html}
                </div>
            </div>

            <div class="section">
                <h2>7. Next Step</h2>

                <p>
                    The selected threshold is saved to:
                    <code>artifacts/threshold.json</code>
                </p>

                <p>
                    The prediction layer should use this selected threshold instead of the default 0.50 threshold.
                </p>
            </div>

        </div>
    </body>
    </html>
    """

    output_path = MODEL_REPORT_DIR / "threshold_tuning_report.html"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return output_path


# =========================================================
# Console report
# =========================================================

def print_threshold_summary(
    results_df: pd.DataFrame,
    selected_threshold_info: dict
):
    """
    Print threshold tuning summary.
    """

    print("=" * 70)
    print("THRESHOLD TUNING SUMMARY")
    print("=" * 70)

    print("\nFull threshold comparison:")
    print(results_df)

    print("\nSelected threshold:")
    print(f"Threshold : {selected_threshold_info['selected_threshold']}")
    print(f"Precision : {selected_threshold_info['selected_precision']}")
    print(f"Recall    : {selected_threshold_info['selected_recall']}")
    print(f"F1-score  : {selected_threshold_info['selected_f1_score']}")
    print(f"FP        : {selected_threshold_info['selected_false_positive']}")
    print(f"FN        : {selected_threshold_info['selected_false_negative']}")
    print(f"TP        : {selected_threshold_info['selected_true_positive']}")
    print(f"TN        : {selected_threshold_info['selected_true_negative']}")

    print("\nBest F1 threshold:")
    print(f"Threshold : {selected_threshold_info['best_f1_threshold']}")
    print(f"Precision : {selected_threshold_info['best_f1_precision']}")
    print(f"Recall    : {selected_threshold_info['best_f1_recall']}")
    print(f"F1-score  : {selected_threshold_info['best_f1_score']}")

    print("\nArtifacts saved:")
    print(f"- {THRESHOLD_PATH}")
    print(f"- {THRESHOLD_PATH.parent / 'threshold_tuning_results.json'}")
    print(f"- {MODEL_REPORT_DIR / 'threshold_precision_recall_f1.png'}")
    print(f"- {MODEL_REPORT_DIR / 'threshold_error_counts.png'}")
    print(f"- {MODEL_REPORT_DIR / 'threshold_predicted_fraud_count.png'}")
    print(f"- {MODEL_REPORT_DIR / 'threshold_tuning_report.html'}")

    print("=" * 70)


# =========================================================
# Main
# =========================================================

def main():
    MODEL_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    model, data = load_model_and_data()

    X_test = data["X_test_processed"]
    y_test = data["y_test"]

    y_proba = model.predict_proba(X_test)[:, 1]

    results_df = evaluate_thresholds(
        y_true=y_test,
        y_proba=y_proba,
        thresholds=THRESHOLDS
    )

    selected_threshold_info = select_best_threshold(results_df)

    save_precision_recall_f1_plot(results_df)
    save_error_counts_plot(results_df)
    save_predicted_fraud_count_plot(results_df)

    save_threshold_artifacts(
        results_df=results_df,
        selected_threshold_info=selected_threshold_info
    )

    create_html_report(
        results_df=results_df,
        selected_threshold_info=selected_threshold_info
    )

    print_threshold_summary(
        results_df=results_df,
        selected_threshold_info=selected_threshold_info
    )


if __name__ == "__main__":
    main()