import json
import joblib
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)

from config import (
    PREPROCESSED_DATA_PATH,
    MODEL_PATH,
    METRICS_PATH,
    MODEL_REPORT_DIR,
    DEFAULT_THRESHOLD,
)


# =========================================================
# Data and model loading
# =========================================================

def load_model_and_data():
    """
    Load trained model and preprocessed data.
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
# Prediction
# =========================================================

def predict_with_threshold(model, X_test, threshold: float):
    """
    Generate probability and binary predictions.
    """

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    return y_proba, y_pred


# =========================================================
# Metrics
# =========================================================

def calculate_metrics(y_test, y_pred, y_proba, threshold: float) -> dict:
    """
    Calculate classification metrics.
    """

    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "threshold": float(threshold),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "pr_auc": round(float(average_precision_score(y_test, y_proba)), 4),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "support_not_fraud": int((y_test == 0).sum()),
        "support_fraud": int((y_test == 1).sum()),
    }

    return metrics


def create_classification_report_dict(y_test, y_pred) -> dict:
    """
    Create sklearn classification report as dictionary.
    """

    report = classification_report(
        y_test,
        y_pred,
        target_names=["Not Fraud", "Fraud"],
        output_dict=True,
        zero_division=0,
    )

    return report


# =========================================================
# Plotting
# =========================================================

def save_confusion_matrix_plot(y_test, y_pred):
    """
    Save confusion matrix plot.
    """

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()

    classes = ["Not Fraud", "Fraud"]
    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=12,
            )

    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()

    output_path = MODEL_REPORT_DIR / "confusion_matrix.png"
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()

    return output_path


def save_roc_curve_plot(y_test, y_proba):
    """
    Save ROC curve plot.
    """

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {auc_score:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Baseline")

    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate / Recall")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = MODEL_REPORT_DIR / "roc_curve.png"
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()

    return output_path


def save_precision_recall_curve_plot(y_test, y_proba):
    """
    Save Precision-Recall curve plot.
    """

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    baseline = float(np.mean(y_test))

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label=f"PR-AUC = {pr_auc:.4f}")
    plt.axhline(
        y=baseline,
        linestyle="--",
        label=f"Fraud Rate Baseline = {baseline:.4f}"
    )

    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = MODEL_REPORT_DIR / "precision_recall_curve.png"
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()

    return output_path


# =========================================================
# Save reports
# =========================================================

def save_metrics(metrics: dict, classification_report_dict: dict):
    """
    Save metrics as JSON.
    """

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "metrics": metrics,
        "classification_report": classification_report_dict,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)


def create_html_report(metrics: dict):
    """
    Create simple visual model evaluation report.
    """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Vehicle Insurance Fraud Detection - Model Evaluation</title>

        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                background-color: #f5f6f8;
                color: #222;
            }}

            .container {{
                max-width: 1100px;
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

            .styled-table {{
                border-collapse: collapse;
                width: 100%;
                font-size: 14px;
                margin-top: 12px;
            }}

            .styled-table th, .styled-table td {{
                border: 1px solid #ddd;
                padding: 10px;
                text-align: left;
            }}

            .styled-table th {{
                background-color: #f3f4f6;
            }}

            .note {{
                background-color: #fff7ed;
                border-left: 5px solid #f97316;
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

            <h1>Vehicle Insurance Claim Fraud Detection - Model Evaluation Report</h1>

            <p>
                This report summarizes the first XGBoost model evaluation on the test set.
                Since the target is highly imbalanced, accuracy should not be interpreted alone.
            </p>

            <div class="summary-grid">
                <div class="card">
                    <div class="label">Accuracy</div>
                    <div class="value">{metrics["accuracy"]}</div>
                </div>

                <div class="card">
                    <div class="label">Precision</div>
                    <div class="value">{metrics["precision"]}</div>
                </div>

                <div class="card">
                    <div class="label">Recall</div>
                    <div class="value">{metrics["recall"]}</div>
                </div>

                <div class="card">
                    <div class="label">F1-score</div>
                    <div class="value">{metrics["f1_score"]}</div>
                </div>
            </div>

            <div class="section">
                <h2>1. Main Metrics</h2>

                <table class="styled-table">
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Threshold</td>
                        <td>{metrics["threshold"]}</td>
                    </tr>
                    <tr>
                        <td>Accuracy</td>
                        <td>{metrics["accuracy"]}</td>
                    </tr>
                    <tr>
                        <td>Precision</td>
                        <td>{metrics["precision"]}</td>
                    </tr>
                    <tr>
                        <td>Recall</td>
                        <td>{metrics["recall"]}</td>
                    </tr>
                    <tr>
                        <td>F1-score</td>
                        <td>{metrics["f1_score"]}</td>
                    </tr>
                    <tr>
                        <td>ROC-AUC</td>
                        <td>{metrics["roc_auc"]}</td>
                    </tr>
                    <tr>
                        <td>PR-AUC</td>
                        <td>{metrics["pr_auc"]}</td>
                    </tr>
                </table>

                <div class="note">
                    <strong>Interpretation:</strong>
                    For this project, recall shows how many actual fraud cases the model can catch.
                    Precision shows how many predicted fraud cases are truly fraud.
                    PR-AUC is especially important because fraud cases are rare.
                </div>
            </div>

            <div class="section">
                <h2>2. Confusion Matrix Counts</h2>

                <table class="styled-table">
                    <tr>
                        <th>Type</th>
                        <th>Count</th>
                        <th>Meaning</th>
                    </tr>
                    <tr>
                        <td>True Negative</td>
                        <td>{metrics["true_negative"]}</td>
                        <td>Not fraud claims correctly predicted as not fraud.</td>
                    </tr>
                    <tr>
                        <td>False Positive</td>
                        <td>{metrics["false_positive"]}</td>
                        <td>Not fraud claims incorrectly flagged as fraud.</td>
                    </tr>
                    <tr>
                        <td>False Negative</td>
                        <td>{metrics["false_negative"]}</td>
                        <td>Fraud claims missed by the model.</td>
                    </tr>
                    <tr>
                        <td>True Positive</td>
                        <td>{metrics["true_positive"]}</td>
                        <td>Fraud claims correctly detected by the model.</td>
                    </tr>
                </table>

                <div class="danger">
                    <strong>Business note:</strong>
                    In fraud detection, false negatives are usually very costly because actual fraud cases are missed.
                    Threshold tuning should be used to control this trade-off.
                </div>
            </div>

            <div class="section">
                <h2>3. Confusion Matrix</h2>
                <img src="confusion_matrix.png" />
            </div>

            <div class="section">
                <h2>4. ROC Curve</h2>
                <img src="roc_curve.png" />
            </div>

            <div class="section">
                <h2>5. Precision-Recall Curve</h2>
                <img src="precision_recall_curve.png" />
            </div>

            <div class="section">
                <h2>6. Next Step</h2>

                <p>
                    The current evaluation uses the default threshold:
                    <code>{metrics["threshold"]}</code>.
                    The next step is to run threshold tuning and compare different thresholds
                    based on precision, recall, F1-score, false positives and false negatives.
                </p>
            </div>

        </div>
    </body>
    </html>
    """

    output_path = MODEL_REPORT_DIR / "model_evaluation_report.html"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return output_path


# =========================================================
# Console reporting
# =========================================================

def print_evaluation_summary(metrics: dict):
    """
    Print evaluation summary to terminal.
    """

    print("=" * 70)
    print("MODEL EVALUATION SUMMARY")
    print("=" * 70)

    print(f"Threshold : {metrics['threshold']}")
    print(f"Accuracy  : {metrics['accuracy']}")
    print(f"Precision : {metrics['precision']}")
    print(f"Recall    : {metrics['recall']}")
    print(f"F1-score  : {metrics['f1_score']}")
    print(f"ROC-AUC   : {metrics['roc_auc']}")
    print(f"PR-AUC    : {metrics['pr_auc']}")

    print("\nConfusion Matrix Counts:")
    print(f"TN: {metrics['true_negative']}")
    print(f"FP: {metrics['false_positive']}")
    print(f"FN: {metrics['false_negative']}")
    print(f"TP: {metrics['true_positive']}")

    print("\nArtifacts saved:")
    print(f"- {METRICS_PATH}")
    print(f"- {MODEL_REPORT_DIR / 'confusion_matrix.png'}")
    print(f"- {MODEL_REPORT_DIR / 'roc_curve.png'}")
    print(f"- {MODEL_REPORT_DIR / 'precision_recall_curve.png'}")
    print(f"- {MODEL_REPORT_DIR / 'model_evaluation_report.html'}")

    print("=" * 70)


# =========================================================
# Main
# =========================================================

def main():
    MODEL_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    model, data = load_model_and_data()

    X_test = data["X_test_processed"]
    y_test = data["y_test"]

    y_proba, y_pred = predict_with_threshold(
        model=model,
        X_test=X_test,
        threshold=DEFAULT_THRESHOLD
    )

    metrics = calculate_metrics(
        y_test=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        threshold=DEFAULT_THRESHOLD
    )

    classification_report_dict = create_classification_report_dict(
        y_test=y_test,
        y_pred=y_pred
    )

    save_confusion_matrix_plot(y_test, y_pred)
    save_roc_curve_plot(y_test, y_proba)
    save_precision_recall_curve_plot(y_test, y_proba)

    save_metrics(metrics, classification_report_dict)

    create_html_report(metrics)

    print_evaluation_summary(metrics)


if __name__ == "__main__":
    main()