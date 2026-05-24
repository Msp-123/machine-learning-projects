import json
import joblib
import numpy as np

from xgboost import XGBClassifier

from config import (
    PREPROCESSED_DATA_PATH,
    MODEL_PATH,
    ARTIFACTS_DIR,
    RANDOM_STATE,
)


# =========================================================
# Data loading
# =========================================================

def load_preprocessed_data():
    """
    Load preprocessed train/test data.
    """

    if not PREPROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessed data not found: {PREPROCESSED_DATA_PATH}. "
            "Run preprocessing.py first."
        )

    data = joblib.load(PREPROCESSED_DATA_PATH)

    required_keys = [
        "X_train_processed",
        "X_test_processed",
        "y_train",
        "y_test",
    ]

    for key in required_keys:
        if key not in data:
            raise KeyError(f"Missing key in preprocessed data: {key}")

    return data


# =========================================================
# Class imbalance handling
# =========================================================

def calculate_scale_pos_weight(y_train):
    """
    Calculate scale_pos_weight for imbalanced binary classification.

    Formula:
    scale_pos_weight = number_of_negative_samples / number_of_positive_samples
    """

    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())

    if positive_count == 0:
        raise ValueError("No positive fraud samples found in y_train.")

    scale_pos_weight = negative_count / positive_count

    return scale_pos_weight, negative_count, positive_count


# =========================================================
# Model
# =========================================================

def build_xgboost_model(scale_pos_weight: float) -> XGBClassifier:
    """
    Build XGBoost classifier.
    """

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return model


def train_model(model, X_train, y_train):
    """
    Train model.
    """

    model.fit(X_train, y_train)

    return model


# =========================================================
# Save artifacts
# =========================================================

def save_model(model):
    """
    Save trained model.
    """

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)


def save_training_summary(summary: dict):
    """
    Save training summary as JSON.
    """

    output_path = ARTIFACTS_DIR / "model_training_summary.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)


# =========================================================
# Reporting
# =========================================================

def print_training_summary(summary: dict):
    """
    Print training summary.
    """

    print("=" * 70)
    print("MODEL TRAINING SUMMARY")
    print("=" * 70)

    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nArtifacts saved:")
    print(f"- {MODEL_PATH}")
    print(f"- {ARTIFACTS_DIR / 'model_training_summary.json'}")

    print("=" * 70)


# =========================================================
# Main
# =========================================================

def main():
    data = load_preprocessed_data()

    X_train = data["X_train_processed"]
    y_train = data["y_train"]

    X_test = data["X_test_processed"]
    y_test = data["y_test"]

    scale_pos_weight, negative_count, positive_count = calculate_scale_pos_weight(y_train)

    model = build_xgboost_model(scale_pos_weight=scale_pos_weight)

    model = train_model(model, X_train, y_train)

    save_model(model)

    summary = {
        "model_type": "XGBClassifier",
        "train_rows": int(X_train.shape[0]),
        "train_features": int(X_train.shape[1]),
        "test_rows": int(X_test.shape[0]),
        "test_features": int(X_test.shape[1]),
        "train_negative_count": negative_count,
        "train_positive_count": positive_count,
        "scale_pos_weight": round(float(scale_pos_weight), 4),
        "fraud_ratio_train_percent": round(float(np.mean(y_train) * 100), 4),
        "fraud_ratio_test_percent": round(float(np.mean(y_test) * 100), 4),
        "random_state": RANDOM_STATE,
        "model_params": model.get_params(),
    }

    save_training_summary(summary)
    print_training_summary(summary)


if __name__ == "__main__":
    main()