import json
import joblib
from typing import Dict, Any, Union

import numpy as np
import pandas as pd

from config import (
    MODEL_PATH,
    PREPROCESSOR_PATH,
    THRESHOLD_PATH,
    RAW_FEATURE_COLUMNS_PATH,
    PREPROCESSED_DATA_PATH,
    DEFAULT_THRESHOLD,
    TARGET,
    ID_COLUMNS,
    RISK_LEVEL_THRESHOLDS,
)

from feature_engineering import apply_feature_engineering


# =========================================================
# Artifact loading
# =========================================================

def load_model():
    """
    Load trained model.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. Run train.py first."
        )

    return joblib.load(MODEL_PATH)


def load_preprocessor():
    """
    Load fitted preprocessor.
    """

    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessor not found: {PREPROCESSOR_PATH}. "
            "Run preprocessing.py first."
        )

    return joblib.load(PREPROCESSOR_PATH)


def load_selected_threshold() -> float:
    """
    Load selected threshold from threshold.json.
    If threshold file does not exist, fallback to DEFAULT_THRESHOLD.
    """

    if not THRESHOLD_PATH.exists():
        return DEFAULT_THRESHOLD

    with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
        threshold_data = json.load(f)

    return float(threshold_data.get("selected_threshold", DEFAULT_THRESHOLD))


def load_raw_feature_columns() -> list:
    """
    Load raw feature column names used during training after feature engineering.
    """

    if not RAW_FEATURE_COLUMNS_PATH.exists():
        raise FileNotFoundError(
            f"Raw feature columns file not found: {RAW_FEATURE_COLUMNS_PATH}. "
            "Run preprocessing.py first."
        )

    with open(RAW_FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
        raw_feature_columns = json.load(f)

    return raw_feature_columns


# =========================================================
# Input preparation
# =========================================================

def convert_input_to_dataframe(
    input_data: Union[Dict[str, Any], pd.DataFrame]
) -> pd.DataFrame:
    """
    Convert dictionary or dataframe input to dataframe.
    """

    if isinstance(input_data, pd.DataFrame):
        return input_data.copy()

    if isinstance(input_data, dict):
        return pd.DataFrame([input_data])

    raise TypeError(
        "input_data must be either a dictionary or a pandas DataFrame."
    )


def prepare_input_for_prediction(
    input_data: Union[Dict[str, Any], pd.DataFrame],
    raw_feature_columns: list
) -> pd.DataFrame:
    """
    Prepare raw input for model prediction.

    Steps:
    - Convert input to dataframe
    - Apply feature engineering
    - Drop target and ID columns if they exist
    - Align columns with training-time raw feature columns
    """

    df = convert_input_to_dataframe(input_data)

    df_fe = apply_feature_engineering(df)

    drop_columns = ID_COLUMNS + [TARGET]
    existing_drop_columns = [
        col for col in drop_columns
        if col in df_fe.columns
    ]

    X = df_fe.drop(columns=existing_drop_columns)

    # Align with training columns.
    # Missing columns are filled with NaN.
    # Extra columns are removed.
    X = X.reindex(columns=raw_feature_columns)

    return X


# =========================================================
# Business output logic
# =========================================================

def assign_risk_level(fraud_probability: float) -> str:
    """
    Assign risk level based on probability.
    """

    low_threshold = RISK_LEVEL_THRESHOLDS["low"]
    medium_threshold = RISK_LEVEL_THRESHOLDS["medium"]

    if fraud_probability < low_threshold:
        return "Low"

    if fraud_probability < medium_threshold:
        return "Medium"

    return "High"


def create_recommendation(
    fraud_probability: float,
    prediction: int,
    risk_level: str
) -> str:
    """
    Create business recommendation based on model output.
    """

    if risk_level == "High":
        return "Manual investigation required"

    if risk_level == "Medium" and prediction == 1:
        return "Additional document check recommended"

    if risk_level == "Medium" and prediction == 0:
        return "Monitor claim and continue standard process"

    return "Standard claim process"


# =========================================================
# Prediction functions
# =========================================================

def predict_single_claim(
    input_data: Dict[str, Any],
    model=None,
    preprocessor=None,
    threshold: float = None,
    raw_feature_columns: list = None
) -> Dict[str, Any]:
    """
    Predict fraud probability for a single claim record.
    """

    if model is None:
        model = load_model()

    if preprocessor is None:
        preprocessor = load_preprocessor()

    if threshold is None:
        threshold = load_selected_threshold()

    if raw_feature_columns is None:
        raw_feature_columns = load_raw_feature_columns()

    X = prepare_input_for_prediction(
        input_data=input_data,
        raw_feature_columns=raw_feature_columns
    )

    X_processed = preprocessor.transform(X)

    fraud_probability = float(model.predict_proba(X_processed)[:, 1][0])

    prediction = int(fraud_probability >= threshold)

    risk_level = assign_risk_level(fraud_probability)

    recommendation = create_recommendation(
        fraud_probability=fraud_probability,
        prediction=prediction,
        risk_level=risk_level
    )

    result = {
        "fraud_probability": round(fraud_probability, 4),
        "threshold": round(float(threshold), 4),
        "prediction": prediction,
        "prediction_label": "Fraud" if prediction == 1 else "Not Fraud",
        "risk_level": risk_level,
        "recommendation": recommendation,
    }

    return result


def predict_batch_claims(
    input_data: pd.DataFrame,
    model=None,
    preprocessor=None,
    threshold: float = None,
    raw_feature_columns: list = None
) -> pd.DataFrame:
    """
    Predict fraud probability for multiple claim records.
    """

    if model is None:
        model = load_model()

    if preprocessor is None:
        preprocessor = load_preprocessor()

    if threshold is None:
        threshold = load_selected_threshold()

    if raw_feature_columns is None:
        raw_feature_columns = load_raw_feature_columns()

    X = prepare_input_for_prediction(
        input_data=input_data,
        raw_feature_columns=raw_feature_columns
    )

    X_processed = preprocessor.transform(X)

    fraud_probabilities = model.predict_proba(X_processed)[:, 1]

    predictions = (fraud_probabilities >= threshold).astype(int)

    results = input_data.copy()

    results["fraud_probability"] = np.round(fraud_probabilities, 4)
    results["threshold"] = round(float(threshold), 4)
    results["prediction"] = predictions
    results["prediction_label"] = np.where(
        predictions == 1,
        "Fraud",
        "Not Fraud"
    )
    results["risk_level"] = [
        assign_risk_level(prob)
        for prob in fraud_probabilities
    ]
    results["recommendation"] = [
        create_recommendation(
            fraud_probability=float(prob),
            prediction=int(pred),
            risk_level=assign_risk_level(float(prob))
        )
        for prob, pred in zip(fraud_probabilities, predictions)
    ]

    return results


# =========================================================
# Smoke test
# =========================================================

def get_sample_claim_from_test_data() -> Dict[str, Any]:
    """
    Get one raw test claim from preprocessed_data.pkl for local testing.
    """

    if not PREPROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessed data not found: {PREPROCESSED_DATA_PATH}. "
            "Run preprocessing.py first."
        )

    data = joblib.load(PREPROCESSED_DATA_PATH)

    if "X_test_raw" not in data:
        raise KeyError(
            "X_test_raw not found in preprocessed data. "
            "Run updated preprocessing.py first."
        )

    sample_claim = data["X_test_raw"].iloc[0].to_dict()

    return sample_claim


def main():
    """
    Local test for prediction pipeline.
    """

    sample_claim = get_sample_claim_from_test_data()

    result = predict_single_claim(sample_claim)

    print("=" * 70)
    print("SAMPLE CLAIM PREDICTION")
    print("=" * 70)

    print("\nInput sample:")
    for key, value in sample_claim.items():
        print(f"{key}: {value}")

    print("\nPrediction result:")
    for key, value in result.items():
        print(f"{key}: {value}")

    print("=" * 70)


if __name__ == "__main__":
    main()