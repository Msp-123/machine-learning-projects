import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


# =========================================================
# Make src imports available
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


from predict import (
    load_model,
    load_preprocessor,
    load_selected_threshold,
    load_raw_feature_columns,
    predict_single_claim,
    predict_batch_claims,
)

from explain import (
    load_feature_names,
    build_shap_explainer,
    explain_single_claim,
)


# =========================================================
# Model service
# =========================================================

class FraudModelService:
    """
    Service layer for loading model artifacts once and serving predictions.
    """

    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.threshold = None
        self.raw_feature_columns = None
        self.feature_names = None
        self.explainer = None

    def load_artifacts(self) -> None:
        """
        Load model, preprocessor, threshold, feature columns and SHAP explainer.
        """

        self.model = load_model()
        self.preprocessor = load_preprocessor()
        self.threshold = load_selected_threshold()
        self.raw_feature_columns = load_raw_feature_columns()
        self.feature_names = load_feature_names()
        self.explainer = build_shap_explainer(self.model)

    def is_ready(self) -> bool:
        """
        Check whether all required artifacts are loaded.
        """

        return (
            self.model is not None
            and self.preprocessor is not None
            and self.threshold is not None
            and self.raw_feature_columns is not None
            and self.feature_names is not None
            and self.explainer is not None
        )

    def predict(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict fraud risk for a single claim.
        """

        if not self.is_ready():
            self.load_artifacts()

        result = predict_single_claim(
            input_data=claim_data,
            model=self.model,
            preprocessor=self.preprocessor,
            threshold=self.threshold,
            raw_feature_columns=self.raw_feature_columns,
        )

        return result

    def predict_batch(self, claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Predict fraud risk for multiple claims.
        """

        if not self.is_ready():
            self.load_artifacts()

        input_df = pd.DataFrame(claims)

        result_df = predict_batch_claims(
            input_data=input_df,
            model=self.model,
            preprocessor=self.preprocessor,
            threshold=self.threshold,
            raw_feature_columns=self.raw_feature_columns,
        )

        return result_df.to_dict(orient="records")


    def predict_with_explanation(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict fraud risk for a single claim and explain prediction using SHAP.
        """

        if not self.is_ready():
            self.load_artifacts()

        result = explain_single_claim(
            input_data=claim_data,
            model=self.model,
            preprocessor=self.preprocessor,
            threshold=self.threshold,
            raw_feature_columns=self.raw_feature_columns,
            feature_names=self.feature_names,
            explainer=self.explainer,
            top_n=10,
        )

        return result

fraud_model_service = FraudModelService()