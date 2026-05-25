from typing import Optional
from pathlib import Path
from datetime import datetime
import uuid

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse

from api.schemas import (
    ClaimRequest,
    PredictionResponse,
    BatchPredictionRequest,
    HealthResponse,
    ExplanationResponse,
)

from api.model_service import fraud_model_service

from api.file_utils import (
    read_uploaded_file_to_dataframe,
    validate_file_prediction_input,
)

BASE_DIR = Path(__file__).resolve().parents[1]
PREDICTION_OUTPUT_DIR = BASE_DIR / "outputs" / "predictions"
PREDICTION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


app = FastAPI(
    title="Vehicle Insurance Claim Fraud Detection API",
    description=(
        "An explainable vehicle insurance claim fraud scoring API "
        "using XGBoost, feature engineering, threshold tuning and FastAPI."
    ),
    version="1.0.0",
)


# =========================================================
# Startup
# =========================================================

@app.on_event("startup")
def startup_event():
    """
    Load model artifacts when the API starts.
    """

    fraud_model_service.load_artifacts()


# =========================================================
# Helpers
# =========================================================

def pydantic_to_dict(model):
    """
    Pydantic v1/v2 compatible conversion.
    Removes None values from request body.
    """

    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_none=True)

    return model.dict(exclude_none=True)


# =========================================================
# Endpoints
# =========================================================

@app.get("/", response_model=HealthResponse)
def root():
    return {
        "status": "ok",
        "model_loaded": fraud_model_service.model is not None,
        "preprocessor_loaded": fraud_model_service.preprocessor is not None,
        "threshold": fraud_model_service.threshold,
    }


@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status": "ok",
        "model_loaded": fraud_model_service.model is not None,
        "preprocessor_loaded": fraud_model_service.preprocessor is not None,
        "threshold": fraud_model_service.threshold,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_claim(request: ClaimRequest):
    try:
        claim_data = pydantic_to_dict(request)
        result = fraud_model_service.predict(claim_data)
        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict-explain", response_model=ExplanationResponse)
def predict_claim_with_explanation(request: ClaimRequest):
    try:
        claim_data = pydantic_to_dict(request)
        result = fraud_model_service.predict_with_explanation(claim_data)
        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction explanation failed: {str(e)}"
        )


@app.post("/batch-predict")
def batch_predict_claims(request: BatchPredictionRequest):
    try:
        results = fraud_model_service.predict_batch(request.claims)

        return {
            "count": len(results),
            "results": results,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )

@app.post("/predict-file")
async def predict_file(
    file: UploadFile = File(...),
    sheet_name: Optional[str] = Form(None),
    preview_rows: int = Form(20)
):
    """
    Predict fraud risk for claims uploaded as CSV or Excel file.

    Instead of returning all prediction rows as JSON, this endpoint:
    - scores the full uploaded file,
    - saves the prediction result as a CSV file,
    - returns summary statistics and a small preview.

    Supported formats:
    - .csv
    - .xlsx
    - .xls
    """

    try:
        df = await read_uploaded_file_to_dataframe(
            file=file,
            sheet_name=sheet_name
        )

        validate_file_prediction_input(df)

        claims = df.to_dict(orient="records")

        results = fraud_model_service.predict_batch(claims)

        result_df = pd.DataFrame(results)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]

        output_filename = f"fraud_predictions_{timestamp}_{unique_id}.csv"
        output_path = PREDICTION_OUTPUT_DIR / output_filename

        result_df.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig"
        )

        fraud_count = int((result_df["prediction"] == 1).sum())
        not_fraud_count = int((result_df["prediction"] == 0).sum())

        risk_level_counts = (
            result_df["risk_level"]
            .value_counts()
            .to_dict()
            if "risk_level" in result_df.columns
            else {}
        )

        preview_df = result_df.head(preview_rows).copy()
        preview_df = preview_df.astype(object).where(pd.notnull(preview_df), None)

        return {
            "filename": file.filename,
            "input_rows": int(df.shape[0]),
            "input_columns": int(df.shape[1]),
            "result_count": int(result_df.shape[0]),
            "fraud_count": fraud_count,
            "not_fraud_count": not_fraud_count,
            "fraud_rate_percent": round(fraud_count / len(result_df) * 100, 2),
            "risk_level_counts": risk_level_counts,
            "output_file": output_filename,
            "download_url": f"/download-predictions/{output_filename}",
            "preview_rows": int(preview_rows),
            "preview": preview_df.to_dict(orient="records"),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"File prediction failed: {str(e)}"
        )
    
@app.get("/download-predictions/{file_name}")
def download_predictions(file_name: str):
    """
    Download prediction result CSV file.
    """

    safe_file_name = Path(file_name).name
    file_path = PREDICTION_OUTPUT_DIR / safe_file_name

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Prediction file not found."
        )

    return FileResponse(
        path=file_path,
        filename=safe_file_name,
        media_type="text/csv"
    )