from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import numpy as np
import joblib
import os

class PredictRequest(BaseModel):

    radius_mean: float
    texture_mean: float
    perimeter_mean: float
    area_mean: float
    smoothness_mean: float
    compactness_mean: float
    concavity_mean: float
    concave_points_mean: float
    symmetry_mean: float
    fractal_dimension_mean: float
    radius_se: float
    texture_se: float
    perimeter_se: float
    area_se: float
    smoothness_se: float
    compactness_se: float
    concavity_se: float
    concave_points_se: float
    symmetry_se: float
    fractal_dimension_se: float
    radius_worst: float
    texture_worst: float
    perimeter_worst: float
    area_worst: float
    smoothness_worst: float
    compactness_worst: float
    concavity_worst: float
    concave_points_worst: float
    symmetry_worst: float
    fractal_dimension_worst: float
    

class PredictResponse(BaseModel):
    prediction: int
    model_version: str

app = FastAPI(title="Breast Cancer Prediction API")

# Production modeli al
MODEL_URI = "models:/breast_cancer_model/Production"
try:
    model = mlflow.pyfunc.load_model(MODEL_URI)
except Exception:
    # fallback: en son run'dan yükle
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    experiment = client.get_experiment_by_name("breast_cancer_classification")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1
    )
    if not runs:
        raise RuntimeError("Model bulunamadı.")
    latest_run = runs[0]
    model = mlflow.pyfunc.load_model(f"runs:/{latest_run.info.run_id}/model")
    MODEL_URI = latest_run.info.run_id

# Scaler'ı yükle
scaler_path = "models/preprocessor.joblib"
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
else:
    scaler = None  # hata ya da fallback

@app.get("/health")
def health():
    return {"status": "ok", "model_source": MODEL_URI}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # Input dict -> array
    input_dict = req.dict()
    X = np.array([list(input_dict.values())], dtype=float)

    # scale
    if scaler is not None:
        X = scaler.transform(X)

    try:
        pred = model.predict(X)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return PredictResponse(prediction=int(pred), model_version=str(MODEL_URI))
