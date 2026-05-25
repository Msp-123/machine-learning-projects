import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import joblib

from ingestion import load_data
from preprocessing import preprocess


def evaluate_model(model, X_train, y_train, X_val, y_val):
    # Validation üzerindeki temel metrikler
    y_pred = model.predict(X_val)
    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred),
        "recall": recall_score(y_val, y_pred),
        "f1": f1_score(y_val, y_pred),
        "roc_auc": roc_auc_score(y_val, y_pred),
    }

    # Cross-validation (train üzerinde, çoklu metrik)
    scoring = {
        "accuracy": "accuracy",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
        "roc_auc": "roc_auc"
    }

    cv_results = cross_validate(
        estimator=model,
        X=X_train,
        y=y_train,
        cv=5,
        scoring=scoring,
        return_train_score=False
    )

    metrics["cv_mean_accuracy"] = cv_results["test_accuracy"].mean()
    metrics["cv_std_accuracy"] = cv_results["test_accuracy"].std()
    metrics["cv_mean_f1"] = cv_results["test_f1"].mean()
    metrics["cv_std_f1"] = cv_results["test_f1"].std()
    metrics["cv_mean_precision"] = cv_results["test_precision"].mean()
    metrics["cv_mean_recall"] = cv_results["test_recall"].mean()
    metrics["cv_mean_roc_auc"] = cv_results["test_roc_auc"].mean()

    return metrics


def main():
    
    df = load_data("data/breast_cancer_data.csv")

    #Preprocessing
    X_scaled, y, scaler, vif_data, dropped_cols = preprocess(df)

    # Train/Test Split
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size= 0.2, random_state=34, stratify=y)

    mlflow.set_experiment('breast_cancer_classification')

    with mlflow.start_run():
        # Log preprocessing metadata
        mlflow.log_param("Dropped_Columns:",",".join(dropped_cols)) # Liste uzun olma ihtimali için Json olarak kaydedilebilir veya mlflow.log_artifact ile dosya olarak kaydedilebilir.


