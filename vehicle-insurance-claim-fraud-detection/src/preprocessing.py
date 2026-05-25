from pathlib import Path
import json
import joblib

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from config import (
    DATA_PATH,
    ARTIFACTS_DIR,
    PREPROCESSOR_PATH,
    PREPROCESSED_DATA_PATH,
    FEATURE_NAMES_PATH,
    RAW_FEATURE_COLUMNS_PATH,
    TARGET,
    ID_COLUMNS,
    RANDOM_STATE,
    TEST_SIZE,
)

from feature_engineering import apply_feature_engineering


# =========================================================
# Data loading
# =========================================================

def load_data(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load raw fraud dataset.
    """

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)

    if TARGET not in df.columns:
        raise ValueError(f"Target column not found: {TARGET}")

    return df


# =========================================================
# Feature / target split
# =========================================================

def split_features_target(df: pd.DataFrame):
    """
    Split dataframe into X and y.
    ID columns and target column are removed from X.
    """

    drop_columns = ID_COLUMNS + [TARGET]
    existing_drop_columns = [col for col in drop_columns if col in df.columns]

    X = df.drop(columns=existing_drop_columns)
    y = df[TARGET].astype(int)

    return X, y


# =========================================================
# Preprocessor
# =========================================================

def create_one_hot_encoder():
    """
    Create OneHotEncoder with compatibility for different scikit-learn versions.
    """

    try:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    except TypeError:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )

    return encoder


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build preprocessing pipeline:
    - Numeric columns: median imputation
    - Categorical columns: most frequent imputation + one-hot encoding
    """

    numeric_features = X.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", create_one_hot_encoder())
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features)
        ],
        remainder="drop"
    )

    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """
    Get feature names after preprocessing.
    """

    try:
        feature_names = preprocessor.get_feature_names_out()
        feature_names = [str(name) for name in feature_names]
    except Exception:
        feature_names = []

    cleaned_feature_names = []

    for name in feature_names:
        name = name.replace("numeric__", "")
        name = name.replace("categorical__", "")
        cleaned_feature_names.append(name)

    return cleaned_feature_names


# =========================================================
# Main preprocessing flow
# =========================================================

def create_train_test_data(df: pd.DataFrame):
    """
    Apply feature engineering, split into train/test,
    fit preprocessor on train only, then transform train/test.
    """

    df_fe = apply_feature_engineering(df)

    X, y = split_features_target(df_fe)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    preprocessor = build_preprocessor(X_train)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = get_feature_names(preprocessor)

    return {
        "X_train_raw": X_train,
        "X_test_raw": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_processed": X_train_processed,
        "X_test_processed": X_test_processed,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "raw_feature_columns": X.columns.tolist(),
    }


def save_preprocessing_artifacts(data_dict: dict) -> None:
    """
    Save preprocessing artifacts for model training and API inference.
    """

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        data_dict["preprocessor"],
        PREPROCESSOR_PATH
    )

    joblib.dump(
        {
            "X_train_processed": data_dict["X_train_processed"],
            "X_test_processed": data_dict["X_test_processed"],
            "X_train_raw": data_dict["X_train_raw"],
            "X_test_raw": data_dict["X_test_raw"],
            "y_train": data_dict["y_train"],
            "y_test": data_dict["y_test"],
        },
        PREPROCESSED_DATA_PATH
    )

    with open(FEATURE_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(data_dict["feature_names"], f, indent=4)

    with open(RAW_FEATURE_COLUMNS_PATH, "w", encoding="utf-8") as f:
        json.dump(data_dict["raw_feature_columns"], f, indent=4)


def print_preprocessing_summary(data_dict: dict) -> None:
    """
    Print preprocessing summary.
    """

    y_train = data_dict["y_train"]
    y_test = data_dict["y_test"]

    print("=" * 70)
    print("PREPROCESSING SUMMARY")
    print("=" * 70)

    print(f"Raw train shape: {data_dict['X_train_raw'].shape}")
    print(f"Raw test shape : {data_dict['X_test_raw'].shape}")

    print(f"Processed train shape: {data_dict['X_train_processed'].shape}")
    print(f"Processed test shape : {data_dict['X_test_processed'].shape}")

    print("\nTrain target distribution:")
    print(y_train.value_counts())
    print((y_train.value_counts(normalize=True) * 100).round(2))

    print("\nTest target distribution:")
    print(y_test.value_counts())
    print((y_test.value_counts(normalize=True) * 100).round(2))

    print("\nNumber of raw features after feature engineering:")
    print(len(data_dict["raw_feature_columns"]))

    print("\nNumber of processed features after encoding:")
    print(len(data_dict["feature_names"]))

    print("\nFirst 30 processed feature names:")
    print(data_dict["feature_names"][:30])

    print("\nArtifacts saved:")
    print(f"- {PREPROCESSOR_PATH}")
    print(f"- {PREPROCESSED_DATA_PATH}")
    print(f"- {FEATURE_NAMES_PATH}")
    print(f"- {RAW_FEATURE_COLUMNS_PATH}")


def main():
    df = load_data()

    data_dict = create_train_test_data(df)

    save_preprocessing_artifacts(data_dict)

    print_preprocessing_summary(data_dict)


if __name__ == "__main__":
    main()