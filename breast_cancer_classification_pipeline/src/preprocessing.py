import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(df: pd.DataFrame, features:list):
    X = df[features].assign(const=1)
    vif_data = []
    for i, feature in enumerate(features):
        vif = variance_inflation_factor(X.values, i)
        vif_data.append({'feature': feature, 'VIF': vif})

    return pd.DataFrame(vif_data).set_index('feature')


def drop_highly_correlated(df: pd.DataFrame, threshold: float = 0.95):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    return df.drop(columns=to_drop), to_drop


def preprocess(df: pd.DataFrame, scaler: StandardScaler = None):
    if 'diagnosis' in df.columns:
        df['diagnosis'] = pd.get_dummies(df['diagnosis'], drop_first=True)

    X = df.drop('diagnosis', axis='columns')
    y = df['diagnosis'].to_numpy()

    vif_data = calculate_vif(X,X.columns.to_list())

    X, dropped_cols = drop_highly_correlated(X, threshold=0.95)

    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, vif_data, dropped_cols


