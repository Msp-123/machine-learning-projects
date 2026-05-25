import base64
import html
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

DATA_PATH = Path("data/raw/fraud_oracle.csv")
REPORT_DIR = Path("reports/eda")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "FraudFound_P"
ID_COLUMNS = ["PolicyNumber"]

MAX_CATEGORICAL_PLOT_CATEGORIES = 15
MAX_NUMERIC_PLOTS = 20
MIN_CATEGORY_COUNT_FOR_RISK_TABLE = 50


# =========================================================
# Helper functions
# =========================================================

def fig_to_base64() -> str:
    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png", bbox_inches="tight", dpi=120)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close()
    return image_base64


def safe_round(value, digits=4):
    try:
        if pd.isna(value):
            return None
        return round(float(value), digits)
    except Exception:
        return value


def dataframe_to_html_table(df: pd.DataFrame, max_rows: int = 100) -> str:
    if df is None or df.empty:
        return "<p>No data available.</p>"

    df_show = df.head(max_rows).copy()

    return df_show.to_html(
        index=False,
        classes="styled-table",
        escape=True
    )


def get_column_type(df: pd.DataFrame, col: str) -> str:
    if col == TARGET:
        return "target"

    if col in ID_COLUMNS:
        return "id"

    if pd.api.types.is_numeric_dtype(df[col]):
        return "numeric"

    return "categorical"


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    confusion_matrix = pd.crosstab(x, y)

    if confusion_matrix.shape[0] < 2 or confusion_matrix.shape[1] < 2:
        return 0.0

    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()

    if n == 0:
        return 0.0

    phi2 = chi2 / n
    r, k = confusion_matrix.shape

    denominator = min(k - 1, r - 1)

    if denominator == 0:
        return 0.0

    return float(np.sqrt(phi2 / denominator))


def calculate_outlier_count_iqr(series: pd.Series) -> int:
    s = pd.to_numeric(series, errors="coerce").dropna()

    if s.empty:
        return 0

    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1

    if iqr == 0:
        return 0

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return int(((s < lower) | (s > upper)).sum())


def top_values_as_text(series: pd.Series, n: int = 5) -> str:
    vc = series.value_counts(dropna=False).head(n)

    parts = []
    for idx, val in vc.items():
        label = str(idx)
        parts.append(f"{label}: {val}")

    return " | ".join(parts)


# =========================================================
# Dataset summaries
# =========================================================

def build_column_overview(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    total_rows = len(df)

    for col in df.columns:
        missing_count = int(df[col].isnull().sum())
        missing_percent = missing_count / total_rows * 100 if total_rows else 0

        unique_count = int(df[col].nunique(dropna=False))
        unique_percent = unique_count / total_rows * 100 if total_rows else 0

        col_type = get_column_type(df, col)

        warning_list = []

        if missing_percent > 0:
            warning_list.append("missing_values")

        if unique_count == 1:
            warning_list.append("constant")

        if unique_percent > 90 and col != TARGET:
            warning_list.append("high_cardinality")

        if col in ID_COLUMNS:
            warning_list.append("id_column")

        if col == TARGET:
            warning_list.append("target_column")

        if col == "Age" and (df[col] == 0).sum() > 0:
            warning_list.append("age_zero_values")

        rows.append({
            "column": col,
            "type": col_type,
            "dtype": str(df[col].dtype),
            "missing_count": missing_count,
            "missing_percent": round(missing_percent, 2),
            "unique_count": unique_count,
            "unique_percent": round(unique_percent, 2),
            "top_values": top_values_as_text(df[col], n=5),
            "warnings": ", ".join(warning_list) if warning_list else ""
        })

    return pd.DataFrame(rows)


def build_numeric_profile(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        col for col in df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns
        if col != TARGET and col not in ID_COLUMNS
    ]

    rows = []

    for col in numeric_cols:
        s = pd.to_numeric(df[col], errors="coerce")

        rows.append({
            "column": col,
            "count": int(s.count()),
            "missing": int(s.isnull().sum()),
            "mean": safe_round(s.mean(), 4),
            "std": safe_round(s.std(), 4),
            "min": safe_round(s.min(), 4),
            "p01": safe_round(s.quantile(0.01), 4),
            "p05": safe_round(s.quantile(0.05), 4),
            "p25": safe_round(s.quantile(0.25), 4),
            "median": safe_round(s.median(), 4),
            "p75": safe_round(s.quantile(0.75), 4),
            "p95": safe_round(s.quantile(0.95), 4),
            "p99": safe_round(s.quantile(0.99), 4),
            "max": safe_round(s.max(), 4),
            "skew": safe_round(s.skew(), 4),
            "kurtosis": safe_round(s.kurtosis(), 4),
            "zero_count": int((s == 0).sum()),
            "outlier_count_iqr": calculate_outlier_count_iqr(s),
            "fraud_mean": safe_round(df.loc[df[TARGET] == 1, col].mean(), 4),
            "non_fraud_mean": safe_round(df.loc[df[TARGET] == 0, col].mean(), 4),
        })

    return pd.DataFrame(rows)


def build_categorical_profile(df: pd.DataFrame) -> pd.DataFrame:
    categorical_cols = [
        col for col in df.select_dtypes(include=["object"]).columns
        if col != TARGET and col not in ID_COLUMNS
    ]

    rows = []

    for col in categorical_cols:
        vc = df[col].value_counts(dropna=False)

        grouped = (
            df.groupby(col, dropna=False)[TARGET]
            .agg(["count", "sum", "mean"])
            .reset_index()
        )

        max_fraud_rate = grouped["mean"].max() * 100 if not grouped.empty else 0
        min_fraud_rate = grouped["mean"].min() * 100 if not grouped.empty else 0

        highest_risk_category = ""
        highest_risk_rate = None

        grouped_filtered = grouped[grouped["count"] >= MIN_CATEGORY_COUNT_FOR_RISK_TABLE]

        if not grouped_filtered.empty:
            top_row = grouped_filtered.sort_values("mean", ascending=False).iloc[0]
            highest_risk_category = str(top_row[col])
            highest_risk_rate = round(top_row["mean"] * 100, 2)

        rows.append({
            "column": col,
            "unique_count": int(df[col].nunique(dropna=False)),
            "most_common_value": str(vc.index[0]) if len(vc) > 0 else "",
            "most_common_count": int(vc.iloc[0]) if len(vc) > 0 else 0,
            "most_common_percent": round(vc.iloc[0] / len(df) * 100, 2) if len(vc) > 0 else 0,
            "highest_risk_category_min_count_50": highest_risk_category,
            "highest_risk_category_fraud_rate_percent": highest_risk_rate,
            "min_fraud_rate_percent": round(min_fraud_rate, 2),
            "max_fraud_rate_percent": round(max_fraud_rate, 2),
            "cramers_v_with_target": round(cramers_v(df[col], df[TARGET]), 4),
            "top_values": top_values_as_text(df[col], n=5)
        })

    result = pd.DataFrame(rows)

    if not result.empty:
        result = result.sort_values("cramers_v_with_target", ascending=False)

    return result


def build_high_risk_categories(df: pd.DataFrame) -> pd.DataFrame:
    categorical_cols = [
        col for col in df.select_dtypes(include=["object"]).columns
        if col not in ID_COLUMNS
    ]

    rows = []

    for col in categorical_cols:
        grouped = (
            df.groupby(col, dropna=False)[TARGET]
            .agg(["count", "sum", "mean"])
            .reset_index()
        )

        grouped = grouped[grouped["count"] >= MIN_CATEGORY_COUNT_FOR_RISK_TABLE]

        for _, row in grouped.iterrows():
            rows.append({
                "column": col,
                "category": str(row[col]),
                "total_count": int(row["count"]),
                "fraud_count": int(row["sum"]),
                "fraud_rate_percent": round(row["mean"] * 100, 2),
                "lift_vs_dataset_avg": None
            })

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    dataset_fraud_rate = df[TARGET].mean() * 100
    result["lift_vs_dataset_avg"] = (
        result["fraud_rate_percent"] / dataset_fraud_rate
    ).round(2)

    return result.sort_values("fraud_rate_percent", ascending=False).head(25)


def build_numeric_target_relationship(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        col for col in df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns
        if col != TARGET and col not in ID_COLUMNS
    ]

    rows = []

    for col in numeric_cols:
        s = pd.to_numeric(df[col], errors="coerce")

        if s.nunique() <= 1:
            corr = 0
        else:
            corr = s.corr(df[TARGET])

        fraud_values = s[df[TARGET] == 1]
        non_fraud_values = s[df[TARGET] == 0]

        rows.append({
            "column": col,
            "correlation_with_target": safe_round(corr, 4),
            "fraud_mean": safe_round(fraud_values.mean(), 4),
            "non_fraud_mean": safe_round(non_fraud_values.mean(), 4),
            "mean_difference": safe_round(fraud_values.mean() - non_fraud_values.mean(), 4),
            "fraud_median": safe_round(fraud_values.median(), 4),
            "non_fraud_median": safe_round(non_fraud_values.median(), 4),
        })

    result = pd.DataFrame(rows)

    if not result.empty:
        result["abs_corr"] = result["correlation_with_target"].abs()
        result = result.sort_values("abs_corr", ascending=False).drop(columns=["abs_corr"])

    return result


def build_data_quality_alerts(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total_rows = len(df)

    for col in df.columns:
        missing_count = int(df[col].isnull().sum())
        missing_percent = missing_count / total_rows * 100 if total_rows else 0

        unique_count = int(df[col].nunique(dropna=False))
        unique_percent = unique_count / total_rows * 100 if total_rows else 0

        if missing_count > 0:
            rows.append({
                "severity": "medium",
                "column": col,
                "alert": "Missing values detected",
                "details": f"{missing_count} missing values ({missing_percent:.2f}%)"
            })

        if unique_count == 1:
            rows.append({
                "severity": "high",
                "column": col,
                "alert": "Constant column",
                "details": "Column has only one unique value"
            })

        if unique_percent > 90 and col != TARGET:
            rows.append({
                "severity": "medium",
                "column": col,
                "alert": "High cardinality",
                "details": f"{unique_count} unique values ({unique_percent:.2f}%)"
            })

        if col in ID_COLUMNS:
            rows.append({
                "severity": "high",
                "column": col,
                "alert": "Identifier column",
                "details": "Should be excluded from model features"
            })

        if col == "Age":
            zero_count = int((df[col] == 0).sum())
            if zero_count > 0:
                rows.append({
                    "severity": "medium",
                    "column": col,
                    "alert": "Suspicious zero values",
                    "details": f"Age has {zero_count} rows with value 0"
                })

    if df[TARGET].mean() < 0.10:
        rows.append({
            "severity": "high",
            "column": TARGET,
            "alert": "Imbalanced target",
            "details": f"Fraud class ratio is {df[TARGET].mean() * 100:.2f}%"
        })

    duplicate_count = int(df.duplicated().sum())
    if duplicate_count > 0:
        rows.append({
            "severity": "medium",
            "column": "ALL",
            "alert": "Duplicate rows",
            "details": f"{duplicate_count} duplicate rows found"
        })

    return pd.DataFrame(rows)


# =========================================================
# Plot functions
# =========================================================

def plot_target_distribution(df: pd.DataFrame) -> str:
    target_counts = df[TARGET].value_counts().sort_index()

    plt.figure(figsize=(6, 4))
    target_counts.plot(kind="bar")
    plt.title("Target Distribution")
    plt.xlabel("FraudFound_P")
    plt.ylabel("Count")
    plt.xticks(rotation=0)

    return fig_to_base64()


def plot_target_ratio_pie(df: pd.DataFrame) -> str:
    counts = df[TARGET].value_counts().sort_index()
    labels = ["Not Fraud", "Fraud"]

    plt.figure(figsize=(5, 5))
    plt.pie(counts.values, labels=labels, autopct="%1.1f%%", startangle=90)
    plt.title("Fraud vs Non-Fraud Ratio")

    return fig_to_base64()


def plot_missing_values(df: pd.DataFrame) -> str:
    missing = df.isnull().sum().sort_values(ascending=False)
    missing = missing[missing > 0]

    plt.figure(figsize=(8, 4))

    if missing.empty:
        plt.text(0.5, 0.5, "No missing values detected", ha="center", va="center", fontsize=14)
        plt.axis("off")
    else:
        missing.plot(kind="bar")
        plt.title("Missing Values by Column")
        plt.xlabel("Column")
        plt.ylabel("Missing Count")
        plt.xticks(rotation=45, ha="right")

    return fig_to_base64()


def plot_numeric_histograms(df: pd.DataFrame) -> str:
    numeric_cols = [
        col for col in df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns
        if col != TARGET and col not in ID_COLUMNS
    ]

    numeric_cols = numeric_cols[:MAX_NUMERIC_PLOTS]

    if not numeric_cols:
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, "No numeric columns available", ha="center", va="center")
        plt.axis("off")
        return fig_to_base64()

    n_cols = 3
    n_rows = int(np.ceil(len(numeric_cols) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(numeric_cols):
        axes[i].hist(df[col].dropna(), bins=30)
        axes[i].set_title(col)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Count")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Numeric Feature Distributions", fontsize=16)

    return fig_to_base64()


def plot_correlation_matrix(df: pd.DataFrame) -> str:
    numeric_cols = [
        col for col in df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns
        if col not in ID_COLUMNS
    ]

    if len(numeric_cols) < 2:
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, "Not enough numeric columns for correlation matrix", ha="center", va="center")
        plt.axis("off")
        return fig_to_base64()

    corr = df[numeric_cols].corr()

    plt.figure(figsize=(10, 8))
    plt.imshow(corr, aspect="auto")
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Numeric Correlation Matrix")

    return fig_to_base64()


def plot_numeric_target_relationship(df: pd.DataFrame) -> str:
    numeric_cols = [
        col for col in df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns
        if col != TARGET and col not in ID_COLUMNS
    ]

    numeric_cols = numeric_cols[:MAX_NUMERIC_PLOTS]

    if not numeric_cols:
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, "No numeric columns available", ha="center", va="center")
        plt.axis("off")
        return fig_to_base64()

    n_cols = 3
    n_rows = int(np.ceil(len(numeric_cols) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(numeric_cols):
        data = [
            df.loc[df[TARGET] == 0, col].dropna(),
            df.loc[df[TARGET] == 1, col].dropna()
        ]

        axes[i].boxplot(data, tick_labels=["Not Fraud", "Fraud"])
        axes[i].set_title(col)
        axes[i].set_ylabel("Value")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Numeric Features by Target", fontsize=16)

    return fig_to_base64()


def plot_fraud_rate_by_column(df: pd.DataFrame, column: str) -> str:
    fraud_rate = (
        df.groupby(column, dropna=False)[TARGET]
        .agg(["count", "mean"])
        .reset_index()
    )

    fraud_rate = fraud_rate.sort_values("mean", ascending=False)
    fraud_rate = fraud_rate.head(MAX_CATEGORICAL_PLOT_CATEGORIES)

    labels = fraud_rate[column].astype(str)
    values = fraud_rate["mean"] * 100

    plt.figure(figsize=(9, 4))
    plt.bar(labels, values)
    plt.title(f"Fraud Rate by {column}")
    plt.xlabel(column)
    plt.ylabel("Fraud Rate (%)")
    plt.xticks(rotation=45, ha="right")

    return fig_to_base64()


def plot_time_based_fraud(df: pd.DataFrame) -> str:
    time_cols = [
        col for col in ["Year", "Month", "WeekOfMonth", "DayOfWeek", "MonthClaimed", "WeekOfMonthClaimed", "DayOfWeekClaimed"]
        if col in df.columns
    ]

    if not time_cols:
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, "No time-based columns available", ha="center", va="center")
        plt.axis("off")
        return fig_to_base64()

    n_cols = 2
    n_rows = int(np.ceil(len(time_cols) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 4))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(time_cols):
        fraud_rate = df.groupby(col, dropna=False)[TARGET].mean() * 100
        fraud_rate.plot(kind="bar", ax=axes[i])
        axes[i].set_title(f"Fraud Rate by {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Fraud Rate (%)")
        axes[i].tick_params(axis="x", rotation=45)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Time-Based Fraud Rate Analysis", fontsize=16)

    return fig_to_base64()


def plot_categorical_cardinality(df: pd.DataFrame) -> str:
    categorical_cols = [
        col for col in df.select_dtypes(include=["object"]).columns
        if col not in ID_COLUMNS
    ]

    if not categorical_cols:
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, "No categorical columns available", ha="center", va="center")
        plt.axis("off")
        return fig_to_base64()

    cardinality = df[categorical_cols].nunique(dropna=False).sort_values(ascending=False)

    plt.figure(figsize=(10, 5))
    cardinality.plot(kind="bar")
    plt.title("Categorical Cardinality")
    plt.xlabel("Column")
    plt.ylabel("Unique Count")
    plt.xticks(rotation=45, ha="right")

    return fig_to_base64()


def build_important_feature_plots(df: pd.DataFrame) -> str:
    important_columns = [
        "Fault",
        "BasePolicy",
        "VehicleCategory",
        "AddressChange_Claim",
        "PastNumberOfClaims",
        "AgeOfVehicle",
        "VehiclePrice",
        "PoliceReportFiled",
        "WitnessPresent",
        "AgentType",
        "AccidentArea",
        "PolicyType",
        "Sex",
        "MaritalStatus",
        "NumberOfSuppliments",
        "Days_Policy_Accident",
        "Days_Policy_Claim",
        "NumberOfCars",
    ]

    plot_sections = ""

    for col in important_columns:
        if col in df.columns:
            image = plot_fraud_rate_by_column(df, col)
            plot_sections += f"""
            <div class="chart-card">
                <h3>Fraud Rate by {html.escape(col)}</h3>
                <img src="data:image/png;base64,{image}" />
            </div>
            """

    return plot_sections


# =========================================================
# HTML report
# =========================================================

def create_report(df: pd.DataFrame) -> str:
    total_rows = df.shape[0]
    total_columns = df.shape[1]

    fraud_count = int(df[TARGET].sum())
    non_fraud_count = int((df[TARGET] == 0).sum())
    fraud_ratio = fraud_count / total_rows * 100

    missing_count = int(df.isnull().sum().sum())
    duplicate_count = int(df.duplicated().sum())

    age_zero_count = int((df["Age"] == 0).sum()) if "Age" in df.columns else 0
    policy_number_unique = int(df["PolicyNumber"].nunique()) if "PolicyNumber" in df.columns else 0

    column_overview = build_column_overview(df)
    numeric_profile = build_numeric_profile(df)
    categorical_profile = build_categorical_profile(df)
    high_risk_categories = build_high_risk_categories(df)
    numeric_target_relationship = build_numeric_target_relationship(df)
    data_quality_alerts = build_data_quality_alerts(df)

    target_plot = plot_target_distribution(df)
    target_pie = plot_target_ratio_pie(df)
    missing_plot = plot_missing_values(df)
    numeric_histograms = plot_numeric_histograms(df)
    numeric_target_plot = plot_numeric_target_relationship(df)
    correlation_plot = plot_correlation_matrix(df)
    categorical_cardinality_plot = plot_categorical_cardinality(df)
    time_based_plot = plot_time_based_fraud(df)
    important_feature_plots = build_important_feature_plots(df)

    column_overview_table = dataframe_to_html_table(column_overview, max_rows=200)
    numeric_profile_table = dataframe_to_html_table(numeric_profile, max_rows=200)
    categorical_profile_table = dataframe_to_html_table(categorical_profile, max_rows=200)
    high_risk_table = dataframe_to_html_table(high_risk_categories, max_rows=50)
    numeric_target_table = dataframe_to_html_table(numeric_target_relationship, max_rows=100)
    data_quality_table = dataframe_to_html_table(data_quality_alerts, max_rows=100)

    html_report = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Vehicle Insurance Fraud Detection - Detailed EDA Report</title>

        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                background-color: #f5f6f8;
                color: #222;
            }}

            .container {{
                max-width: 1280px;
                margin: auto;
                padding: 36px;
            }}

            h1, h2, h3 {{
                color: #111827;
            }}

            h1 {{
                margin-bottom: 8px;
            }}

            .subtitle {{
                color: #4b5563;
                margin-bottom: 28px;
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

            .card .value {{
                font-size: 28px;
                font-weight: bold;
                margin-top: 8px;
            }}

            .card .label {{
                font-size: 14px;
                color: #555;
            }}

            .section {{
                background: white;
                padding: 24px;
                border-radius: 14px;
                margin-bottom: 28px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
            }}

            .chart-card {{
                background: white;
                padding: 24px;
                border-radius: 14px;
                margin-bottom: 24px;
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
                font-size: 13px;
                margin-top: 12px;
            }}

            .styled-table th, .styled-table td {{
                border: 1px solid #ddd;
                padding: 9px;
                text-align: left;
                vertical-align: top;
            }}

            .styled-table th {{
                background-color: #f3f4f6;
                font-weight: bold;
            }}

            .table-wrapper {{
                overflow-x: auto;
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

            .toc {{
                background: white;
                padding: 20px;
                border-radius: 14px;
                margin-bottom: 28px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
            }}

            .toc a {{
                display: block;
                margin: 6px 0;
                color: #2563eb;
                text-decoration: none;
            }}

            .toc a:hover {{
                text-decoration: underline;
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

            <h1>Vehicle Insurance Claim Fraud Detection - Detailed EDA Report</h1>
            <p class="subtitle">
                This report provides a detailed exploratory data analysis for the vehicle insurance claim fraud detection project.
                It combines general dataset profiling with fraud-specific business analysis.
            </p>

            <div class="summary-grid">
                <div class="card">
                    <div class="label">Rows</div>
                    <div class="value">{total_rows:,}</div>
                </div>

                <div class="card">
                    <div class="label">Columns</div>
                    <div class="value">{total_columns}</div>
                </div>

                <div class="card">
                    <div class="label">Fraud Cases</div>
                    <div class="value">{fraud_count:,}</div>
                </div>

                <div class="card">
                    <div class="label">Fraud Rate</div>
                    <div class="value">{fraud_ratio:.2f}%</div>
                </div>
            </div>

            <div class="toc">
                <h2>Report Sections</h2>
                <a href="#overview">1. Dataset Overview</a>
                <a href="#quality">2. Data Quality Alerts</a>
                <a href="#target">3. Target Analysis</a>
                <a href="#columns">4. Column-Level Profile</a>
                <a href="#numeric">5. Numeric Variable Profile</a>
                <a href="#categorical">6. Categorical Variable Profile</a>
                <a href="#missing">7. Missing Values Analysis</a>
                <a href="#correlation">8. Correlation Analysis</a>
                <a href="#target-relationship">9. Target Relationship Analysis</a>
                <a href="#risk-categories">10. High-Risk Category Analysis</a>
                <a href="#time">11. Time-Based Fraud Analysis</a>
                <a href="#feature-plots">12. Fraud Rate by Important Features</a>
                <a href="#recommendations">13. Modeling Recommendations</a>
            </div>

            <div class="section" id="overview">
                <h2>1. Dataset Overview</h2>

                <div class="table-wrapper">
                    <table class="styled-table">
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                        <tr>
                            <td>Total Rows</td>
                            <td>{total_rows:,}</td>
                        </tr>
                        <tr>
                            <td>Total Columns</td>
                            <td>{total_columns}</td>
                        </tr>
                        <tr>
                            <td>Target Column</td>
                            <td>{TARGET}</td>
                        </tr>
                        <tr>
                            <td>Non-Fraud Cases</td>
                            <td>{non_fraud_count:,}</td>
                        </tr>
                        <tr>
                            <td>Fraud Cases</td>
                            <td>{fraud_count:,}</td>
                        </tr>
                        <tr>
                            <td>Fraud Rate</td>
                            <td>{fraud_ratio:.2f}%</td>
                        </tr>
                        <tr>
                            <td>Missing Values</td>
                            <td>{missing_count}</td>
                        </tr>
                        <tr>
                            <td>Duplicate Rows</td>
                            <td>{duplicate_count}</td>
                        </tr>
                        <tr>
                            <td>Age = 0 Count</td>
                            <td>{age_zero_count}</td>
                        </tr>
                        <tr>
                            <td>PolicyNumber Unique Count</td>
                            <td>{policy_number_unique}</td>
                        </tr>
                    </table>
                </div>

                <div class="danger">
                    <strong>Important:</strong> This is an imbalanced binary classification problem.
                    The fraud class ratio is only <strong>{fraud_ratio:.2f}%</strong>.
                    Therefore, accuracy alone is not reliable for model evaluation.
                </div>
            </div>

            <div class="section" id="quality">
                <h2>2. Data Quality Alerts</h2>
                <p>
                    This section highlights potential data quality issues such as high-cardinality columns,
                    suspicious values, identifier columns and target imbalance.
                </p>

                <div class="table-wrapper">
                    {data_quality_table}
                </div>
            </div>

            <div class="section" id="target">
                <h2>3. Target Analysis</h2>

                <h3>Target Distribution</h3>
                <img src="data:image/png;base64,{target_plot}" />

                <h3>Fraud vs Non-Fraud Ratio</h3>
                <img src="data:image/png;base64,{target_pie}" />

                <div class="note">
                    In this dataset, fraud cases are rare. For this reason, model development should focus on
                    <strong>Recall</strong>, <strong>Precision</strong>, <strong>F1-score</strong>,
                    <strong>PR-AUC</strong> and confusion matrix analysis.
                </div>
            </div>

            <div class="section" id="columns">
                <h2>4. Column-Level Profile</h2>
                <p>
                    This table gives a profiling-style overview for every column:
                    data type, missing values, unique counts, most frequent values and warnings.
                </p>

                <div class="table-wrapper">
                    {column_overview_table}
                </div>
            </div>

            <div class="section" id="numeric">
                <h2>5. Numeric Variable Profile</h2>
                <p>
                    This section includes descriptive statistics, quantiles, skewness, kurtosis,
                    zero counts, IQR-based outlier counts and target-based mean comparison.
                </p>

                <div class="table-wrapper">
                    {numeric_profile_table}
                </div>

                <h3>Numeric Feature Distributions</h3>
                <img src="data:image/png;base64,{numeric_histograms}" />
            </div>

            <div class="section" id="categorical">
                <h2>6. Categorical Variable Profile</h2>
                <p>
                    This section summarizes categorical variables, including cardinality,
                    most common values, fraud-rate ranges and Cramér's V association with the target.
                </p>

                <div class="table-wrapper">
                    {categorical_profile_table}
                </div>

                <h3>Categorical Cardinality</h3>
                <img src="data:image/png;base64,{categorical_cardinality_plot}" />
            </div>

            <div class="section" id="missing">
                <h2>7. Missing Values Analysis</h2>
                <p>
                    This section checks missing values across all columns.
                    Note that values such as <code>Age = 0</code> are not technically missing,
                    but they can still be suspicious and should be handled during preprocessing.
                </p>

                <img src="data:image/png;base64,{missing_plot}" />
            </div>

            <div class="section" id="correlation">
                <h2>8. Correlation Analysis</h2>
                <p>
                    This section shows the correlation matrix for numeric variables.
                    Since most variables in this dataset are categorical, correlation analysis should not be used alone.
                </p>

                <img src="data:image/png;base64,{correlation_plot}" />
            </div>

            <div class="section" id="target-relationship">
                <h2>9. Target Relationship Analysis</h2>

                <h3>Numeric Variables vs Target</h3>
                <div class="table-wrapper">
                    {numeric_target_table}
                </div>

                <h3>Numeric Feature Distribution by Target</h3>
                <img src="data:image/png;base64,{numeric_target_plot}" />

                <div class="note">
                    Numeric differences between fraud and non-fraud groups are useful,
                    but the strongest fraud signals in this dataset may come from categorical variables.
                </div>
            </div>

            <div class="section" id="risk-categories">
                <h2>10. High-Risk Category Analysis</h2>
                <p>
                    This table lists categories with the highest fraud rates.
                    Categories with very low sample size are filtered out using a minimum count threshold.
                </p>

                <div class="table-wrapper">
                    {high_risk_table}
                </div>
            </div>

            <div class="section" id="time">
                <h2>11. Time-Based Fraud Analysis</h2>
                <p>
                    This section analyzes fraud rates across year, month, week and day-related fields.
                </p>

                <img src="data:image/png;base64,{time_based_plot}" />
            </div>

            <div id="feature-plots">
                <h2>12. Fraud Rate by Important Features</h2>
                <p>
                    The following charts show fraud rate differences across important business variables.
                </p>

                {important_feature_plots}
            </div>

            <div class="section" id="recommendations">
                <h2>13. Initial Modeling Recommendations</h2>

                <div class="good">
                    <strong>Target:</strong> Use <code>{TARGET}</code> as the binary target variable.
                </div>

                <div class="good">
                    <strong>Main model:</strong> Use <code>XGBoostClassifier</code> as the primary model.
                </div>

                <div class="good">
                    <strong>Recommended metrics:</strong>
                    Precision, Recall, F1-score, ROC-AUC, PR-AUC and confusion matrix.
                </div>

                <div class="note">
                    <strong>Class imbalance:</strong>
                    Since fraud rate is {fraud_ratio:.2f}%, use stratified train/test split,
                    class weighting or <code>scale_pos_weight</code> in XGBoost.
                </div>

                <div class="note">
                    <strong>Threshold tuning:</strong>
                    Do not rely only on the default 0.50 threshold.
                    Compare thresholds such as 0.20, 0.30, 0.40 and 0.50 based on recall,
                    precision and investigation capacity.
                </div>

                <div class="danger">
                    <strong>Column exclusion:</strong>
                    <code>PolicyNumber</code> should be excluded from model training because it behaves like an ID column.
                </div>

                <div class="note">
                    <strong>Suspicious value:</strong>
                    <code>Age = 0</code> appears in the dataset.
                    Add an <code>AgeUnknown</code> flag or handle it during preprocessing.
                </div>

                <div class="good">
                    <strong>Explainability:</strong>
                    Use SHAP after model training to explain both global model behavior
                    and individual claim-level fraud predictions.
                </div>
            </div>

        </div>
    </body>
    </html>
    """

    return html_report


def main():
    df = pd.read_csv(DATA_PATH)

    if TARGET not in df.columns:
        raise ValueError(f"Target column not found: {TARGET}")

    report_html = create_report(df)

    output_path = REPORT_DIR / "eda_report.html"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_html)

    print("=" * 70)
    print("DETAILED EDA REPORT CREATED")
    print("=" * 70)
    print(f"Report path: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()