import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/raw/fraud_oracle.csv")

def main():
    df = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print("DATASET SHAPE")
    print("=" * 60)
    print(df.shape)

    print("\n" + "=" * 60)
    print("COLUMNS")
    print("=" * 60)
    print(df.columns.tolist())

    print("\n" + "=" * 60)
    print("FIRST 5 ROWS")
    print("=" * 60)
    print(df.head())

    print("\n" + "=" * 60)
    print("MISSING VALUES")
    print("=" * 60)
    print(df.isnull().sum())

    print("\n" + "=" * 60)
    print("DUPLICATE ROWS")
    print("=" * 60)
    print(df.duplicated().sum())

    print("\n" + "=" * 60)
    print("TARGET DISTRIBUTION")
    print("=" * 60)
    print(df["FraudFound_P"].value_counts())
    print(df["FraudFound_P"].value_counts(normalize=True) * 100)

    print("\n" + "=" * 60)
    print("DATA TYPES")
    print("=" * 60)
    print(df.dtypes)

    print("\n" + "=" * 60)
    print("UNIQUE VALUES PER COLUMN")
    print("=" * 60)
    print(df.nunique().sort_values(ascending=False))


if __name__ == "__main__":
    main()