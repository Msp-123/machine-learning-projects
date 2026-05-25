import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dop(['id', 'Unnamed: 32'], axis=1, errors=ignore)

    return df