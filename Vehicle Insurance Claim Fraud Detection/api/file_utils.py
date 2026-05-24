from io import BytesIO
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import UploadFile


SUPPORTED_FILE_EXTENSIONS = [".csv", ".xlsx", ".xls"]


def get_file_extension(filename: str) -> str:
    """
    Return lowercase file extension.
    """

    return Path(filename).suffix.lower()


async def read_uploaded_file_to_dataframe(
    file: UploadFile,
    sheet_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Read uploaded CSV or Excel file into pandas DataFrame.

    Supported formats:
    - .csv
    - .xlsx
    - .xls
    """

    filename = file.filename

    if filename is None:
        raise ValueError("Uploaded file must have a filename.")

    extension = get_file_extension(filename)

    if extension not in SUPPORTED_FILE_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format: {extension}. "
            f"Supported formats are: {SUPPORTED_FILE_EXTENSIONS}"
        )

    content = await file.read()

    if len(content) == 0:
        raise ValueError("Uploaded file is empty.")

    buffer = BytesIO(content)

    if extension == ".csv":
        try:
            df = pd.read_csv(buffer)
        except UnicodeDecodeError:
            buffer.seek(0)
            df = pd.read_csv(buffer, encoding="latin1")

    elif extension in [".xlsx", ".xls"]:
        if sheet_name:
            df = pd.read_excel(buffer, sheet_name=sheet_name)
        else:
            df = pd.read_excel(buffer)

    else:
        raise ValueError(f"Unsupported file format: {extension}")

    if df.empty:
        raise ValueError("Uploaded file was read successfully but contains no rows.")

    return df


def validate_file_prediction_input(df: pd.DataFrame) -> None:
    """
    Basic validation for uploaded prediction file.
    """

    if df.empty:
        raise ValueError("Input dataframe is empty.")

    if df.shape[0] == 0:
        raise ValueError("Input file has no data rows.")

    if df.shape[1] == 0:
        raise ValueError("Input file has no columns.")