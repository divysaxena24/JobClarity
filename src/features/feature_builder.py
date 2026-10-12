"""
Feature engineering utilities for JobClarity.
"""

import pandas as pd


def create_combined_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine important text columns into a single feature.
    """

    text_columns = [
        "title",
        "company_profile",
        "description",
        "requirements",
        "benefits"
    ]

    # Validate required columns exist
    missing_cols = [c for c in text_columns if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns in dataframe: {missing_cols}")

    if df.empty:
        df["combined_text"] = pd.Series(dtype=object)
        return df

    df["combined_text"] = (
        df[text_columns]
        .fillna("")
        .agg(" ".join, axis=1)
    )

    return df


def prepare_job_description_text(
    data: pd.DataFrame | pd.Series | str | list[str],
    text_column: str = "description",
) -> pd.Series | str:
    """
    Extract and clean job description text for unified training and inference representation.

    Ensures zero training-serving skew:
    - Training: extracts the job description column ('description' or 'job_description') and cleans it.
    - Inference: cleans the single incoming job description string.
    - Does not invent text, duplicate text across fields, or assume missing optional fields.

    Parameters
    ----------
    data : pd.DataFrame, pd.Series, str, or list of str
        Input data containing job description text.
    text_column : str, default='description'
        Column name to extract when data is a DataFrame.

    Returns
    -------
    pd.Series or str
        Cleaned job description text.
    """
    from src.preprocessing.text_cleaner import clean_text

    if isinstance(data, str):
        return clean_text(data)

    if isinstance(data, pd.DataFrame):
        if text_column in data.columns:
            series = data[text_column]
        elif "job_description" in data.columns:
            series = data["job_description"]
        elif "description" in data.columns:
            series = data["description"]
        elif "combined_text" in data.columns:
            series = data["combined_text"]
        else:
            raise KeyError(
                f"Required column '{text_column}' not found in DataFrame. Available columns: {list(data.columns)}"
            )

        if data.empty:
            return pd.Series(dtype=object)

        return series.fillna("").astype(str).apply(clean_text)

    if isinstance(data, pd.Series):
        if data.empty:
            return pd.Series(dtype=object)
        return data.fillna("").astype(str).apply(clean_text)

    if isinstance(data, (list, tuple)):
        return [clean_text(item) for item in data]

    raise TypeError(
        f"Unsupported data type for prepare_job_description_text: {type(data)}. Expected DataFrame, Series, str, or list."
    )