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