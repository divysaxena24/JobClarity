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

    df["combined_text"] = (
        df[text_columns]
        .fillna("")
        .agg(" ".join, axis=1)
    )

    return df