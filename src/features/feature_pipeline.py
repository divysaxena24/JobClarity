"""
Modular feature engineering pipelines for JobClarity experiments.

Supports:
  1. Word-level TF-IDF (baseline and tuned).
  2. Character-level TF-IDF and Word+Char FeatureUnion.
  3. Domain-specific suspicious language indicators.
  4. Composite text + metadata pipelines with backward-compatible schema handling.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import hstack, issparse, csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from src.features.suspicious_language import SuspiciousLanguageExtractor
from src.preprocessing.text_cleaner import clean_text


class CleanTextTransformer(BaseEstimator, TransformerMixin):
    """Transformer that applies clean_text to raw text inputs."""

    def fit(self, X: Any, y: Any = None) -> CleanTextTransformer:
        return self

    def transform(self, X: Any) -> list[str]:
        if isinstance(X, str):
            return [clean_text(X)]
        if isinstance(X, pd.DataFrame):
            col = "description" if "description" in X.columns else ("job_description" if "job_description" in X.columns else X.columns[0])
            return [clean_text(t) for t in X[col].fillna("").astype(str)]
        if isinstance(X, pd.Series):
            return [clean_text(t) for t in X.fillna("").astype(str)]
        return [clean_text(str(t)) for t in X]


def create_word_tfidf_vectorizer(
    max_features: int = 10000,
    ngram_range: tuple[int, int] = (1, 2),
    sublinear_tf: bool = True,
    min_df: int = 1,
) -> TfidfVectorizer:
    """Create a word-level TF-IDF vectorizer."""
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
        min_df=min_df,
        stop_words="english",
    )


def create_char_tfidf_vectorizer(
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (3, 5),
    sublinear_tf: bool = True,
    min_df: int = 1,
) -> TfidfVectorizer:
    """Create a character n-gram (char_wb) TF-IDF vectorizer."""
    return TfidfVectorizer(
        max_features=max_features,
        analyzer="char_wb",
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
        min_df=min_df,
    )


def create_hybrid_tfidf_union(
    word_max_features: int = 8000,
    char_max_features: int = 4000,
    word_ngram_range: tuple[int, int] = (1, 2),
    char_ngram_range: tuple[int, int] = (3, 5),
    word_min_df: int = 1,
    char_min_df: int = 1,
) -> FeatureUnion:
    """
    Create a FeatureUnion combining word n-grams and character n-grams.
    """
    return FeatureUnion(
        [
            (
                "word_tfidf",
                create_word_tfidf_vectorizer(
                    max_features=word_max_features,
                    ngram_range=word_ngram_range,
                    sublinear_tf=True,
                    min_df=word_min_df,
                ),
            ),
            (
                "char_tfidf",
                create_char_tfidf_vectorizer(
                    max_features=char_max_features,
                    ngram_range=char_ngram_range,
                    sublinear_tf=True,
                    min_df=char_min_df,
                ),
            ),
        ]
    )


class MetadataFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts optional metadata features with backward-compatible defaults.

    When metadata fields are absent (e.g. at single-description inference time),
    imputes neutral values (-1 / missing indicator) so the model receives
    a deterministic, valid feature vector without raising an error.
    """

    METADATA_COLUMNS = [
        "has_company_logo",
        "has_questions",
        "telecommuting",
    ]
    EMPLOYMENT_TYPES = ["Full-time", "Part-time", "Contract", "Temporary", "Other"]

    def __init__(self) -> None:
        pass

    def fit(self, X: Any, y: Any = None) -> MetadataFeatureExtractor:
        return self

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        names = list(self.METADATA_COLUMNS)
        for emp in self.EMPLOYMENT_TYPES:
            names.append(f"emp_type_{emp.lower()}")
        names.append("emp_type_missing")
        return np.array(names, dtype=object)

    def transform(self, X: Any) -> np.ndarray:
        n_features = len(self.get_feature_names_out())

        # If single text string or list of strings (inference without metadata):
        if isinstance(X, (str, list, tuple)) and (
            not isinstance(X, pd.DataFrame)
        ):
            n_rows = 1 if isinstance(X, str) else len(X)
            # Default: logo=-1 (unknown), questions=-1 (unknown), telecommuting=-1 (unknown),
            # emp_types=0, emp_type_missing=1
            rows = []
            for _ in range(n_rows):
                row = [-1.0, -1.0, -1.0] + [0.0] * len(self.EMPLOYMENT_TYPES) + [1.0]
                rows.append(row)
            return np.array(rows, dtype=np.float32)

        if isinstance(X, pd.DataFrame):
            n_rows = len(X)
            rows = []
            for _, r in X.iterrows():
                # Binary flags
                logo = float(r["has_company_logo"]) if "has_company_logo" in r and pd.notna(r["has_company_logo"]) else -1.0
                questions = float(r["has_questions"]) if "has_questions" in r and pd.notna(r["has_questions"]) else -1.0
                tele = float(r["telecommuting"]) if "telecommuting" in r and pd.notna(r["telecommuting"]) else -1.0

                # Employment type one-hot
                emp_val = str(r["employment_type"]).strip() if "employment_type" in r and pd.notna(r["employment_type"]) else ""
                emp_one_hot = []
                matched = False
                for emp in self.EMPLOYMENT_TYPES:
                    if emp_val.lower() == emp.lower():
                        emp_one_hot.append(1.0)
                        matched = True
                    else:
                        emp_one_hot.append(0.0)
                emp_missing = 0.0 if matched else 1.0

                rows.append([logo, questions, tele] + emp_one_hot + [emp_missing])

            return np.array(rows, dtype=np.float32)

        # Fallback
        return np.full((len(X), n_features), -1.0, dtype=np.float32)


class CompositeFeaturePipeline:
    """
    Unified composite pipeline combining TF-IDF, suspicious language features,
    and optional metadata features.
    """

    def __init__(
        self,
        tfidf_model: Any,
        use_suspicious_features: bool = True,
        use_metadata_features: bool = False,
    ) -> None:
        self.tfidf_model = tfidf_model
        self.use_suspicious_features = use_suspicious_features
        self.use_metadata_features = use_metadata_features
        self.suspicious_extractor = (
            SuspiciousLanguageExtractor() if use_suspicious_features else None
        )
        self.metadata_extractor = (
            MetadataFeatureExtractor() if use_metadata_features else None
        )

    def fit(self, df_or_texts: Any, y: Any = None) -> CompositeFeaturePipeline:
        """Fit all transformers on training data only."""
        if isinstance(df_or_texts, pd.DataFrame):
            col = "description" if "description" in df_or_texts.columns else ("job_description" if "job_description" in df_or_texts.columns else df_or_texts.columns[0])
            raw_texts = df_or_texts[col].fillna("").astype(str).tolist()
        elif isinstance(df_or_texts, pd.Series):
            raw_texts = df_or_texts.fillna("").astype(str).tolist()
        elif isinstance(df_or_texts, (list, tuple)):
            raw_texts = [str(t) for t in df_or_texts]
        else:
            raw_texts = [str(df_or_texts)]

        cleaned_texts = [clean_text(t) for t in raw_texts]
        self.tfidf_model.fit(cleaned_texts)

        if self.suspicious_extractor is not None:
            self.suspicious_extractor.fit(raw_texts)

        if self.metadata_extractor is not None:
            self.metadata_extractor.fit(df_or_texts)

        return self

    def transform(self, df_or_texts: Any) -> Any:
        """Transform inputs into a concatenated sparse feature matrix."""
        if isinstance(df_or_texts, pd.DataFrame):
            col = "description" if "description" in df_or_texts.columns else ("job_description" if "job_description" in df_or_texts.columns else df_or_texts.columns[0])
            raw_texts = df_or_texts[col].fillna("").astype(str).tolist()
        elif isinstance(df_or_texts, pd.Series):
            raw_texts = df_or_texts.fillna("").astype(str).tolist()
        elif isinstance(df_or_texts, (list, tuple)):
            raw_texts = [str(t) for t in df_or_texts]
        else:
            raw_texts = [str(df_or_texts)]

        cleaned_texts = [clean_text(t) for t in raw_texts]
        X_tfidf = self.tfidf_model.transform(cleaned_texts)

        blocks = [X_tfidf]

        if self.suspicious_extractor is not None:
            X_suspicious = csr_matrix(self.suspicious_extractor.transform(raw_texts))
            blocks.append(X_suspicious)

        if self.metadata_extractor is not None:
            X_meta = csr_matrix(self.metadata_extractor.transform(df_or_texts))
            blocks.append(X_meta)

        return hstack(blocks, format="csr")

    def fit_transform(self, df_or_texts: Any, y: Any = None) -> Any:
        """Fit and transform in one call efficiently without duplicate text cleaning."""
        if isinstance(df_or_texts, pd.DataFrame):
            col = "description" if "description" in df_or_texts.columns else ("job_description" if "job_description" in df_or_texts.columns else df_or_texts.columns[0])
            raw_texts = df_or_texts[col].fillna("").astype(str).tolist()
        elif isinstance(df_or_texts, pd.Series):
            raw_texts = df_or_texts.fillna("").astype(str).tolist()
        elif isinstance(df_or_texts, (list, tuple)):
            raw_texts = [str(t) for t in df_or_texts]
        else:
            raw_texts = [str(df_or_texts)]

        cleaned_texts = [clean_text(t) for t in raw_texts]
        X_tfidf = self.tfidf_model.fit_transform(cleaned_texts)

        blocks = [X_tfidf]

        if self.suspicious_extractor is not None:
            self.suspicious_extractor.fit(raw_texts)
            X_suspicious = csr_matrix(self.suspicious_extractor.transform(raw_texts))
            blocks.append(X_suspicious)

        if self.metadata_extractor is not None:
            self.metadata_extractor.fit(df_or_texts)
            X_meta = csr_matrix(self.metadata_extractor.transform(df_or_texts))
            blocks.append(X_meta)

        return hstack(blocks, format="csr")
