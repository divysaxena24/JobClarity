"""
Suspicious language and structural text feature extractor for JobClarity.

Extracts domain-informed fraud indicators and stylistic anomalies directly
from raw and cleaned job descriptions, preserving 100% training-serving parity.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.preprocessing.text_cleaner import clean_text

# Regex patterns for lexical scam categories
RE_URGENCY = re.compile(
    r"\b(urgent|urgently|immediate|immediately|immediate start|start immediately|"
    r"hiring immediately|act now|apply now|don't wait|dont wait|hurry|fast cash|"
    r"start today|start asap|immediate vacancy)\b",
    re.IGNORECASE,
)

RE_FINANCIAL_PROMISE = re.compile(
    r"\b(wire transfer|western union|moneygram|money gram|cryptocurrency|crypto|"
    r"bitcoin|passive income|residual income|earn daily|earn weekly|earn per day|"
    r"earn per hour|cash bonus|guaranteed income|unlimited earnings|weekly payout|"
    r"daily payout|quick cash|extra income|financial freedom|paid weekly|paid daily)\b",
    re.IGNORECASE,
)

RE_LOW_BARRIER = re.compile(
    r"\b(no experience|no prior experience|no resume|no cv|no interview|"
    r"no qualification|no qualifications|work from home|work from sofa|"
    r"work from anywhere|any age|easy job|easy cash|entry level no experience|"
    r"no skills needed|no background check)\b",
    re.IGNORECASE,
)

RE_LOGISTICS_SCAM = re.compile(
    r"\b(package reship|package forwarding|reshipping|re-shipping|mystery shopper|"
    r"secret shopper|courier commission|envelope stuffing|re-ship|re ship|"
    r"package processing|parcel forwarding)\b",
    re.IGNORECASE,
)

RE_CONTACT_EVASION = re.compile(
    r"\b(whatsapp|telegram|viber|skype|hangouts|signal|direct message|dm me|"
    r"text me|send sms|contact me on whatsapp|contact via telegram)\b",
    re.IGNORECASE,
)

RE_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
RE_PHONE = re.compile(r"\+?\d[\d\s().-]{7,}\d")
RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_CURRENCY = re.compile(r"[$€£¥₹]")


class SuspiciousLanguageExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts 18 domain-specific fraud indicators and structural text features.

    Features extracted:
    1.  urgency_score: match count of high-pressure urgency keywords
    2.  financial_promise_score: match count of unrealistic earning/wire cues
    3.  low_barrier_score: match count of no-experience/easy-money promises
    4.  logistics_scam_score: match count of parcel reshipping/forwarding cues
    5.  contact_evasion_score: match count of out-of-band messaging app handles
    6.  has_email: binary indicator of email presence in raw text
    7.  has_phone: binary indicator of phone number pattern in raw text
    8.  has_url: binary indicator of URL presence in raw text
    9.  currency_symbol_count: count of currency symbols ($, €, £, etc.)
    10. uppercase_ratio: proportion of uppercase characters among letters
    11. exclamation_count: frequency of exclamation marks
    12. question_count: frequency of question marks
    13. digit_ratio: proportion of digits among total characters
    14. char_count: total raw character length
    15. log_char_count: log1p(char_count)
    16. word_count: total words in cleaned text
    17. log_word_count: log1p(word_count)
    18. avg_word_length: average characters per word
    """

    FEATURE_NAMES: list[str] = [
        "urgency_score",
        "financial_promise_score",
        "low_barrier_score",
        "logistics_scam_score",
        "contact_evasion_score",
        "has_email",
        "has_phone",
        "has_url",
        "currency_symbol_count",
        "uppercase_ratio",
        "exclamation_count",
        "question_count",
        "digit_ratio",
        "char_count",
        "log_char_count",
        "word_count",
        "log_word_count",
        "avg_word_length",
    ]

    def __init__(self) -> None:
        pass

    def fit(self, X: Any, y: Any = None) -> SuspiciousLanguageExtractor:
        """Fit transformer (stateless, validates input)."""
        return self

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return names of generated features."""
        return np.array(self.FEATURE_NAMES, dtype=object)

    def _extract_from_single_text(self, raw_text: str) -> list[float]:
        """Extract the 18 numeric features from a single raw text string."""
        if not raw_text or not isinstance(raw_text, str):
            return [0.0] * len(self.FEATURE_NAMES)

        text_len = len(raw_text)
        cleaned = clean_text(raw_text)
        words = cleaned.split()
        word_count = len(words)

        # Lexical pattern scores
        urgency = len(RE_URGENCY.findall(raw_text))
        financial = len(RE_FINANCIAL_PROMISE.findall(raw_text))
        low_barrier = len(RE_LOW_BARRIER.findall(raw_text))
        logistics = len(RE_LOGISTICS_SCAM.findall(raw_text))
        contact = len(RE_CONTACT_EVASION.findall(raw_text))

        # Surface markers (before HTML stripping / normalisation)
        has_email = 1.0 if RE_EMAIL.search(raw_text) else 0.0
        has_phone = 1.0 if RE_PHONE.search(raw_text) else 0.0
        has_url = 1.0 if RE_URL.search(raw_text) else 0.0
        currency_count = float(len(RE_CURRENCY.findall(raw_text)))

        # Character distributions
        alphas = sum(1 for c in raw_text if c.isalpha())
        uppers = sum(1 for c in raw_text if c.isupper())
        upper_ratio = (uppers / alphas) if alphas > 0 else 0.0

        exclamations = float(raw_text.count("!"))
        questions = float(raw_text.count("?"))
        digits = sum(1 for c in raw_text if c.isdigit())
        digit_ratio = (digits / text_len) if text_len > 0 else 0.0

        # Structural length metrics
        char_count = float(text_len)
        log_char = float(np.log1p(char_count))
        w_count = float(word_count)
        log_word = float(np.log1p(w_count))
        avg_word_len = (char_count / max(w_count, 1.0)) if w_count > 0 else 0.0

        return [
            float(urgency),
            float(financial),
            float(low_barrier),
            float(logistics),
            float(contact),
            has_email,
            has_phone,
            has_url,
            currency_count,
            round(upper_ratio, 4),
            exclamations,
            questions,
            round(digit_ratio, 4),
            char_count,
            round(log_char, 4),
            w_count,
            round(log_word, 4),
            round(avg_word_len, 4),
        ]

    def transform(self, X: Any) -> np.ndarray:
        """
        Transform texts into an (N, 18) float32 numpy array.

        Parameters
        ----------
        X : str, list of str, pd.Series, or pd.DataFrame
            Input texts or DataFrame containing 'description'.

        Returns
        -------
        np.ndarray of shape (N, 18)
        """
        if isinstance(X, str):
            rows = [self._extract_from_single_text(X)]
            return np.array(rows, dtype=np.float32)

        if isinstance(X, pd.DataFrame):
            col = "description" if "description" in X.columns else ("job_description" if "job_description" in X.columns else X.columns[0])
            series = X[col].fillna("").astype(str).tolist()
        elif isinstance(X, pd.Series):
            series = X.fillna("").astype(str).tolist()
        elif isinstance(X, (list, tuple, np.ndarray)):
            series = [str(item) if item is not None else "" for item in X]
        else:
            raise TypeError(f"Unsupported input type for SuspiciousLanguageExtractor: {type(X)}")

        rows = [self._extract_from_single_text(t) for t in series]
        return np.array(rows, dtype=np.float32)
