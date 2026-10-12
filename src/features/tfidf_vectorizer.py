"""
TF-IDF feature engineering utilities.
"""

from sklearn.feature_extraction.text import TfidfVectorizer


def create_tfidf_vectorizer(
    max_features: int = 10000,
    ngram_range: tuple = (1, 2),
):
    """
    Create a TF-IDF Vectorizer.

    Parameters
    ----------
    max_features : int
        Maximum vocabulary size.

    ngram_range : tuple
        N-gram range.

    Returns
    -------
    TfidfVectorizer
    """

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words="english"
    )

    return vectorizer


def fit_tfidf_vectorizer(
    vectorizer: TfidfVectorizer,
    train_text,
):
    """
    Fit TF-IDF vectorizer exclusively on training text data.
    Ensures zero data leakage from validation or test splits.

    Parameters
    ----------
    vectorizer : TfidfVectorizer
        Configured TF-IDF vectorizer instance.
    train_text : Iterable[str]
        Training corpus exclusively. Never pass test/validation text.

    Returns
    -------
    tuple
        (fitted_vectorizer, X_train)
    """
    X_train = vectorizer.fit_transform(train_text)
    return vectorizer, X_train


def transform_text(
    vectorizer: TfidfVectorizer,
    text,
):
    """
    Transform evaluation or test text using an already-fitted vectorizer.
    Guarantees no re-fitting or vocabulary leakage occurs.

    Parameters
    ----------
    vectorizer : TfidfVectorizer
        Fitted TF-IDF vectorizer.
    text : Iterable[str]
        Evaluation, test, or inference text.

    Returns
    -------
    scipy.sparse matrix
        Transformed TF-IDF feature matrix.
    """
    return vectorizer.transform(text)