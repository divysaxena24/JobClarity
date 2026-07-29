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