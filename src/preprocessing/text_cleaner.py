"""
Text preprocessing utilities for JobClarity.

Reusable text cleaning functions for training and inference.
"""

from bs4 import BeautifulSoup


def remove_html(text: str) -> str:
    """
    Remove HTML tags from text.
    """
    return BeautifulSoup(text, "html.parser").get_text()


def clean_text(text: str) -> str:
    """
    Main preprocessing pipeline.
    """

    if text is None:
        return ""

    text = str(text)

    text = remove_html(text)
    text = decode_html_entities(text)
    text = remove_urls(text)
    text = remove_emails(text)
    text = remove_phone_numbers(text)
    text = convert_to_lowercase(text)
    text = remove_special_characters(text)
    text = normalize_whitespace(text)

    return text



def decode_html_entities(text: str) -> str:
    """
    Convert HTML entities into normal text.

    Example:
        AT&amp;T -> AT&T
    """

    return html.unescape(text)


def remove_urls(text: str) -> str:
    """
    Remove URLs from text.

    Examples:
        https://abc.com
        http://abc.com
        www.abc.com
    """

    url_pattern = r"https?://\S+|www\.\S+"

    return re.sub(url_pattern, "", text)


def remove_emails(text: str) -> str:
    """
    Remove email addresses from text.

    Examples:
        hr@company.com
        jobs@gmail.com
    """

    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"

    return re.sub(email_pattern, "", text)


def remove_phone_numbers(text: str) -> str:
    """
    Remove phone numbers from text.

    Examples:
        +91 9876543210
        9876543210
        +1-234-567-8900
        (123) 456-7890
    """

    phone_pattern = r"\+?\d[\d\s().-]{7,}\d"

    return re.sub(phone_pattern, "", text)


def convert_to_lowercase(text: str) -> str:
    """
    Convert text to lowercase.
    """

    return text.lower()

def remove_special_characters(text: str) -> str:
    """
    Remove special characters while keeping
    alphabets, numbers and spaces.
    """

    return re.sub(r"[^a-zA-Z0-9\s]", " ", text)


def normalize_whitespace(text: str) -> str:
    """
    Remove extra spaces from text.
    """

    return re.sub(r"\s+", " ", text).strip()