"""
Unit tests for text preprocessing functions in src/preprocessing/text_cleaner.py.
"""

import pytest
from src.preprocessing.text_cleaner import (
    clean_text,
    convert_to_lowercase,
    decode_html_entities,
    normalize_whitespace,
    remove_emails,
    remove_html,
    remove_phone_numbers,
    remove_special_characters,
    remove_urls,
)


@pytest.mark.unit
class TestPreprocessing:
    """Test suite covering text cleaning and normalization functions."""

    # --- clean_text pipeline tests ---

    def test_clean_text_none_input(self):
        """None input should return an empty string gracefully."""
        assert clean_text(None) == ""

    def test_clean_text_empty_string(self):
        """Empty string input should return an empty string."""
        assert clean_text("") == ""

    def test_clean_text_whitespace_only(self):
        """Whitespace-only input should be trimmed to empty string."""
        assert clean_text("   \t  \n  \r\n  ") == ""

    def test_clean_text_normal_job_description(self):
        """Standard job posting text should be normalized to lowercase alphanumeric."""
        input_text = "We are seeking a Senior Python Developer with 3+ years experience!"
        expected = "we are seeking a senior python developer with 3 years experience"
        assert clean_text(input_text) == expected

    def test_clean_text_deterministic(self):
        """Repeated runs on the same input must yield identical output."""
        sample = "<p>Join our team at https://work.io! Call 555-123-4567 or email hr@work.io.</p>"
        result1 = clean_text(sample)
        result2 = clean_text(sample)
        result3 = clean_text(sample)
        assert result1 == result2 == result3

    def test_clean_text_numeric_and_boolean_coercion(self):
        """Non-string inputs like numbers should be coerced to string and cleaned."""
        assert clean_text(12345) == "12345"
        assert clean_text(3.1415) == "3 1415"
        assert clean_text(True) == "true"

    def test_clean_text_unicode_and_emojis(self):
        """Non-ASCII characters and emojis should be replaced with spaces and stripped."""
        text_with_emojis = "Great Job Alert! 🚀 Earn 💰 now: Café & Résumé"
        cleaned = clean_text(text_with_emojis)
        # Accents and emojis are stripped by regex [^a-zA-Z0-9\s]
        assert "great job alert" in cleaned
        assert "earn" in cleaned
        assert "now" in cleaned
        assert "🚀" not in cleaned
        assert "💰" not in cleaned

    # --- Individual component function tests ---

    def test_remove_html(self):
        """HTML tags of various forms should be removed."""
        html_input = "<div><h1>Title</h1><p>Description with <a href='link'>link</a>.</p></div>"
        cleaned = remove_html(html_input)
        assert "<" not in cleaned
        assert ">" not in cleaned
        assert "TitleDescription with link." in cleaned or "Title" in cleaned

    def test_decode_html_entities(self):
        """HTML entities should be converted back to raw characters."""
        assert decode_html_entities("AT&amp;T &lt;corp&gt; &quot;jobs&quot;") == 'AT&T <corp> "jobs"'
        assert decode_html_entities("Tom &amp; Jerry") == "Tom & Jerry"

    @pytest.mark.parametrize(
        "url_text",
        [
            "Visit https://example.com/apply for info",
            "Go to http://sub.domain.org:8080/path?q=1 now",
            "Check www.careers-portal.com/job today",
            "https://t.co/xyz123 is the link",
        ],
    )
    def test_remove_urls(self, url_text):
        """All variations of URLs should be removed."""
        cleaned = remove_urls(url_text)
        assert "http" not in cleaned
        assert "www." not in cleaned
        assert "://" not in cleaned

    @pytest.mark.parametrize(
        "email_text",
        [
            "Send CV to hr@company.com immediately",
            "Contact john.doe+jobs@sub.domain.co.uk today",
            "Support at careers123@portal.io for help",
        ],
    )
    def test_remove_emails(self, email_text):
        """Email addresses should be scrubbed."""
        cleaned = remove_emails(email_text)
        assert "@" not in cleaned

    @pytest.mark.parametrize(
        "phone_text",
        [
            "Call +91 9876543210 for details",
            "Dial (123) 456-7890 now",
            "Phone: +1-234-567-8900 directly",
            "Hotline: 9876543210 available",
        ],
    )
    def test_remove_phone_numbers(self, phone_text):
        """Standard phone number formats should be removed."""
        cleaned = remove_phone_numbers(phone_text)
        assert "9876543210" not in cleaned
        assert "456-7890" not in cleaned

    def test_convert_to_lowercase(self):
        """All characters should be lowercased."""
        assert convert_to_lowercase("ALL CAPS AND MixedCase") == "all caps and mixedcase"

    def test_remove_special_characters(self):
        """Punctuation, symbols, and formatting marks should become whitespace."""
        raw = "Hello, World! $100% #remote (work) @home & more."
        cleaned = remove_special_characters(raw)
        assert "!" not in cleaned
        assert "$" not in cleaned
        assert "%" not in cleaned
        assert "#" not in cleaned
        assert "@" not in cleaned
        assert "(" not in cleaned
        assert ")" not in cleaned
        assert "&" not in cleaned
        assert "100" in cleaned
        assert "remote" in cleaned

    def test_normalize_whitespace(self):
        """Consecutive whitespace characters should be collapsed to single spaces and trimmed."""
        assert normalize_whitespace("   too    many   \t  spaces  \n  here  ") == "too many spaces here"
        assert normalize_whitespace("") == ""
        assert normalize_whitespace("single") == "single"
