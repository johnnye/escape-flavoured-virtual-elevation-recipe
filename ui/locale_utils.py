"""Locale-aware parsing utilities for the UI."""


def parse_float(text):
    """Parse a float from text, handling both '.' and ',' as decimal separators"""
    if not text or text.strip() == "":
        return None

    # Replace comma with dot for German locales
    normalized_text = text.replace(",", ".")

    # Check for multiple decimal points (invalid)
    if normalized_text.count(".") > 1:
        return None

    try:
        return float(normalized_text)
    except ValueError:
        return None


def parse_int(text):
    """Parse an integer from text, handling empty strings gracefully"""
    if not text or text.strip() == "":
        return None

    try:
        return int(text)
    except ValueError:
        return None
