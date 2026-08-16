"""Helper functions"""

import re


def strip_punctuation(text: str) -> str:
    """Match punctuation at beginning or end of string and remove."""
    pattern = r"^[^\w\s]+|[^\w\s]+$"
    return re.sub(pattern, "", text)


def exact_match_tokenize(text: str) -> list[str]:
    """Tokenize for exact match splits on white space and strips punctuation
    at beginning or end.
    """
    tokens = [
        strip_punctuation(token).lower() for token in text.split()
    ]
    return tokens

