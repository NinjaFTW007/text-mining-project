"""
preprocessing.py
────────────────────────────────────────────────────────────────────
Shared NLP text preprocessing utilities used by all three modules.

Pipeline per document:
  1. Lowercase
  2. Strip URLs / emails
  3. Remove digits
  4. Remove punctuation
  5. Remove English stopwords
  6. Drop single-character tokens

Author : Project Team (Niranjan · Vibhu · Gowtham)
"""

import re
import string
import nltk
from nltk.corpus import stopwords


# ──────────────────────────────────────────────────────────────────
#  Bootstrap NLTK corpora  (downloaded once, silently)
# ──────────────────────────────────────────────────────────────────

def _bootstrap_nltk() -> None:
    """Download required NLTK resources if not already present."""
    _resources = {
        "stopwords": "corpora/stopwords",
        "punkt":     "tokenizers/punkt",
        "wordnet":   "corpora/wordnet",
    }
    for pkg, path in _resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


_bootstrap_nltk()

# Build the English stopword set once at import time
STOPWORDS_EN: set[str] = set(stopwords.words("english"))


# ──────────────────────────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Apply the full preprocessing pipeline to a raw text string.

    Parameters
    ----------
    text : str
        Raw input string (review, SMS, etc.)

    Returns
    -------
    str
        Cleaned, space-joined token string ready for TF-IDF.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs and email addresses
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text, flags=re.MULTILINE)
    text = re.sub(r"\S+@\S+", " ", text)

    # 3. Remove digits
    text = re.sub(r"\d+", " ", text)

    # 4. Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # 5. Tokenise, remove stopwords, drop very short tokens
    tokens = [
        tok for tok in text.split()
        if tok not in STOPWORDS_EN and len(tok) > 1
    ]

    return " ".join(tokens)


def word_count(text: str) -> int:
    """Return the number of whitespace-separated words in *text*."""
    return len(text.strip().split()) if text.strip() else 0


def truncate(text: str, max_chars: int = 60) -> str:
    """Truncate text to *max_chars* and append ellipsis if needed."""
    text = str(text)
    return text if len(text) <= max_chars else text[:max_chars] + "…"
