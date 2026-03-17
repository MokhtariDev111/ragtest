"""
text_processing/text_cleaner.py
=================================
Normalises and cleans raw text extracted from documents.

Techniques applied:
  - Unicode normalisation (NFC)
  - Remove control characters
  - Collapse repeated whitespace / blank lines
  - Strip leading/trailing whitespace per line
"""

from __future__ import annotations

import re
import unicodedata
from loguru import logger


class TextCleaner:
    """
    Configurable text cleaner.

    Parameters
    ----------
    normalize_unicode      : apply NFC Unicode normalisation (default True)
    remove_control_chars   : strip non-printable control characters (default True)
    remove_extra_whitespace: collapse multiple spaces / blank lines (default True)
    """

    def __init__(
        self,
        normalize_unicode: bool = True,
        remove_control_chars: bool = True,
        remove_extra_whitespace: bool = True,
    ):
        self.normalize_unicode = normalize_unicode
        self.remove_control_chars = remove_control_chars
        self.remove_extra_whitespace = remove_extra_whitespace

    def clean(self, text: str) -> str:
        """
        Apply all enabled cleaning steps to *text* and return the result.
        """
        if not text:
            return ""

        if self.normalize_unicode:
            text = unicodedata.normalize("NFC", text)

        if self.remove_control_chars:
            text = self._strip_control_chars(text)

        if self.remove_extra_whitespace:
            text = self._collapse_whitespace(text)

        return text.strip()

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _strip_control_chars(text: str) -> str:
        """
        Remove control characters (category Cc) except for newline (\\n),
        carriage return (\\r), and tab (\\t).
        """
        cleaned = []
        for ch in text:
            cat = unicodedata.category(ch)
            if cat == "Cc" and ch not in ("\n", "\r", "\t"):
                continue
            cleaned.append(ch)
        return "".join(cleaned)

    @staticmethod
    def _collapse_whitespace(text: str) -> str:
        """Collapse multiple spaces and 3+ consecutive blank lines."""
        # Replace multiple spaces / tabs on a single line with a single space
        text = re.sub(r"[ \t]+", " ", text)
        # Strip trailing whitespace from each line
        text = "\n".join(line.rstrip() for line in text.splitlines())
        # Collapse 3 or more consecutive blank lines into 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text


def clean_text_with_config(text: str, config: dict) -> str:
    """
    Convenience factory: create a TextCleaner from a config dict and apply it.

    Config dict keys (all optional, default True):
        normalize_unicode, remove_extra_whitespace, remove_control_chars
    """
    cleaner = TextCleaner(
        normalize_unicode=config.get("normalize_unicode", True),
        remove_control_chars=config.get("remove_control_chars", True),
        remove_extra_whitespace=config.get("remove_extra_whitespace", True),
    )
    result = cleaner.clean(text)
    logger.debug(f"Text cleaned: {len(text)} → {len(result)} chars")
    return result
