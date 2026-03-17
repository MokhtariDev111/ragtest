"""
data_ingestion/document_loader.py
==================================
Unified document loader supporting PDF, TXT, and image files.
Returns either raw extracted text (for PDF/TXT) or PIL Images (for image files)
so they can be passed to the OCR pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Generator

from loguru import logger
from PIL import Image


# ── Supported file extensions ──────────────────────────────────────────────────
PDF_EXTENSIONS  = {".pdf"}
TEXT_EXTENSIONS = {".txt", ".md", ".rst", ".csv"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}


class DocumentLoader:
    """
    Loads documents from a directory or a single file path.

    Usage
    -----
    loader = DocumentLoader()
    for doc in loader.load_directory("data/documents"):
        print(doc["type"], doc["source"], doc["content"][:100])
    """

    def load_file(self, file_path: str | Path) -> dict:
        """
        Load a single file and return a document dict.

        Returns
        -------
        dict with keys:
            source   (str)  – original file path
            type     (str)  – 'text' | 'image'
            content  (str | PIL.Image.Image)
            language (str)  – detected language hint ('en' or 'fr')
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        ext = path.suffix.lower()
        logger.debug(f"Loading file: {path} (ext={ext})")

        if ext in PDF_EXTENSIONS:
            return self._load_pdf(path)
        elif ext in TEXT_EXTENSIONS:
            return self._load_text(path)
        elif ext in IMAGE_EXTENSIONS:
            return self._load_image(path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def load_directory(
        self,
        directory: str | Path,
        recursive: bool = True,
    ) -> Generator[dict, None, None]:
        """
        Yield document dicts for all supported files in *directory*.
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        pattern = "**/*" if recursive else "*"
        for file_path in sorted(directory.glob(pattern)):
            if not file_path.is_file():
                continue
            ext = file_path.suffix.lower()
            if ext in PDF_EXTENSIONS | TEXT_EXTENSIONS | IMAGE_EXTENSIONS:
                try:
                    yield self.load_file(file_path)
                except Exception as exc:
                    logger.warning(f"Skipping {file_path}: {exc}")

    # ── Private helpers ────────────────────────────────────────────────────────

    def _load_pdf(self, path: Path) -> dict:
        """Extract text from a PDF using PyMuPDF (fitz)."""
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(str(path))
            text = "\n".join(page.get_text() for page in doc)
            doc.close()
            logger.info(f"PDF loaded: {path.name} ({len(text)} chars)")
            return {
                "source": str(path),
                "type": "text",
                "content": text,
                "language": _detect_language_hint(text),
            }
        except ImportError:
            logger.warning("PyMuPDF not installed, falling back to pdfplumber")
            return self._load_pdf_pdfplumber(path)

    def _load_pdf_pdfplumber(self, path: Path) -> dict:
        """Fallback PDF loader using pdfplumber."""
        import pdfplumber
        with pdfplumber.open(str(path)) as pdf:
            text = "\n".join(
                (page.extract_text() or "") for page in pdf.pages
            )
        logger.info(f"PDF (pdfplumber) loaded: {path.name} ({len(text)} chars)")
        return {
            "source": str(path),
            "type": "text",
            "content": text,
            "language": _detect_language_hint(text),
        }

    def _load_text(self, path: Path) -> dict:
        """Load a plain-text file."""
        text = path.read_text(encoding="utf-8", errors="replace")
        logger.info(f"Text file loaded: {path.name} ({len(text)} chars)")
        return {
            "source": str(path),
            "type": "text",
            "content": text,
            "language": _detect_language_hint(text),
        }

    def _load_image(self, path: Path) -> dict:
        """Load an image file as a PIL Image for OCR processing."""
        image = Image.open(str(path)).convert("RGB")
        logger.info(f"Image loaded: {path.name} ({image.width}x{image.height})")
        return {
            "source": str(path),
            "type": "image",
            "content": image,
            "language": "en",  # will be overridden by OCR config
        }


def _detect_language_hint(text: str) -> str:
    """
    Heuristic language detection based on common French stopwords.
    Returns 'fr' if French cues are detected, else 'en'.
    """
    fr_words = {"le", "la", "les", "de", "du", "des", "est", "une", "pour",
                "dans", "avec", "sur", "par", "sont", "aussi", "comme"}
    words = set(w.lower() for w in text.split()[:200])
    overlap = words & fr_words
    return "fr" if len(overlap) >= 4 else "en"
