"""
data_ingestion/ocr_pipeline.py
================================
Strategy-pattern OCR engine wrappers.

Each engine implements the BaseOCREngine interface:
    extract_text(image: PIL.Image.Image) -> str

Supported engines:
  - Tesseract (via pytesseract)
  - EasyOCR
  - PaddleOCR
  - DocTR
"""

from __future__ import annotations

import abc
import time
from typing import Optional

from loguru import logger
from PIL import Image


# ── Base Interface ─────────────────────────────────────────────────────────────

class BaseOCREngine(abc.ABC):
    """Abstract base class for all OCR engines."""

    name: str = "base"

    @abc.abstractmethod
    def extract_text(self, image: Image.Image, lang: str = "en") -> str:
        """
        Extract text from a PIL Image.

        Parameters
        ----------
        image : PIL.Image.Image
        lang  : language hint ('en' or 'fr')

        Returns
        -------
        str – extracted text
        """

    def extract_text_timed(self, image: Image.Image, lang: str = "en") -> dict:
        """Run extract_text and also return wall-clock time."""
        t0 = time.perf_counter()
        text = self.extract_text(image, lang)
        elapsed = time.perf_counter() - t0
        return {"text": text, "latency_s": round(elapsed, 4)}


# ── Tesseract ──────────────────────────────────────────────────────────────────

class TesseractOCR(BaseOCREngine):
    """OCR using the classic Tesseract LSTM engine via pytesseract."""

    name = "tesseract"

    def __init__(self, config: str = "--psm 6"):
        self.config = config

    def extract_text(self, image: Image.Image, lang: str = "en") -> str:
        try:
            import pytesseract
            tess_lang = "fra" if lang == "fr" else "eng"
            text = pytesseract.image_to_string(image, lang=tess_lang, config=self.config)
            return text.strip()
        except ImportError:
            logger.error("pytesseract not installed. Run: pip install pytesseract")
            return ""
        except Exception as exc:
            logger.error(f"Tesseract error: {exc}")
            return ""


# ── EasyOCR ────────────────────────────────────────────────────────────────────

class EasyOCROCR(BaseOCREngine):
    """OCR using EasyOCR (deep-learning based, multi-language)."""

    name = "easyocr"

    def __init__(self, gpu: bool = False):
        self.gpu = gpu
        self._reader: Optional[object] = None

    def _get_reader(self, langs: list[str]):
        """Lazy-initialise the EasyOCR reader."""
        try:
            import easyocr
            if self._reader is None:
                logger.info(f"Initialising EasyOCR reader (langs={langs}, gpu={self.gpu})")
                self._reader = easyocr.Reader(langs, gpu=self.gpu)
            return self._reader
        except ImportError:
            logger.error("easyocr not installed. Run: pip install easyocr")
            return None

    def extract_text(self, image: Image.Image, lang: str = "en") -> str:
        langs = ["en", "fr"] if lang == "fr" else ["en"]
        reader = self._get_reader(langs)
        if reader is None:
            return ""
        try:
            import numpy as np
            img_array = np.array(image)
            results = reader.readtext(img_array, detail=0)
            return "\n".join(results)
        except Exception as exc:
            logger.error(f"EasyOCR error: {exc}")
            return ""


# ── PaddleOCR ──────────────────────────────────────────────────────────────────

class PaddleOCROCR(BaseOCREngine):
    """OCR using PaddleOCR (Baidu's high-accuracy pipeline)."""

    name = "paddleocr"

    def __init__(
        self,
        use_angle_cls: bool = True,
        use_gpu: bool = False,
    ):
        self.use_angle_cls = use_angle_cls
        self.use_gpu = use_gpu
        self._ocr = None

    def _get_ocr(self, lang: str):
        try:
            from paddleocr import PaddleOCR
            paddle_lang = "french" if lang == "fr" else "en"
            if self._ocr is None:
                logger.info("Initialising PaddleOCR")
                self._ocr = PaddleOCR(
                    use_angle_cls=self.use_angle_cls,
                    lang=paddle_lang,
                    use_gpu=self.use_gpu,
                    show_log=False,
                )
            return self._ocr
        except ImportError:
            logger.error("paddleocr not installed. Run: pip install paddleocr paddlepaddle")
            return None

    def extract_text(self, image: Image.Image, lang: str = "en") -> str:
        ocr = self._get_ocr(lang)
        if ocr is None:
            return ""
        try:
            import numpy as np
            img_array = np.array(image)
            result = ocr.ocr(img_array, cls=True)
            lines = []
            if result and result[0]:
                for line in result[0]:
                    lines.append(line[1][0])
            return "\n".join(lines)
        except Exception as exc:
            logger.error(f"PaddleOCR error: {exc}")
            return ""


# ── DocTR ──────────────────────────────────────────────────────────────────────

class DocTROCR(BaseOCREngine):
    """OCR using DocTR (Mindee's transformer-based document OCR)."""

    name = "doctr"

    def __init__(self, pretrained: bool = True):
        self.pretrained = pretrained
        self._model = None

    def _get_model(self):
        try:
            from doctr.models import ocr_predictor
            if self._model is None:
                logger.info("Initialising DocTR OCR predictor (pretrained=True)")
                self._model = ocr_predictor(pretrained=self.pretrained)
            return self._model
        except ImportError:
            logger.error("python-doctr not installed. Run: pip install python-doctr[torch]")
            return None

    def extract_text(self, image: Image.Image, lang: str = "en") -> str:
        model = self._get_model()
        if model is None:
            return ""
        try:
            import numpy as np
            from doctr.io import DocumentFile
            img_array = np.array(image)
            doc = DocumentFile.from_images([img_array])
            result = model(doc)
            lines = []
            for page in result.pages:
                for block in page.blocks:
                    for line in block.lines:
                        lines.append(" ".join(w.value for w in line.words))
            return "\n".join(lines)
        except Exception as exc:
            logger.error(f"DocTR error: {exc}")
            return ""


# ── Factory ────────────────────────────────────────────────────────────────────

_ENGINE_REGISTRY: dict[str, type[BaseOCREngine]] = {
    "tesseract": TesseractOCR,
    "easyocr":   EasyOCROCR,
    "paddleocr": PaddleOCROCR,
    "doctr":     DocTROCR,
}


def get_ocr_engine(name: str, **kwargs) -> BaseOCREngine:
    """
    Factory function to instantiate an OCR engine by name.

    Parameters
    ----------
    name : str  – one of 'tesseract', 'easyocr', 'paddleocr', 'doctr'
    **kwargs    – forwarded to the engine constructor

    Returns
    -------
    BaseOCREngine instance
    """
    name = name.lower()
    if name not in _ENGINE_REGISTRY:
        raise ValueError(
            f"Unknown OCR engine '{name}'. "
            f"Available: {list(_ENGINE_REGISTRY.keys())}"
        )
    return _ENGINE_REGISTRY[name](**kwargs)


def register_ocr_engine(name: str, cls: type[BaseOCREngine]) -> None:
    """Register a custom OCR engine class under *name*."""
    _ENGINE_REGISTRY[name] = cls
    logger.info(f"Registered custom OCR engine: {name}")
