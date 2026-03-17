"""
embedding_layer/embedder.py
============================
HuggingFace SentenceTransformer embedding interface.

Supported models (configurable):
  - BGE       : BAAI/bge-base-en-v1.5
  - E5        : intfloat/e5-base-v2
  - Instructor: hkunlp/instructor-base
  - MiniLM    : sentence-transformers/all-MiniLM-L6-v2

Common interface: embed(texts: list[str]) -> np.ndarray

Usage
-----
    from embedding_layer.embedder import get_embedder
    embedder = get_embedder("minilm")
    vectors = embedder.embed(["hello world", "another sentence"])
"""

from __future__ import annotations

import abc
import time
from typing import Optional

import numpy as np
from loguru import logger


# ── Base Interface ─────────────────────────────────────────────────────────────

class BaseEmbedder(abc.ABC):
    """Abstract base class for embedding models."""

    name: str = "base"
    model_id: str = ""
    dimension: Optional[int] = None

    @abc.abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Convert a list of text strings into a 2D numpy array of shape (N, D).
        """

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single string and return a 1D array."""
        return self.embed([text])[0]

    def embed_timed(self, texts: list[str]) -> dict:
        """Embed texts and also report total latency."""
        t0 = time.perf_counter()
        vectors = self.embed(texts)
        elapsed = time.perf_counter() - t0
        return {
            "vectors": vectors,
            "latency_s": round(elapsed, 4),
            "texts_per_second": round(len(texts) / elapsed, 2) if elapsed > 0 else None,
        }


# ── SentenceTransformer base ───────────────────────────────────────────────────

class SentenceTransformerEmbedder(BaseEmbedder):
    """Generic SentenceTransformer-backed embedder."""

    def __init__(self, model_id: str, device: str = "cpu", batch_size: int = 32, name: str = ""):
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self.name = name or model_id.split("/")[-1].lower()
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading embedding model: {self.model_id} (device={self.device})")
                self._model = SentenceTransformer(self.model_id, device=self.device)
                self.dimension = self._model.get_sentence_embedding_dimension()
                logger.info(f"Model loaded — dim={self.dimension}")
            except ImportError:
                logger.error("sentence-transformers not installed")
                raise

    def embed(self, texts: list[str]) -> np.ndarray:
        self._load()
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.array(vectors, dtype=np.float32)


# ── Instructor Embedder ────────────────────────────────────────────────────────

class InstructorEmbedder(BaseEmbedder):
    """
    Instructor-based embedder. Requires the InstructorEmbedding package.
    Each text is prepended with an instruction for better task alignment.
    """

    name = "instructor"
    model_id = "hkunlp/instructor-base"

    def __init__(
        self,
        model_id: str = "hkunlp/instructor-base",
        device: str = "cpu",
        batch_size: int = 16,
        instruction: str = "Represent the document for retrieval:",
    ):
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self.instruction = instruction
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from InstructorEmbedding import INSTRUCTOR
                logger.info(f"Loading Instructor model: {self.model_id}")
                self._model = INSTRUCTOR(self.model_id)
                self.dimension = 768  # base models
            except ImportError:
                logger.warning("InstructorEmbedding not installed, falling back to SentenceTransformer")
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_id, device=self.device)
                self.dimension = self._model.get_sentence_embedding_dimension()
                self._use_fallback = True

    def embed(self, texts: list[str]) -> np.ndarray:
        self._load()
        if getattr(self, "_use_fallback", False):
            return np.array(self._model.encode(texts, show_progress_bar=False), dtype=np.float32)
        pairs = [[self.instruction, t] for t in texts]
        vectors = self._model.encode(pairs, show_progress_bar=False)
        return np.array(vectors, dtype=np.float32)


# ── Pre-configured model factories ────────────────────────────────────────────

_MODEL_CONFIGS = {
    "bge": {
        "cls": SentenceTransformerEmbedder,
        "model_id": "BAAI/bge-base-en-v1.5",
        "name": "bge",
    },
    "e5": {
        "cls": SentenceTransformerEmbedder,
        "model_id": "intfloat/e5-base-v2",
        "name": "e5",
    },
    "instructor": {
        "cls": InstructorEmbedder,
        "model_id": "hkunlp/instructor-base",
        "name": "instructor",
    },
    "minilm": {
        "cls": SentenceTransformerEmbedder,
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "name": "minilm",
    },
}

_EMBEDDER_REGISTRY: dict[str, type[BaseEmbedder]] = {}


def get_embedder(
    name: str,
    device: str = "cpu",
    batch_size: int = 32,
    **kwargs,
) -> BaseEmbedder:
    """
    Instantiate an embedding model by short name.

    Parameters
    ----------
    name      : 'bge' | 'e5' | 'instructor' | 'minilm' | custom registered name
    device    : 'cpu' or 'cuda'
    batch_size: encoding batch size
    **kwargs  : forwarded to the embedder constructor
    """
    name = name.lower()

    # Check custom registry first
    if name in _EMBEDDER_REGISTRY:
        return _EMBEDDER_REGISTRY[name](device=device, batch_size=batch_size, **kwargs)

    if name not in _MODEL_CONFIGS:
        raise ValueError(
            f"Unknown embedder '{name}'. Available: {list(_MODEL_CONFIGS.keys())}"
        )
    cfg = _MODEL_CONFIGS[name]
    cls = cfg["cls"]
    return cls(
        model_id=cfg["model_id"],
        device=device,
        batch_size=batch_size,
        name=cfg["name"],
        **kwargs,
    )


def register_embedder(name: str, cls: type[BaseEmbedder]) -> None:
    """Register a custom embedder class under *name*."""
    _EMBEDDER_REGISTRY[name] = cls
    logger.info(f"Registered custom embedder: {name}")
