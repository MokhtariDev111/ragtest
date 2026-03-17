"""
text_processing/chunker.py
============================
Strategy-pattern text chunkers supporting:
  - FixedChunker      – fixed size with optional overlap
  - RecursiveChunker  – LangChain RecursiveCharacterTextSplitter
  - SlidingWindowChunker – overlapping window
  - SemanticChunker   – groups sentences by embedding cosine similarity

Each strategy implements BaseChunker with a single `split(text) -> list[str]` method.

Usage
-----
    from text_processing.chunker import get_chunker
    chunker = get_chunker("fixed", chunk_size=512, chunk_overlap=50)
    chunks = chunker.split(text)
"""

from __future__ import annotations

import abc
import re
from typing import Optional

import numpy as np
from loguru import logger


# ── Base Interface ─────────────────────────────────────────────────────────────

class BaseChunker(abc.ABC):
    """Abstract base class for all chunking strategies."""

    name: str = "base"

    @abc.abstractmethod
    def split(self, text: str) -> list[str]:
        """
        Split *text* into a list of non-empty string chunks.
        """

    def split_batch(self, texts: list[str]) -> list[list[str]]:
        """Apply split() to each text in a batch."""
        return [self.split(t) for t in texts]


# ── Fixed Chunker ──────────────────────────────────────────────────────────────

class FixedChunker(BaseChunker):
    """
    Split text into chunks of exactly `chunk_size` characters,
    with `overlap` characters shared between consecutive chunks.
    """

    name = "fixed"

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> list[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end].strip())
            if end >= len(text):
                break
            start = end - self.chunk_overlap
        return [c for c in chunks if c]


# ── Recursive Chunker ──────────────────────────────────────────────────────────

class RecursiveChunker(BaseChunker):
    """
    Splits text using a hierarchy of separators (paragraph → newline → space),
    mimicking LangChain's RecursiveCharacterTextSplitter behaviour.
    """

    name = "recursive"

    _SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> list[str]:
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=self._SEPARATORS,
            )
            return [c.strip() for c in splitter.split_text(text) if c.strip()]
        except ImportError:
            logger.warning("langchain_text_splitters not available, using fallback")
            return self._fallback_split(text)

    def _fallback_split(self, text: str) -> list[str]:
        """Pure-Python recursive split as fallback."""
        return self._recursive_split(text, self._SEPARATORS)

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []
        if not separators:
            return FixedChunker(self.chunk_size, self.chunk_overlap).split(text)

        sep = separators[0]
        if sep == "":
            return FixedChunker(self.chunk_size, self.chunk_overlap).split(text)

        parts = text.split(sep)
        chunks = []
        cur = ""
        for part in parts:
            candidate = cur + sep + part if cur else part
            if len(candidate) <= self.chunk_size:
                cur = candidate
            else:
                if cur:
                    chunks.append(cur.strip())
                cur = part
        if cur:
            chunks.append(cur.strip())

        result = []
        for chunk in chunks:
            if len(chunk) > self.chunk_size:
                result.extend(
                    self._recursive_split(chunk, separators[1:])
                )
            elif chunk:
                result.append(chunk)
        return result


# ── Sliding Window Chunker ─────────────────────────────────────────────────────

class SlidingWindowChunker(BaseChunker):
    """
    Creates overlapping chunks using a sliding window.
    Window moves by `stride` characters each step.
    """

    name = "sliding_window"

    def __init__(self, chunk_size: int = 512, stride: int = 256):
        self.chunk_size = chunk_size
        self.stride = stride

    def split(self, text: str) -> list[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
            start += self.stride
        return chunks


# ── Semantic Chunker ───────────────────────────────────────────────────────────

class SemanticChunker(BaseChunker):
    """
    Groups consecutive sentences into chunks based on cosine similarity of
    their embeddings. A new chunk is started when similarity drops below
    `similarity_threshold`.
    """

    name = "semantic"

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: float = 0.5,
        max_chunk_size: int = 1024,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size
        self.device = device
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"[SemanticChunker] Loading model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name, device=self.device)
            except ImportError:
                logger.error("sentence-transformers not installed")
                return None
        return self._model

    def split(self, text: str) -> list[str]:
        model = self._get_model()
        sentences = _split_sentences(text)

        if not model or len(sentences) <= 1:
            # Fallback to fixed chunker
            return FixedChunker(max(self.max_chunk_size, 512)).split(text)

        embeddings = model.encode(sentences, show_progress_bar=False)

        chunks = []
        current_sents = [sentences[0]]

        for i in range(1, len(sentences)):
            sim = _cosine_sim(embeddings[i - 1], embeddings[i])
            candidate = " ".join(current_sents + [sentences[i]])
            if sim >= self.similarity_threshold and len(candidate) <= self.max_chunk_size:
                current_sents.append(sentences[i])
            else:
                chunks.append(" ".join(current_sents))
                current_sents = [sentences[i]]

        if current_sents:
            chunks.append(" ".join(current_sents))

        return [c.strip() for c in chunks if c.strip()]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """Simple sentence splitter using punctuation."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ── Registry & Factory ─────────────────────────────────────────────────────────

_CHUNKER_REGISTRY: dict[str, type[BaseChunker]] = {
    "fixed":          FixedChunker,
    "recursive":      RecursiveChunker,
    "sliding_window": SlidingWindowChunker,
    "semantic":       SemanticChunker,
}


def get_chunker(name: str, chunk_size: int = 512, chunk_overlap: int = 50, **kwargs) -> BaseChunker:
    """
    Factory function to instantiate a chunker by name.

    Parameters
    ----------
    name         : strategy name ('fixed' | 'recursive' | 'sliding_window' | 'semantic')
    chunk_size   : target chunk size in characters
    chunk_overlap: overlap / stride in characters
    **kwargs     : forwarded to the chunker constructor
    """
    name = name.lower()
    if name not in _CHUNKER_REGISTRY:
        raise ValueError(
            f"Unknown chunker '{name}'. Available: {list(_CHUNKER_REGISTRY.keys())}"
        )
    cls = _CHUNKER_REGISTRY[name]
    if name == "sliding_window":
        return cls(chunk_size=chunk_size, stride=chunk_overlap, **kwargs)
    return cls(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)


def register_chunker(name: str, cls: type[BaseChunker]) -> None:
    """Register a custom chunker class under *name*."""
    _CHUNKER_REGISTRY[name] = cls
