"""
retrieval_system/retrievers.py
================================
Retrieval strategies for the RAG benchmark.

Implemented strategies:
  - BasicRetriever       – pure vector similarity search
  - RerankerRetriever    – vector search + cross-encoder reranking
  - HybridRetriever      – BM25 + vector score fusion
  - MultiQueryRetriever  – multi-paraphrase query expansion

All strategies share a BaseRetriever interface:
    retrieve(query: str, top_k: int) -> list[str]
"""

from __future__ import annotations

import abc
from typing import Optional

import numpy as np
from loguru import logger

from embedding_layer.embedder import BaseEmbedder
from vector_database.vector_store import BaseVectorStore


# ── Base Interface ─────────────────────────────────────────────────────────────

class BaseRetriever(abc.ABC):
    """Abstract base class for all retrieval strategies."""

    name: str = "base"

    @abc.abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """Return the top-k relevant text chunks for *query*."""


# ── Basic Retriever ────────────────────────────────────────────────────────────

class BasicRetriever(BaseRetriever):
    """Pure vector similarity search."""

    name = "basic"

    def __init__(self, embedder: BaseEmbedder, store: BaseVectorStore):
        self.embedder = embedder
        self.store = store

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        vec = self.embedder.embed_single(query)
        return self.store.query(vec, top_k=top_k)


# ── Reranker Retriever ─────────────────────────────────────────────────────────

class RerankerRetriever(BaseRetriever):
    """
    Two-stage retrieval:
      1. Retrieve top (top_k * multiplier) candidates via vector search.
      2. Re-score with a cross-encoder and return top_k.
    """

    name = "reranking"

    def __init__(
        self,
        embedder: BaseEmbedder,
        store: BaseVectorStore,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        candidate_multiplier: int = 3,
    ):
        self.embedder = embedder
        self.store = store
        self.reranker_model = reranker_model
        self.candidate_multiplier = candidate_multiplier
        self._cross_encoder = None

    def _get_cross_encoder(self):
        if self._cross_encoder is None:
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"Loading cross-encoder: {self.reranker_model}")
                self._cross_encoder = CrossEncoder(self.reranker_model)
            except ImportError:
                logger.error("sentence-transformers not installed")
                return None
        return self._cross_encoder

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        cross_encoder = self._get_cross_encoder()
        candidates_k = top_k * self.candidate_multiplier
        vec = self.embedder.embed_single(query)
        candidates = self.store.query(vec, top_k=candidates_k)

        if not candidates or not cross_encoder:
            return candidates[:top_k]

        pairs = [[query, c] for c in candidates]
        scores = cross_encoder.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [c for c, _ in ranked[:top_k]]


# ── Hybrid Retriever ───────────────────────────────────────────────────────────

class HybridRetriever(BaseRetriever):
    """
    Hybrid BM25 + dense vector retrieval with linear score fusion.
    BM25 and vector scores are normalised to [0,1] then combined via weights.
    """

    name = "hybrid"

    def __init__(
        self,
        embedder: BaseEmbedder,
        store: BaseVectorStore,
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6,
    ):
        self.embedder = embedder
        self.store = store
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self._corpus_chunks: list[str] = []
        self._bm25 = None

    def set_corpus(self, chunks: list[str]) -> None:
        """Build the BM25 index from the indexed chunks."""
        self._corpus_chunks = chunks
        try:
            from rank_bm25 import BM25Okapi
            tokenised = [c.lower().split() for c in chunks]
            self._bm25 = BM25Okapi(tokenised)
            logger.debug(f"BM25 index built over {len(chunks)} chunks")
        except ImportError:
            logger.error("rank_bm25 not installed. Run: pip install rank_bm25")

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        vec = self.embedder.embed_single(query)

        # Vector scores from store (approximate - sorted by cosine sim)
        vector_results = self.store.query(vec, top_k=len(self._corpus_chunks) or top_k * 5)

        if not self._bm25 or not self._corpus_chunks:
            return vector_results[:top_k]

        # BM25 scores over full corpus
        bm25_scores = self._bm25.get_scores(query.lower().split())
        bm25_norm = _normalize(bm25_scores)

        # Convert vector results to score dict
        vec_score_dict: dict[str, float] = {}
        for rank, text in enumerate(vector_results):
            # Assign decaying score based on rank
            vec_score_dict[text] = 1.0 - (rank / max(len(vector_results), 1))

        # Fuse scores
        chunk_scores: dict[int, float] = {}
        for idx, chunk in enumerate(self._corpus_chunks):
            v_score = vec_score_dict.get(chunk, 0.0)
            b_score = float(bm25_norm[idx])
            chunk_scores[idx] = (
                self.vector_weight * v_score + self.bm25_weight * b_score
            )

        ranked = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        return [self._corpus_chunks[i] for i, _ in ranked[:top_k]]


# ── Multi-Query Retriever ──────────────────────────────────────────────────────

class MultiQueryRetriever(BaseRetriever):
    """
    Generates N paraphrase queries using an LLM, retrieves chunks for each,
    and returns deduplicated union (sorted by frequency of appearance).
    """

    name = "multi_query"

    def __init__(
        self,
        embedder: BaseEmbedder,
        store: BaseVectorStore,
        llm_generate_fn,         # callable(prompt: str) -> str
        num_queries: int = 3,
    ):
        self.embedder = embedder
        self.store = store
        self.llm_generate_fn = llm_generate_fn
        self.num_queries = num_queries

    def _generate_paraphrases(self, query: str) -> list[str]:
        prompt = (
            f"Generate {self.num_queries} different paraphrases of the following question. "
            f"Output ONLY the questions, one per line, no numbering:\n\n{query}"
        )
        try:
            raw = self.llm_generate_fn(prompt)
            lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
            paraphrases = lines[: self.num_queries]
            return [query] + paraphrases
        except Exception as exc:
            logger.warning(f"Paraphrase generation failed: {exc}")
            return [query]

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        queries = self._generate_paraphrases(query)
        seen: dict[str, int] = {}

        for q in queries:
            vec = self.embedder.embed_single(q)
            results = self.store.query(vec, top_k=top_k)
            for chunk in results:
                seen[chunk] = seen.get(chunk, 0) + 1

        # Sort by frequency (chunks appearing across more queries rank higher)
        ranked = sorted(seen.items(), key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in ranked[:top_k]]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _normalize(scores: np.ndarray) -> np.ndarray:
    """Min-max normalise a score array to [0, 1]."""
    mn, mx = scores.min(), scores.max()
    if mx == mn:
        return np.zeros_like(scores, dtype=float)
    return (scores - mn) / (mx - mn)


# ── Factory ────────────────────────────────────────────────────────────────────

_RETRIEVER_REGISTRY: dict[str, type[BaseRetriever]] = {
    "basic":       BasicRetriever,
    "reranking":   RerankerRetriever,
    "hybrid":      HybridRetriever,
    "multi_query": MultiQueryRetriever,
}


def get_retriever(
    name: str,
    embedder: BaseEmbedder,
    store: BaseVectorStore,
    **kwargs,
) -> BaseRetriever:
    """
    Instantiate a retriever by name.

    Parameters
    ----------
    name     : 'basic' | 'reranking' | 'hybrid' | 'multi_query'
    embedder : an initialised BaseEmbedder
    store    : an initialised BaseVectorStore (already indexed)
    **kwargs : forwarded to the retriever constructor
    """
    name = name.lower()
    if name not in _RETRIEVER_REGISTRY:
        raise ValueError(
            f"Unknown retriever '{name}'. Available: {list(_RETRIEVER_REGISTRY.keys())}"
        )
    return _RETRIEVER_REGISTRY[name](embedder=embedder, store=store, **kwargs)
