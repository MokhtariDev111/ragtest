"""
embedding_layer/embedding_benchmark.py
========================================
Benchmarks embedding models: timing, vector dimensions, and a retrieval accuracy
test on a small held-out query set.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from embedding_layer.embedder import get_embedder, BaseEmbedder


class EmbeddingBenchmarker:
    """
    Benchmarks embedding generation speed, dimensionality, and retrieval accuracy.

    Parameters
    ----------
    embedding_config : list[dict]
        Each item: {"name": "bge", "model_id": "BAAI/...", "enabled": True}
    device           : 'cpu' or 'cuda'
    batch_size       : encoding batch size
    """

    def __init__(
        self,
        embedding_config: list[dict],
        device: str = "cpu",
        batch_size: int = 32,
    ):
        self.embedding_config = embedding_config
        self.device = device
        self.batch_size = batch_size
        self.results: list[dict] = []

    def run(
        self,
        corpus: list[str],
        queries: list[str],
        relevant_indices: Optional[list[list[int]]] = None,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Benchmark each enabled embedding model.

        Parameters
        ----------
        corpus           : list of text chunks (the index)
        queries          : list of query strings to test retrieval
        relevant_indices : for each query, list of relevant corpus indices.
                           If None, retrieval accuracy is not computed.
        top_k            : top-k chunks to retrieve per query
        """
        self.results = []
        for cfg in self.embedding_config:
            if not cfg.get("enabled", True):
                continue
            name = cfg["name"]
            try:
                embedder = get_embedder(name, device=self.device, batch_size=self.batch_size)
            except Exception as exc:
                logger.warning(f"Could not load embedder '{name}': {exc}")
                continue

            logger.info(f"Benchmarking embedder: {name}")
            record = self._benchmark_model(
                embedder=embedder,
                name=name,
                corpus=corpus,
                queries=queries,
                relevant_indices=relevant_indices,
                top_k=top_k,
            )
            self.results.append(record)

        return self.results

    def _benchmark_model(
        self,
        embedder: BaseEmbedder,
        name: str,
        corpus: list[str],
        queries: list[str],
        relevant_indices: Optional[list[list[int]]],
        top_k: int,
    ) -> dict:
        # ── Corpus embedding time ──────────────────────────────────────────────
        t0 = time.perf_counter()
        corpus_vecs = embedder.embed(corpus)
        corpus_time_s = round(time.perf_counter() - t0, 4)

        # ── Query embedding time ──────────────────────────────────────────────
        t0 = time.perf_counter()
        query_vecs = embedder.embed(queries)
        query_time_s = round(time.perf_counter() - t0, 4)

        dim = corpus_vecs.shape[1] if corpus_vecs.ndim == 2 else None

        # ── Retrieval accuracy ────────────────────────────────────────────────
        precision_at_k = None
        recall_at_k = None
        if relevant_indices:
            precision_at_k, recall_at_k = _compute_retrieval_metrics(
                corpus_vecs, query_vecs, relevant_indices, top_k
            )

        return {
            "embedder": name,
            "model_id": embedder.model_id,
            "vector_dimension": dim,
            "corpus_size": len(corpus),
            "corpus_embed_time_s": corpus_time_s,
            "corpus_embed_per_doc_ms": round(corpus_time_s * 1000 / max(len(corpus), 1), 2),
            "query_embed_time_s": query_time_s,
            "query_embed_per_query_ms": round(query_time_s * 1000 / max(len(queries), 1), 2),
            f"precision_at_{top_k}": precision_at_k,
            f"recall_at_{top_k}": recall_at_k,
        }

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.results)


def _compute_retrieval_metrics(
    corpus_vecs: np.ndarray,
    query_vecs: np.ndarray,
    relevant_indices: list[list[int]],
    top_k: int,
) -> tuple[float, float]:
    """Compute mean P@k and R@k using cosine similarity."""
    precisions, recalls = [], []
    sim_matrix = query_vecs @ corpus_vecs.T  # shape: (Q, C)

    for i, rel_ids in enumerate(relevant_indices):
        if not rel_ids:
            continue
        ranked = np.argsort(-sim_matrix[i])[:top_k].tolist()
        hits = len(set(ranked) & set(rel_ids))
        precisions.append(hits / top_k)
        recalls.append(hits / len(rel_ids))

    p_at_k = round(float(np.mean(precisions)), 4) if precisions else None
    r_at_k = round(float(np.mean(recalls)), 4) if recalls else None
    return p_at_k, r_at_k
