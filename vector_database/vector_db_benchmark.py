"""
vector_database/vector_db_benchmark.py
=========================================
Benchmarks vector stores: indexing time, query latency, memory usage,
and retrieval accuracy (P@k, R@k).
"""

from __future__ import annotations

import gc
import time
from typing import Optional

import numpy as np
import pandas as pd
import psutil
from loguru import logger

from vector_database.vector_store import get_vector_store, BaseVectorStore


class VectorDBBenchmarker:
    """
    Benchmark multiple vector databases on the same corpus.

    Parameters
    ----------
    db_config : list[dict]
        Each item: {"name": "faiss", "enabled": True, ...}
    """

    def __init__(self, db_config: list[dict]):
        self.db_config = db_config
        self.results: list[dict] = []

    def run(
        self,
        chunks: list[str],
        embeddings: np.ndarray,
        query_embeddings: np.ndarray,
        relevant_indices: Optional[list[list[int]]] = None,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Benchmark each enabled vector store.

        Parameters
        ----------
        chunks           : pre-computed text chunks
        embeddings       : (N, D) embeddings for chunks
        query_embeddings : (Q, D) embeddings for test queries
        relevant_indices : ground-truth relevant chunk indices per query
        top_k            : number of results to retrieve per query
        """
        self.results = []
        for cfg in self.db_config:
            if not cfg.get("enabled", True):
                continue
            name = cfg["name"]
            kwargs = {k: v for k, v in cfg.items() if k not in ("name", "enabled")}
            try:
                store = get_vector_store(name, **kwargs)
            except Exception as exc:
                logger.warning(f"Could not load vector store '{name}': {exc}")
                continue

            logger.info(f"Benchmarking vector DB: {name}")
            record = self._benchmark_store(
                store=store,
                name=name,
                chunks=chunks,
                embeddings=embeddings,
                query_embeddings=query_embeddings,
                relevant_indices=relevant_indices,
                top_k=top_k,
            )
            self.results.append(record)
        return self.results

    def _benchmark_store(
        self,
        store: BaseVectorStore,
        name: str,
        chunks: list[str],
        embeddings: np.ndarray,
        query_embeddings: np.ndarray,
        relevant_indices: Optional[list[list[int]]],
        top_k: int,
    ) -> dict:
        proc = psutil.Process()

        # ── Index ──────────────────────────────────────────────────────────────
        mem_before = proc.memory_info().rss / (1024 * 1024)
        t0 = time.perf_counter()
        store.index(chunks, embeddings)
        index_time_s = round(time.perf_counter() - t0, 4)
        gc.collect()
        mem_after = proc.memory_info().rss / (1024 * 1024)
        mem_mb = round(max(mem_after - mem_before, 0), 2)

        # ── Query latency ──────────────────────────────────────────────────────
        latencies = []
        retrieved = []
        for qvec in query_embeddings:
            t0 = time.perf_counter()
            results = store.query(qvec, top_k=top_k)
            latencies.append(time.perf_counter() - t0)
            retrieved.append(results)

        mean_latency_ms = round(float(np.mean(latencies)) * 1000, 3) if latencies else None
        p99_latency_ms = round(float(np.percentile(latencies, 99)) * 1000, 3) if latencies else None

        # ── Retrieval accuracy ─────────────────────────────────────────────────
        precision_at_k = None
        recall_at_k = None
        if relevant_indices:
            precisions, recalls = [], []
            for i, rel_ids in enumerate(relevant_indices):
                if not rel_ids or i >= len(retrieved):
                    continue
                retrieved_texts = retrieved[i]
                # Map text → index for accuracy calculation
                chunk_index = {c: idx for idx, c in enumerate(chunks)}
                retrieved_ids = [chunk_index.get(t, -1) for t in retrieved_texts]
                hits = len(set(retrieved_ids) & set(rel_ids))
                precisions.append(hits / top_k)
                recalls.append(hits / len(rel_ids))
            if precisions:
                precision_at_k = round(float(np.mean(precisions)), 4)
                recall_at_k = round(float(np.mean(recalls)), 4)

        store.clear()

        return {
            "vector_db": name,
            "corpus_size": len(chunks),
            "index_time_s": index_time_s,
            "memory_mb": mem_mb,
            "mean_query_latency_ms": mean_latency_ms,
            "p99_query_latency_ms": p99_latency_ms,
            f"precision_at_{top_k}": precision_at_k,
            f"recall_at_{top_k}": recall_at_k,
        }

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.results)
