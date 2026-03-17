"""
evaluation_framework/retrieval_metrics.py
==========================================
Standard IR retrieval evaluation metrics:
  - Precision@K
  - Recall@K
  - Mean Reciprocal Rank (MRR)
  - NDCG@K
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


def precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """
    Fraction of top-k retrieved chunks that are relevant.

    Parameters
    ----------
    retrieved : ordered list of retrieved chunk texts
    relevant  : set of ground-truth relevant chunk texts
    k         : cutoff
    """
    top_k = retrieved[:k]
    relevant_set = set(relevant)
    hits = sum(1 for c in top_k if c in relevant_set)
    return hits / k if k > 0 else 0.0


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """
    Fraction of all relevant chunks that appear in the top-k results.
    """
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    relevant_set = set(relevant)
    hits = sum(1 for c in top_k if c in relevant_set)
    return hits / len(relevant_set)


def mrr(retrieved: list[str], relevant: list[str]) -> float:
    """
    Mean Reciprocal Rank for a single query.
    Returns 1/rank of the first relevant chunk, or 0 if none found.
    """
    relevant_set = set(relevant)
    for rank, chunk in enumerate(retrieved, start=1):
        if chunk in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """
    Normalised Discounted Cumulative Gain at k.
    Assumes binary relevance (1 = relevant, 0 = not relevant).
    """
    relevant_set = set(relevant)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, chunk in enumerate(retrieved[:k], start=1)
        if chunk in relevant_set
    )
    ideal_len = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_len + 1))
    return dcg / idcg if idcg > 0 else 0.0


def compute_retrieval_metrics(
    retrieved_list: list[list[str]],
    relevant_list: list[list[str]],
    k: int = 5,
) -> dict:
    """
    Aggregate retrieval metrics over a batch of queries.

    Parameters
    ----------
    retrieved_list : list of retrieved chunk lists (one per query)
    relevant_list  : list of relevant chunk lists (one per query, same order)
    k              : cutoff for P@K, R@K, NDCG@K

    Returns
    -------
    dict with mean scores across all queries
    """
    p_scores, r_scores, mrr_scores, ndcg_scores = [], [], [], []

    for retrieved, relevant in zip(retrieved_list, relevant_list):
        p_scores.append(precision_at_k(retrieved, relevant, k))
        r_scores.append(recall_at_k(retrieved, relevant, k))
        mrr_scores.append(mrr(retrieved, relevant))
        ndcg_scores.append(ndcg_at_k(retrieved, relevant, k))

    return {
        f"precision_at_{k}": round(float(np.mean(p_scores)), 4) if p_scores else None,
        f"recall_at_{k}":    round(float(np.mean(r_scores)), 4) if r_scores else None,
        "mrr":              round(float(np.mean(mrr_scores)), 4) if mrr_scores else None,
        f"ndcg_at_{k}":     round(float(np.mean(ndcg_scores)), 4) if ndcg_scores else None,
        "num_queries":      len(retrieved_list),
    }
