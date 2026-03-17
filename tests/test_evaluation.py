"""
tests/test_evaluation.py
==========================
Unit tests for retrieval and RAG evaluation metrics.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from evaluation_framework.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    mrr,
    ndcg_at_k,
    compute_retrieval_metrics,
)
from evaluation_framework.rag_metrics import (
    compute_faithfulness,
    compute_answer_relevancy,
    compute_context_precision,
    compute_context_recall,
)


RETRIEVED = ["chunk_a", "chunk_b", "chunk_c", "chunk_d", "chunk_e"]
RELEVANT  = ["chunk_a", "chunk_c"]


class TestRetrievalMetrics:
    def test_precision_at_k_perfect(self):
        p = precision_at_k(["a", "b"], ["a", "b"], k=2)
        assert p == 1.0

    def test_precision_at_k_zero(self):
        p = precision_at_k(["x", "y"], ["a", "b"], k=2)
        assert p == 0.0

    def test_precision_at_k_partial(self):
        p = precision_at_k(RETRIEVED, RELEVANT, k=5)
        assert 0.0 < p < 1.0

    def test_recall_at_k_perfect(self):
        r = recall_at_k(["a", "b"], ["a", "b"], k=2)
        assert r == 1.0

    def test_recall_at_k_zero(self):
        r = recall_at_k(["x", "y"], ["a", "b"], k=2)
        assert r == 0.0

    def test_mrr_first_relevant(self):
        score = mrr(["a", "b", "c"], ["a"])
        assert score == 1.0

    def test_mrr_second_relevant(self):
        score = mrr(["x", "a", "c"], ["a"])
        assert abs(score - 0.5) < 1e-6

    def test_mrr_not_found(self):
        score = mrr(["x", "y", "z"], ["a"])
        assert score == 0.0

    def test_ndcg_perfect(self):
        score = ndcg_at_k(["a", "b"], ["a", "b"], k=2)
        assert abs(score - 1.0) < 1e-6

    def test_compute_retrieval_metrics_batch(self):
        result = compute_retrieval_metrics(
            retrieved_list=[RETRIEVED, RETRIEVED],
            relevant_list=[RELEVANT, RELEVANT],
            k=5,
        )
        assert "precision_at_5" in result
        assert "recall_at_5" in result
        assert "mrr" in result
        assert "ndcg_at_5" in result
        assert result["num_queries"] == 2


class TestRAGMetrics:
    def test_faithfulness_heuristic_high(self):
        answer = "retrieval augmented generation combines retrieval with generation"
        context = ["retrieval augmented generation is a technique that combines retrieval with generation to produce answers"]
        score = compute_faithfulness(answer, context)
        assert score > 0.5

    def test_faithfulness_heuristic_low(self):
        answer = "bananas are yellow fruits grown in tropical regions"
        context = ["retrieval augmented generation is a machine learning technique"]
        score = compute_faithfulness(answer, context)
        assert score < 0.5

    def test_faithfulness_empty_answer(self):
        score = compute_faithfulness("", ["some context"])
        assert score == 0.0

    def test_answer_relevancy_heuristic_high(self):
        score = compute_answer_relevancy(
            question="What is RAG?",
            answer="RAG stands for Retrieval Augmented Generation.",
        )
        assert score > 0.3

    def test_context_precision(self):
        retrieved = ["relevant chunk", "irrelevant chunk"]
        relevant = ["relevant chunk"]
        score = compute_context_precision(retrieved, relevant)
        assert score > 0.0 and score <= 1.0

    def test_context_recall_full(self):
        retrieved = ["chunk_a", "chunk_b"]
        relevant = ["chunk_a"]
        score = compute_context_recall(retrieved, relevant)
        assert score == 1.0

    def test_context_recall_zero(self):
        retrieved = ["chunk_x", "chunk_y"]
        relevant = ["chunk_a"]
        score = compute_context_recall(retrieved, relevant)
        assert score == 0.0
