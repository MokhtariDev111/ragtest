"""
evaluation_framework/rag_metrics.py
=====================================
RAG-specific evaluation metrics:
  - Faithfulness
  - Answer Relevancy
  - Context Precision
  - Context Recall

These metrics use either:
  (a) LLM-as-a-judge (for faithfulness, answer relevancy)
  (b) Semantic similarity via sentence embeddings (answer relevancy fallback)
"""

from __future__ import annotations

import re
from typing import Optional, Callable

import numpy as np
from loguru import logger


# ── Faithfulness ───────────────────────────────────────────────────────────────

def compute_faithfulness(
    answer: str,
    context_chunks: list[str],
    llm_judge: Optional[Callable[[str], str]] = None,
) -> float:
    """
    Measure whether the answer claims are grounded in the retrieved context.

    Uses LLM-as-a-judge if *llm_judge* is provided, otherwise falls back to
    a simple word-overlap heuristic.

    Returns a score in [0, 1] where 1.0 = fully faithful.
    """
    if not answer or not context_chunks:
        return 0.0

    if llm_judge:
        return _llm_faithfulness(answer, context_chunks, llm_judge)
    return _heuristic_faithfulness(answer, context_chunks)


def _llm_faithfulness(
    answer: str,
    context_chunks: list[str],
    llm_judge: Callable[[str], str],
) -> float:
    """Ask the LLM to rate faithfulness on a 0–5 scale."""
    context = "\n\n".join(context_chunks[:3])[:2000]
    prompt = (
        "Rate how faithful the following ANSWER is to the CONTEXT.\n"
        "Score from 0 (completely hallucinated) to 5 (fully grounded).\n"
        "Output ONLY the integer score.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"ANSWER:\n{answer}\n\n"
        "SCORE:"
    )
    try:
        raw = llm_judge(prompt).strip()
        match = re.search(r"\d", raw)
        score = int(match.group()) if match else 0
        return min(max(score, 0), 5) / 5.0
    except Exception as exc:
        logger.warning(f"Faithfulness LLM judge failed: {exc}")
        return _heuristic_faithfulness(answer, context_chunks)


def _heuristic_faithfulness(answer: str, context_chunks: list[str]) -> float:
    """Simple word-overlap heuristic as fallback."""
    context = " ".join(context_chunks).lower()
    answer_words = set(re.findall(r"\b\w{4,}\b", answer.lower()))
    context_words = set(re.findall(r"\b\w{4,}\b", context))
    if not answer_words:
        return 0.0
    overlap = answer_words & context_words
    return round(len(overlap) / len(answer_words), 4)


# ── Answer Relevancy ───────────────────────────────────────────────────────────

def compute_answer_relevancy(
    question: str,
    answer: str,
    embedding_fn: Optional[Callable[[list[str]], np.ndarray]] = None,
    llm_judge: Optional[Callable[[str], str]] = None,
) -> float:
    """
    Measure semantic relevancy between the generated answer and the question.

    Priority: LLM judge → embedding cosine similarity → word overlap heuristic.

    Returns a score in [0, 1].
    """
    if not answer:
        return 0.0

    if llm_judge:
        return _llm_answer_relevancy(question, answer, llm_judge)
    if embedding_fn:
        return _embedding_answer_relevancy(question, answer, embedding_fn)
    return _heuristic_answer_relevancy(question, answer)


def _llm_answer_relevancy(
    question: str,
    answer: str,
    llm_judge: Callable[[str], str],
) -> float:
    prompt = (
        "Rate how relevant the ANSWER is to the QUESTION.\n"
        "Score from 0 (completely off-topic) to 5 (perfectly relevant).\n"
        "Output ONLY the integer score.\n\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER: {answer}\n\n"
        "SCORE:"
    )
    try:
        raw = llm_judge(prompt).strip()
        match = re.search(r"\d", raw)
        score = int(match.group()) if match else 0
        return min(max(score, 0), 5) / 5.0
    except Exception as exc:
        logger.warning(f"Answer relevancy LLM judge failed: {exc}")
        return _heuristic_answer_relevancy(question, answer)


def _embedding_answer_relevancy(
    question: str,
    answer: str,
    embedding_fn: Callable[[list[str]], np.ndarray],
) -> float:
    vecs = embedding_fn([question, answer])
    q_vec, a_vec = vecs[0], vecs[1]
    norm_q = np.linalg.norm(q_vec)
    norm_a = np.linalg.norm(a_vec)
    if norm_q == 0 or norm_a == 0:
        return 0.0
    return float(round(np.dot(q_vec, a_vec) / (norm_q * norm_a), 4))


def _heuristic_answer_relevancy(question: str, answer: str) -> float:
    q_words = set(re.findall(r"\b\w{3,}\b", question.lower()))
    a_words = set(re.findall(r"\b\w{3,}\b", answer.lower()))
    if not q_words:
        return 0.0
    overlap = q_words & a_words
    return round(len(overlap) / len(q_words), 4)


# ── Context Precision ──────────────────────────────────────────────────────────

def compute_context_precision(
    question: str,
    retrieved_chunks: list[str],
    relevant_chunks: Optional[list[str]] = None,
    llm_judge: Optional[Callable[[str], str]] = None,
) -> float:
    """
    Measure the signal-to-noise ratio in the retrieved context.
    
    If llm_judge is provided, it asks the LLM to identify relevant chunks.
    Otherwise, it uses heuristic matching against relevant_chunks if provided.
    """
    if not retrieved_chunks:
        return 0.0

    if llm_judge:
        return _llm_context_precision(question, retrieved_chunks, llm_judge)

    if not relevant_chunks:
        return 0.0
        
    relevant_set = set(relevant_chunks)
    hits = sum(1 for c in retrieved_chunks if c in relevant_set or
               any(_chunk_overlap(c, r) > 0.7 for r in relevant_set))
    return round(hits / len(retrieved_chunks), 4)


def _llm_context_precision(
    question: str,
    retrieved_chunks: list[str],
    llm_judge: Callable[[str], str],
) -> float:
    """Ask LLM to judge relevance of each chunk."""
    hits = 0
    # Judge only top 3 for speed in live chat
    to_judge = retrieved_chunks[:3]
    for chunk in to_judge:
        prompt = (
            "Determine if the following CHUNK is relevant to answering the QUESTION.\n"
            "Output 'YES' or 'NO'.\n\n"
            f"QUESTION: {question}\n"
            f"CHUNK: {chunk[:500]}\n"
            "RELEVANT:"
        )
        try:
            res = llm_judge(prompt).strip().upper()
            if "YES" in res:
                hits += 1
        except Exception:
            pass
    return round(hits / len(to_judge), 4) if to_judge else 0.0


# ── Context Recall ─────────────────────────────────────────────────────────────

def compute_context_recall(
    expected_answer: str,
    retrieved_chunks: list[str],
    relevant_chunks: Optional[list[str]] = None,
) -> float:
    """
    Measure if the retrieved context contains the information in the expected answer.
    """
    if not retrieved_chunks:
        return 0.0
        
    if relevant_chunks:
        hits = sum(
            1 for r in relevant_chunks
            if r in retrieved_chunks or
               any(_chunk_overlap(r, c) > 0.7 for c in retrieved_chunks)
        )
        return round(hits / len(relevant_chunks), 4)

    # Heuristic: Check if keywords from expected answer are in chunks
    if not expected_answer:
        return 0.0
        
    context = " ".join(retrieved_chunks).lower()
    # Extract noun-like words (longer than 4 chars)
    words = set(re.findall(r"\b\w{5,}\b", expected_answer.lower()))
    if not words:
        return 0.0
    found = sum(1 for w in words if w in context)
    return round(found / len(words), 4)


def _chunk_overlap(a: str, b: str) -> float:
    """Word-level Jaccard overlap between two chunk strings."""
    a_words = set(a.lower().split())
    b_words = set(b.lower().split())
    if not a_words or not b_words:
        return 0.0
    return len(a_words & b_words) / len(a_words | b_words)


# ── Batch evaluation ───────────────────────────────────────────────────────────

def compute_rag_metrics(
    questions: list[str],
    answers: list[str],
    retrieved_chunks_list: list[list[str]],
    expected_answers: Optional[list[str]] = None,
    relevant_chunks_list: Optional[list[list[str]]] = None,
    llm_judge: Optional[Callable[[str], str]] = None,
    embedding_fn: Optional[Callable[[list[str]], np.ndarray]] = None,
) -> dict:
    """
    Compute all RAG metrics over a batch of Q&A pairs.
    """
    faithfulness_scores = []
    relevancy_scores = []
    ctx_precision_scores = []
    ctx_recall_scores = []

    for i, (question, answer, chunks) in enumerate(
        zip(questions, answers, retrieved_chunks_list)
    ):
        faithfulness_scores.append(
            compute_faithfulness(answer, chunks, llm_judge=llm_judge)
        )
        relevancy_scores.append(
            compute_answer_relevancy(
                question, answer,
                embedding_fn=embedding_fn,
                llm_judge=llm_judge,
            )
        )
        
        # Improved Context Metrics
        expected = expected_answers[i] if expected_answers and i < len(expected_answers) else ""
        rel_chunks = relevant_chunks_list[i] if relevant_chunks_list and i < len(relevant_chunks_list) else None
        
        ctx_precision_scores.append(
            compute_context_precision(question, chunks, relevant_chunks=rel_chunks, llm_judge=llm_judge)
        )
        ctx_recall_scores.append(
            compute_context_recall(expected, chunks, relevant_chunks=rel_chunks)
        )

    result = {
        "faithfulness":    round(float(np.mean(faithfulness_scores)), 4) if faithfulness_scores else 0.0,
        "answer_relevancy": round(float(np.mean(relevancy_scores)), 4) if relevancy_scores else 0.0,
        "context_precision": round(float(np.mean(ctx_precision_scores)), 4) if ctx_precision_scores else 0.0,
        "context_recall":    round(float(np.mean(ctx_recall_scores)), 4) if ctx_recall_scores else 0.0,
    }

    return result
