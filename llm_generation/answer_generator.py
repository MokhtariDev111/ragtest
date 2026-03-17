"""
llm_generation/answer_generator.py
=====================================
Orchestrates: retrieval → prompt building → LLM call → timing.
"""

from __future__ import annotations

import time
from typing import Optional

from loguru import logger

from llm_generation.llm_interface import BaseLLM
from llm_generation.prompt_builder import PromptBuilder
from retrieval_system.retrievers import BaseRetriever


class AnswerGenerator:
    """
    End-to-end RAG answer generation orchestrator.

    Parameters
    ----------
    retriever      : an initialised retriever (BasicRetriever, etc.)
    llm            : an initialised LLM (OllamaLLM, etc.)
    prompt_builder : a PromptBuilder instance
    top_k          : number of chunks to retrieve
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: BaseLLM,
        prompt_builder: Optional[PromptBuilder] = None,
        top_k: int = 5,
    ):
        self.retriever = retriever
        self.llm = llm
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.top_k = top_k

    def answer(self, question: str) -> dict:
        """
        Run the full RAG answer pipeline for a single question.

        Returns
        -------
        dict with keys:
            question        (str)
            retrieved_chunks (list[str])
            prompt          (str)
            answer          (str)
            retrieval_latency_s (float)
            generation_latency_s (float)
            total_latency_s (float)
        """
        t_total_start = time.perf_counter()

        # ── Retrieval ──────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        chunks = self.retriever.retrieve(question, top_k=self.top_k)
        retrieval_latency_s = round(time.perf_counter() - t0, 4)
        logger.debug(f"Retrieved {len(chunks)} chunks in {retrieval_latency_s:.3f}s")

        # ── Prompt building ────────────────────────────────────────────────────
        prompt = self.prompt_builder.build(question, chunks)

        # ── LLM generation ────────────────────────────────────────────────────
        t0 = time.perf_counter()
        answer_text = self.llm.generate(prompt)
        generation_latency_s = round(time.perf_counter() - t0, 4)
        logger.debug(f"LLM generated answer in {generation_latency_s:.3f}s")

        total_latency_s = round(time.perf_counter() - t_total_start, 4)

        return {
            "question": question,
            "retrieved_chunks": chunks,
            "prompt": prompt,
            "answer": answer_text,
            "retrieval_latency_s": retrieval_latency_s,
            "generation_latency_s": generation_latency_s,
            "total_latency_s": total_latency_s,
        }

    def answer_batch(self, questions: list[str]) -> list[dict]:
        """Run answer() for each question and return list of result dicts."""
        results = []
        for i, q in enumerate(questions):
            logger.info(f"Processing question {i+1}/{len(questions)}: {q[:60]}...")
            results.append(self.answer(q))
        return results
