"""
experiment_runner/runner.py
=============================
Automated experiment runner.

Reads the configuration, builds the full cartesian product of enabled techniques,
runs each combination end-to-end through the RAG pipeline, evaluates results,
and stores them in the SQLite database.

Pipeline per experiment:
  documents → OCR (images) / direct load (text/PDF)
  → text cleaning → chunking → embeddings → vector DB indexing
  → retrieval → LLM answer generation → evaluation → logging
"""

from __future__ import annotations

import json
import traceback
from itertools import product
from pathlib import Path
from typing import Optional

from loguru import logger

from data_ingestion.document_loader import DocumentLoader
from data_ingestion.ocr_pipeline import get_ocr_engine
from text_processing.text_cleaner import TextCleaner
from text_processing.chunker import get_chunker
from embedding_layer.embedder import get_embedder
from vector_database.vector_store import get_vector_store
from retrieval_system.retrievers import BasicRetriever, HybridRetriever, get_retriever
from llm_generation.llm_interface import get_llm
from llm_generation.prompt_builder import PromptBuilder
from llm_generation.answer_generator import AnswerGenerator
from evaluation_framework.retrieval_metrics import compute_retrieval_metrics
from evaluation_framework.rag_metrics import compute_rag_metrics
from results_storage.database import ExperimentDatabase


class ExperimentRunner:
    """
    Automated RAG benchmarking experiment runner.

    Parameters
    ----------
    config : full config dict (loaded from YAML)
    """

    def __init__(self, config: dict):
        self.config = config
        self.db = ExperimentDatabase(config["project"]["db_file"])
        self.evaluation_dataset = self._load_evaluation_dataset()

    # ── Public ─────────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Run all enabled combinations and log results."""
        combinations = list(self._build_combinations())
        max_exp = self.config.get("experiment_runner", {}).get("max_experiments", 500)
        combinations = combinations[:max_exp]

        logger.info(
            f"Starting experiments: {len(combinations)} combinations to run"
        )

        for idx, combo in enumerate(combinations, start=1):
            logger.info(
                f"[{idx}/{len(combinations)}] "
                f"ocr={combo['ocr_engine']} | chunk={combo['chunking_strategy']}({combo['chunk_size']}) | "
                f"emb={combo['embedding_model']} | db={combo['vector_db']} | "
                f"ret={combo['retrieval_strategy']} | llm={combo['llm_model']}"
            )
            try:
                record = self._run_single(combo)
                record["status"] = "completed"
            except Exception as exc:
                logger.error(f"Experiment failed: {exc}\n{traceback.format_exc()}")
                record = {**combo, "status": "failed", "error_message": str(exc)}

            if record.get("status") != "failed" or \
               self.config.get("experiment_runner", {}).get("save_on_failure", True):
                self.db.insert_experiment(record)

        logger.info(f"All experiments completed. Total stored: {self.db.count()}")

    # ── Private helpers ────────────────────────────────────────────────────────

    def _build_combinations(self):
        """Generate the cartesian product of all enabled technique options."""
        cfg = self.config

        ocr_engines = [
            e["name"] for e in cfg.get("ocr", {}).get("engines", [])
            if e.get("enabled", True)
        ] or ["none"]

        chunking_strategies = [
            c["name"] for c in cfg["text_processing"]["chunking"]["strategies"]
            if c.get("enabled", True)
        ]
        chunk_sizes = cfg["text_processing"]["chunking"]["chunk_sizes"]

        embedders = [
            e["name"] for e in cfg["embeddings"]["models"]
            if e.get("enabled", True)
        ]

        vector_dbs = [
            d["name"] for d in cfg["vector_databases"]["stores"]
            if d.get("enabled", True)
        ]

        retrievers = [
            r["name"] for r in cfg["retrieval"]["strategies"]
            if r.get("enabled", True)
        ]

        llm_models = [
            m["name"] for m in cfg["llm"]["models"]
            if m.get("enabled", True)
        ]

        for ocr, strategy, size, emb, vdb, ret, llm in product(
            ocr_engines, chunking_strategies, chunk_sizes,
            embedders, vector_dbs, retrievers, llm_models,
        ):
            yield {
                "ocr_engine": ocr,
                "chunking_strategy": strategy,
                "chunk_size": size,
                "embedding_model": emb,
                "vector_db": vdb,
                "retrieval_strategy": ret,
                "llm_model": llm,
            }

    def _run_single(self, combo: dict) -> dict:
        """Execute one full RAG pipeline and return the result record."""
        cfg = self.config
        questions = [q["question"] for q in self.evaluation_dataset]
        expected_answers = [q["expected_answer"] for q in self.evaluation_dataset]

        # ── 1. Load and preprocess documents ──────────────────────────────────
        loader = DocumentLoader()
        cleaner = TextCleaner(
            normalize_unicode=cfg["text_processing"]["cleaning"].get("normalize_unicode", True),
            remove_control_chars=cfg["text_processing"]["cleaning"].get("remove_control_chars", True),
            remove_extra_whitespace=cfg["text_processing"]["cleaning"].get("remove_extra_whitespace", True),
        )

        raw_texts = []
        doc_dir = Path(cfg["data"]["documents_dir"])

        for doc in loader.load_directory(doc_dir, recursive=True):
            if doc["type"] == "text":
                raw_texts.append(cleaner.clean(doc["content"]))
            elif doc["type"] == "image" and combo["ocr_engine"] != "none":
                try:
                    engine = get_ocr_engine(combo["ocr_engine"])
                    text = engine.extract_text(doc["content"])
                    raw_texts.append(cleaner.clean(text))
                except Exception as exc:
                    logger.warning(f"OCR failed: {exc}")

        if not raw_texts:
            raise RuntimeError("No text extracted from documents")

        full_text = "\n\n".join(raw_texts)

        # ── 2. Chunking ────────────────────────────────────────────────────────
        chunk_overlap = cfg["text_processing"]["chunking"].get("chunk_overlap", 50)
        chunker = get_chunker(
            combo["chunking_strategy"],
            chunk_size=combo["chunk_size"],
            chunk_overlap=chunk_overlap,
        )
        chunks = chunker.split(full_text)
        if not chunks:
            raise RuntimeError("Chunking produced 0 chunks")

        # ── 3. Embeddings ──────────────────────────────────────────────────────
        embedder = get_embedder(
            combo["embedding_model"],
            device=cfg["embeddings"].get("device", "cpu"),
            batch_size=cfg["embeddings"].get("batch_size", 32),
        )
        chunk_vecs = embedder.embed(chunks)

        # ── 4. Vector DB indexing ──────────────────────────────────────────────
        store = get_vector_store(combo["vector_db"])
        store.clear()
        store.index(chunks, chunk_vecs)

        # ── 5. Retriever ───────────────────────────────────────────────────────
        top_k = cfg["retrieval"].get("top_k", 5)
        llm_config = cfg["llm"]
        llm = get_llm(
            combo["llm_model"],
            base_url=llm_config.get("base_url", "http://localhost:11434"),
            timeout=llm_config.get("timeout", 120),
            max_tokens=llm_config.get("max_tokens", 512),
            temperature=llm_config.get("temperature", 0.1),
        )

        retriever_extra = {}
        if combo["retrieval_strategy"] == "multi_query":
            retriever_extra["llm_generate_fn"] = llm.generate
        elif combo["retrieval_strategy"] == "hybrid":
            retriever_extra["bm25_weight"] = 0.4
            retriever_extra["vector_weight"] = 0.6

        retriever = get_retriever(
            combo["retrieval_strategy"],
            embedder=embedder,
            store=store,
            **retriever_extra,
        )

        # For hybrid retriever, set the corpus
        if hasattr(retriever, "set_corpus"):
            retriever.set_corpus(chunks)

        # ── 6. Answer generation ───────────────────────────────────────────────
        prompt_builder = PromptBuilder(language="en")
        generator = AnswerGenerator(
            retriever=retriever,
            llm=llm,
            prompt_builder=prompt_builder,
            top_k=top_k,
        )

        results = generator.answer_batch(questions)
        answers = [r["answer"] for r in results]
        retrieved_chunks_list = [r["retrieved_chunks"] for r in results]
        latencies = [r["total_latency_s"] for r in results]

        mean_retrieval_s = sum(r["retrieval_latency_s"] for r in results) / len(results)
        mean_generation_s = sum(r["generation_latency_s"] for r in results) / len(results)
        mean_total_s = sum(latencies) / len(latencies)

        # ── 7. Evaluation ──────────────────────────────────────────────────────
        judge_fn = None
        # Use the currently tested LLM model as the judge for its own answers, 
        # unless a specific universal judge was set in config.
        judge_model = cfg.get("evaluation", {}).get("judge_model")
        if not judge_model:
            judge_model = combo["llm_model"]
            
        try:
            judge_llm = get_llm(judge_model, base_url=llm_config.get("base_url", "http://localhost:11434"))
            judge_fn = judge_llm.generate
        except Exception:
            pass

        embedding_fn = embedder.embed
        
        # Prepare ground truth context if available
        relevant_chunks_list = [q.get("relevant_chunks", []) for q in self.evaluation_dataset]

        # 7.1 Retrieval Metrics
        retrieval_metrics = compute_retrieval_metrics(
            retrieved_list=retrieved_chunks_list,
            relevant_list=relevant_chunks_list,
            k=top_k
        )

        # 7.2 RAG Metrics
        rag_metrics = compute_rag_metrics(
            questions=questions,
            answers=answers,
            retrieved_chunks_list=retrieved_chunks_list,
            expected_answers=expected_answers,
            relevant_chunks_list=relevant_chunks_list,
            llm_judge=judge_fn,
            embedding_fn=embedding_fn,
        )

        return {
            **combo,
            # Retrieval
            "precision_at_k": retrieval_metrics.get(f"precision_at_{top_k}"),
            "recall_at_k": retrieval_metrics.get(f"recall_at_{top_k}"),
            "mrr": retrieval_metrics.get("mrr"),
            "ndcg_at_k": retrieval_metrics.get(f"ndcg_at_{top_k}"),
            # RAG
            "faithfulness": rag_metrics.get("faithfulness"),
            "answer_relevancy": rag_metrics.get("answer_relevancy"),
            "context_precision": rag_metrics.get("context_precision"),
            "context_recall": rag_metrics.get("context_recall"),
            # Latency
            "retrieval_latency_s": round(mean_retrieval_s, 4),
            "generation_latency_s": round(mean_generation_s, 4),
            "total_latency_s": round(mean_total_s, 4),
            "num_questions": len(questions),
        }

    def _load_evaluation_dataset(self) -> list[dict]:
        """Load the evaluation Q&A dataset from JSON."""
        path = Path(self.config["data"]["evaluation_file"])
        if not path.exists():
            logger.warning(f"Evaluation dataset not found: {path}")
            return []
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        questions = data if isinstance(data, list) else data.get("questions", [])
        logger.info(f"Loaded {len(questions)} evaluation questions")
        return questions
