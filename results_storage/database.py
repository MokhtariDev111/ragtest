"""
results_storage/database.py
=============================
SQLite-backed experiment results store.

Schema (experiments table):
  experiment_id, timestamp, config_hash,
  ocr_engine, chunking_strategy, chunk_size,
  embedding_model, vector_db, retrieval_strategy, llm_model,
  -- retrieval metrics
  precision_at_k, recall_at_k, mrr, ndcg_at_k,
  -- RAG metrics
  faithfulness, answer_relevancy, context_precision, context_recall,
  -- latency
  retrieval_latency_s, generation_latency_s, total_latency_s,
  -- status
  status, error_message
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id       TEXT PRIMARY KEY,
    timestamp           TEXT NOT NULL,
    config_hash         TEXT,

    -- Pipeline configuration
    ocr_engine          TEXT,
    chunking_strategy   TEXT,
    chunk_size          INTEGER,
    embedding_model     TEXT,
    vector_db           TEXT,
    retrieval_strategy  TEXT,
    llm_model           TEXT,

    -- Retrieval metrics
    precision_at_k      REAL,
    recall_at_k         REAL,
    mrr                 REAL,
    ndcg_at_k           REAL,

    -- RAG metrics
    faithfulness        REAL,
    answer_relevancy    REAL,
    context_precision   REAL,
    context_recall      REAL,

    -- Latency (seconds)
    retrieval_latency_s REAL,
    generation_latency_s REAL,
    total_latency_s     REAL,

    -- Status
    num_questions       INTEGER DEFAULT 0,
    status              TEXT DEFAULT 'completed',
    error_message       TEXT
);
"""


class ExperimentDatabase:
    """
    Manages the SQLite database for experiment results.

    Parameters
    ----------
    db_path : path to the SQLite database file
    """

    def __init__(self, db_path: str = "results_storage/experiments.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        """Create tables if they do not exist."""
        with self._connect() as conn:
            conn.execute(CREATE_TABLE_SQL)
        logger.debug(f"Database ready: {self.db_path}")

    def insert_experiment(self, record: dict) -> str:
        """
        Insert a new experiment result.

        Parameters
        ----------
        record : dict matching the experiments table columns.
                 experiment_id and timestamp are auto-generated if missing.

        Returns
        -------
        experiment_id : str
        """
        record = record.copy()
        if "experiment_id" not in record:
            record["experiment_id"] = str(uuid.uuid4())
        if "timestamp" not in record:
            record["timestamp"] = datetime.now().isoformat()

        # Generate a config hash for deduplication
        config_keys = [
            "ocr_engine", "chunking_strategy", "chunk_size",
            "embedding_model", "vector_db", "retrieval_strategy", "llm_model",
        ]
        config_str = json.dumps({k: record.get(k) for k in config_keys}, sort_keys=True)
        record["config_hash"] = hashlib.md5(config_str.encode()).hexdigest()

        columns = ", ".join(record.keys())
        placeholders = ", ".join("?" * len(record))
        sql = f"INSERT OR REPLACE INTO experiments ({columns}) VALUES ({placeholders})"

        with self._connect() as conn:
            conn.execute(sql, list(record.values()))

        logger.info(f"Experiment logged: {record['experiment_id']}")
        return record["experiment_id"]

    def get_all(self) -> pd.DataFrame:
        """Return all experiments as a DataFrame."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM experiments ORDER BY timestamp DESC").fetchall()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([dict(r) for r in rows])

    def get_by_id(self, experiment_id: str) -> Optional[dict]:
        """Return a single experiment by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM experiments WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchone()
        return dict(row) if row else None

    def get_summary(self) -> pd.DataFrame:
        """Return aggregated summary stats grouped by key configuration fields."""
        sql = """
            SELECT
                ocr_engine,
                chunking_strategy,
                chunk_size,
                embedding_model,
                vector_db,
                retrieval_strategy,
                llm_model,
                COUNT(*) as num_runs,
                AVG(faithfulness) as avg_faithfulness,
                AVG(answer_relevancy) as avg_answer_relevancy,
                AVG(context_precision) as avg_context_precision,
                AVG(context_recall) as avg_context_recall,
                AVG(precision_at_k) as avg_precision_at_k,
                AVG(recall_at_k) as avg_recall_at_k,
                AVG(total_latency_s) as avg_total_latency_s
            FROM experiments
            WHERE status = 'completed'
            GROUP BY
                ocr_engine, chunking_strategy, chunk_size,
                embedding_model, vector_db, retrieval_strategy, llm_model
            ORDER BY avg_faithfulness DESC
        """
        with self._connect() as conn:
            rows = conn.execute(sql).fetchall()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([dict(r) for r in rows])

    def count(self) -> int:
        """Return the total number of logged experiments."""
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
