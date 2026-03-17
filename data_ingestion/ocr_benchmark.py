"""
data_ingestion/ocr_benchmark.py
=================================
Runs all enabled OCR engines over a reference set of images and
computes WER, CER, processing time per page, and peak memory usage.

Usage
-----
    from data_ingestion.ocr_benchmark import OCRBenchmarker
    benchmarker = OCRBenchmarker(ocr_config)
    results = benchmarker.run(image_paths, ground_truths)
    df = benchmarker.to_dataframe()
"""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Optional

import psutil
import pandas as pd
from jiwer import wer, cer
from loguru import logger
from PIL import Image

from data_ingestion.ocr_pipeline import get_ocr_engine, BaseOCREngine


class OCRBenchmarker:
    """
    Benchmark multiple OCR engines on a set of reference images.

    Parameters
    ----------
    ocr_config : list[dict]
        Each item: {"name": "tesseract", "enabled": True, "lang": "eng+fra", ...}
    """

    def __init__(self, ocr_config: list[dict]):
        self.ocr_config = ocr_config
        self.results: list[dict] = []

    def run(
        self,
        image_paths: list[str | Path],
        ground_truths: Optional[list[str]] = None,
        lang: str = "en",
    ) -> list[dict]:
        """
        Run each enabled OCR engine on every image.

        Parameters
        ----------
        image_paths   : list of paths to reference images
        ground_truths : list of reference texts (same length as image_paths).
                        If None, WER and CER will be NaN.
        lang          : language hint passed to each engine ('en' | 'fr')

        Returns
        -------
        list[dict] – one dict per (engine, image) combination
        """
        if ground_truths and len(ground_truths) != len(image_paths):
            raise ValueError("ground_truths must have the same length as image_paths")

        self.results = []
        for engine_cfg in self.ocr_config:
            if not engine_cfg.get("enabled", True):
                continue
            engine_name = engine_cfg["name"]
            try:
                engine = get_ocr_engine(engine_name)
            except Exception as exc:
                logger.warning(f"Could not load engine '{engine_name}': {exc}")
                continue

            logger.info(f"Benchmarking OCR engine: {engine_name} on {len(image_paths)} image(s)")
            for idx, img_path in enumerate(image_paths):
                record = self._benchmark_single(
                    engine=engine,
                    engine_name=engine_name,
                    img_path=Path(img_path),
                    reference=ground_truths[idx] if ground_truths else None,
                    lang=lang,
                )
                self.results.append(record)

        return self.results

    def _benchmark_single(
        self,
        engine: BaseOCREngine,
        engine_name: str,
        img_path: Path,
        reference: Optional[str],
        lang: str,
    ) -> dict:
        """Run one engine on one image, capturing all metrics."""
        image = Image.open(img_path).convert("RGB")

        # Measure memory before
        proc = psutil.Process()
        mem_before = proc.memory_info().rss / (1024 * 1024)  # MB

        t0 = time.perf_counter()
        try:
            text = engine.extract_text(image, lang=lang)
        except Exception as exc:
            logger.error(f"{engine_name} failed on {img_path.name}: {exc}")
            text = ""
        elapsed_s = round(time.perf_counter() - t0, 4)

        # Measure memory after
        gc.collect()
        mem_after = proc.memory_info().rss / (1024 * 1024)
        mem_used_mb = round(max(mem_after - mem_before, 0), 2)

        # Compute WER / CER if reference provided
        word_error_rate = None
        char_error_rate = None
        if reference:
            try:
                word_error_rate = round(wer(reference.lower(), text.lower()), 4)
                char_error_rate = round(cer(reference.lower(), text.lower()), 4)
            except Exception:
                pass

        return {
            "engine": engine_name,
            "image": img_path.name,
            "extracted_text_len": len(text),
            "latency_s": elapsed_s,
            "memory_mb": mem_used_mb,
            "wer": word_error_rate,
            "cer": char_error_rate,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert benchmark results to a pandas DataFrame."""
        return pd.DataFrame(self.results)

    def summary(self) -> pd.DataFrame:
        """
        Return per-engine aggregate summary (mean WER, CER, latency, memory).
        """
        df = self.to_dataframe()
        if df.empty:
            return df
        return (
            df.groupby("engine")
            .agg(
                mean_wer=("wer", "mean"),
                mean_cer=("cer", "mean"),
                mean_latency_s=("latency_s", "mean"),
                mean_memory_mb=("memory_mb", "mean"),
                images_processed=("image", "count"),
            )
            .reset_index()
            .sort_values("mean_wer")
        )
