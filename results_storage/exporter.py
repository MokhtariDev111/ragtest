"""
results_storage/exporter.py
=============================
Export experiment results from SQLite to timestamped CSV files.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from loguru import logger

from results_storage.database import ExperimentDatabase


class ResultsExporter:
    """
    Exports experiment data from the SQLite database to CSV.

    Parameters
    ----------
    db     : ExperimentDatabase instance
    output_dir : directory for exported CSV files
    """

    def __init__(
        self,
        db: ExperimentDatabase,
        output_dir: str = "results_storage/exports",
    ):
        self.db = db
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_all(self, filename: str = "") -> str:
        """
        Export all experiments to a CSV file.

        Parameters
        ----------
        filename : optional custom filename (default: timestamped)

        Returns
        -------
        str – absolute path to the exported file
        """
        df = self.db.get_all()
        if df.empty:
            logger.warning("No experiments to export.")
            return ""

        if not filename:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"experiments_{ts}.csv"

        out_path = self.output_dir / filename
        df.to_csv(str(out_path), index=False, encoding="utf-8")
        logger.info(f"Exported {len(df)} experiments → {out_path}")
        return str(out_path)

    def export_summary(self, filename: str = "") -> str:
        """
        Export the aggregated summary table to CSV.
        """
        df = self.db.get_summary()
        if df.empty:
            logger.warning("No summary data to export.")
            return ""

        if not filename:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"summary_{ts}.csv"

        out_path = self.output_dir / filename
        df.to_csv(str(out_path), index=False, encoding="utf-8")
        logger.info(f"Exported summary ({len(df)} rows) → {out_path}")
        return str(out_path)
