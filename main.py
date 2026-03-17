"""
main.py
========
CLI entry point for the RAG Benchmarking Framework.

Commands
--------
  run      – run experiments from a config file
  export   – export results to CSV
  dashboard– launch the Streamlit dashboard

Usage
-----
    python main.py run --config config/experiment_config.yaml
    python main.py export
    python main.py dashboard
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import yaml
from loguru import logger


# ── Logger setup ───────────────────────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> — <level>{message}</level>",
    level="INFO",
)
logger.add(
    "results_storage/run.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG",
)


@click.group()
def cli():
    """RAG Benchmarking Framework — automated pipeline evaluation platform."""


# ── run ───────────────────────────────────────────────────────────────────────

@cli.command()
@click.option(
    "--config",
    "-c",
    default="config/experiment_config.yaml",
    show_default=True,
    help="Path to experiment YAML config file.",
)
@click.option(
    "--max-experiments",
    "-n",
    default=None,
    type=int,
    help="Override max_experiments from config.",
)
def run(config: str, max_experiments: int | None):
    """Run the full benchmark experiment suite."""
    config_path = Path(config)
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if max_experiments is not None:
        cfg.setdefault("experiment_runner", {})["max_experiments"] = max_experiments

    logger.info(f"config: {config_path}")
    logger.info(f"project: {cfg['project']['name']}")

    from experiment_runner.runner import ExperimentRunner
    runner = ExperimentRunner(cfg)
    runner.run()


# ── export ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.option(
    "--db",
    default="results_storage/experiments.db",
    show_default=True,
    help="Path to the SQLite experiments database.",
)
@click.option(
    "--output-dir",
    default="results_storage/exports",
    show_default=True,
    help="Directory for exported CSV files.",
)
@click.option("--summary-only", is_flag=True, default=False, help="Export only the summary table.")
def export(db: str, output_dir: str, summary_only: bool):
    """Export experiment results to CSV."""
    from results_storage.database import ExperimentDatabase
    from results_storage.exporter import ResultsExporter

    database = ExperimentDatabase(db)
    exporter = ResultsExporter(database, output_dir=output_dir)

    if not summary_only:
        path = exporter.export_all()
        if path:
            click.echo(f"✅ Exported: {path}")

    path = exporter.export_summary()
    if path:
        click.echo(f"✅ Summary:  {path}")


# ── dashboard ─────────────────────────────────────────────────────────────────

@cli.command()
@click.option(
    "--port",
    default=8501,
    show_default=True,
    help="Port to run Streamlit on.",
)
def dashboard(port: int):
    """Launch the Streamlit benchmark dashboard."""
    import subprocess
    dashboard_path = Path(__file__).parent / "dashboard" / "app.py"
    logger.info(f"Launching dashboard at http://localhost:{port}")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(dashboard_path),
         "--server.port", str(port)],
        check=True,
    )


if __name__ == "__main__":
    cli()
