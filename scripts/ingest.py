from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import get_settings
from memory.ingestion import ConsolidationStats, EnrichmentStats, IngestionPipeline, IngestionStats

app = typer.Typer(help="Run phased ingestion for WhatsApp memory data.")
console = Console()


def _configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def _render_fast_ingestion(stats: IngestionStats) -> None:
    table = Table(title="Fast Ingestion Summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Files Seen", str(stats.files_seen))
    table.add_row("Files Ingested", str(stats.files_ingested))
    table.add_row("Files Skipped", str(stats.files_skipped))
    table.add_row("Messages Ingested", str(stats.messages_ingested))
    table.add_row("Episodes Ingested", str(stats.episodes_ingested))
    table.add_row("Weekly Summaries", str(stats.weekly_summaries_created))
    table.add_row("Facts Created", str(stats.facts_created))
    console.print(table)


def _render_enrichment(total: EnrichmentStats, passes: int) -> None:
    table = Table(title="Deep Enrichment Summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Passes Run", str(passes))
    table.add_row("Deep Episodes Seen", str(total.deep_episodes_seen))
    table.add_row("Deep Episodes Enriched", str(total.deep_episodes_enriched))
    table.add_row("Facts Created", str(total.facts_created))
    console.print(table)


def _render_consolidation(total: ConsolidationStats, passes: int) -> None:
    table = Table(title="Weekly Consolidation Summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Passes Run", str(passes))
    table.add_row("Weeks Seen", str(total.weeks_seen))
    table.add_row("Weeks Consolidated", str(total.weeks_consolidated))
    console.print(table)


@app.command("run")
def run(
    force: bool = typer.Option(False, help="Reingest files even if hashes are unchanged."),
    log_level: str = typer.Option("INFO", help="Logging level (DEBUG, INFO, WARNING)."),
) -> None:
    _configure_logging(log_level)
    pipeline = IngestionPipeline(get_settings())
    stats = pipeline.ingest_all(force=force)
    _render_fast_ingestion(stats)


@app.command("enrich")
def enrich(
    limit: int = typer.Option(50, min=1, help="Deep episodes to process per pass."),
    drain: bool = typer.Option(
        False,
        help="Keep running passes until fewer than `limit` deep episodes are available.",
    ),
    max_passes: int = typer.Option(30, min=1, help="Upper bound on drain passes."),
    log_level: str = typer.Option("INFO", help="Logging level (DEBUG, INFO, WARNING)."),
) -> None:
    _configure_logging(log_level)
    pipeline = IngestionPipeline(get_settings())

    total = EnrichmentStats()
    passes_run = 0
    target_passes = max_passes if drain else 1
    for _ in range(target_passes):
        batch = pipeline.enrich_deep_episodes(limit=limit)
        passes_run += 1
        total.deep_episodes_seen += batch.deep_episodes_seen
        total.deep_episodes_enriched += batch.deep_episodes_enriched
        total.facts_created += batch.facts_created
        if batch.deep_episodes_seen < limit:
            break

    _render_enrichment(total, passes_run)


@app.command("consolidate")
def consolidate(
    limit: int = typer.Option(20, min=1, help="Weekly groups to process per pass."),
    drain: bool = typer.Option(
        False,
        help="Keep running passes until fewer than `limit` weekly groups are available.",
    ),
    max_passes: int = typer.Option(20, min=1, help="Upper bound on drain passes."),
    log_level: str = typer.Option("INFO", help="Logging level (DEBUG, INFO, WARNING)."),
) -> None:
    _configure_logging(log_level)
    pipeline = IngestionPipeline(get_settings())

    total = ConsolidationStats()
    passes_run = 0
    target_passes = max_passes if drain else 1
    for _ in range(target_passes):
        batch = pipeline.consolidate_weekly_summaries(limit=limit)
        passes_run += 1
        total.weeks_seen += batch.weeks_seen
        total.weeks_consolidated += batch.weeks_consolidated
        if batch.weeks_seen < limit:
            break

    _render_consolidation(total, passes_run)


if __name__ == "__main__":
    app()
