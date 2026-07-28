"""JinPipe command-line entrypoint."""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console

from jinpipe.config import load_config

app = typer.Typer(add_completion=False, help="Async YouTube -> TTS-dataset pipeline")
console = Console()

ConfigOpt = typer.Option(..., "--config", "-c", help="Path to a YAML config file")
VerboseOpt = typer.Option(False, "--verbose", "-v", help="Enable INFO-level logging")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )


@app.command("check-config")
def check_config(config: Path = ConfigOpt) -> None:
    """Load and validate a config file without running anything."""
    cfg = load_config(config)
    console.print(f"[green]OK[/green] config is valid: {config}")
    console.print(cfg.model_dump())


@app.command("run")
def run(config: Path = ConfigOpt, verbose: bool = VerboseOpt) -> None:
    """Run the full pipeline (discover -> download -> ... -> package)."""
    from jinpipe.orchestrator import run_pipeline

    _setup_logging(verbose)
    cfg = load_config(config)
    run_pipeline(cfg)


@app.command("status")
def status(config: Path = ConfigOpt) -> None:
    """Print job-store progress (counts per stage/status)."""
    from jinpipe.db import JobStore

    cfg = load_config(config)
    store = JobStore(cfg.paths.db_path)
    store.print_status(console)
    store.close()


@app.command("resume")
def resume(config: Path = ConfigOpt, verbose: bool = VerboseOpt) -> None:
    """Resume a previous run: reset stale RUNNING rows to PENDING, then run."""
    from jinpipe.db import JobStore
    from jinpipe.orchestrator import run_pipeline

    _setup_logging(verbose)
    cfg = load_config(config)
    store = JobStore(cfg.paths.db_path)
    reset = store.recover_stale_running(max_age_s=0)
    console.print(f"Reset {reset} stale RUNNING row(s) to PENDING")
    store.close()
    run_pipeline(cfg)


@app.command("manifest")
def manifest(config: Path = ConfigOpt) -> None:
    """Rebuild manifest.jsonl in output_dir from the per-segment JSON files on disk."""
    from jinpipe.stages.package import write_manifest

    cfg = load_config(config)
    manifest_path = cfg.paths.output_dir / "manifest.jsonl"
    count = write_manifest(cfg.paths.output_dir, manifest_path)
    console.print(f"[green]OK[/green] wrote {count} entries to {manifest_path}")


if __name__ == "__main__":
    app()
