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


@app.command("reset")
def reset(
    config: Path = ConfigOpt,
    video_id: str = typer.Option(
        None, "--video-id", help="Reset only this video's tracked state (default: reset everything)"
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt"),
) -> None:
    """Clear job-store tracking state so the next run reprocesses from scratch.

    Only resets what the pipeline believes is already done (the SQLite job
    store at paths.db_path) - it never deletes anything under paths.work_dir
    or paths.output_dir. Use this after manually clearing output_dir, so
    videos the job store still thinks are DONE aren't silently skipped.
    """
    from jinpipe.db import JobStore

    cfg = load_config(config)
    store = JobStore(cfg.paths.db_path)
    try:
        if video_id is not None:
            video = store.get_video(video_id)
            if video is None:
                console.print(f"[yellow]No tracked state found for video_id {video_id!r} - nothing to reset[/yellow]")
                return
            n_superchunks = len(store.get_superchunks(video_id))
            n_segments = len(store.get_segments(video_id))
            console.print(
                f"This will delete tracked state for video [bold]{video_id}[/bold]: "
                f"1 video, {n_superchunks} superchunk(s), {n_segments} segment(s) from {cfg.paths.db_path}"
            )
        else:
            videos = store.list_videos()
            if not videos:
                console.print("Job store is already empty - nothing to reset")
                return
            console.print(
                f"This will delete ALL tracked state: {len(videos)} video(s), "
                f"{len(store.list_superchunks())} superchunk(s), {len(store.list_segments())} segment(s) "
                f"from {cfg.paths.db_path}"
            )
        console.print("[dim]This does not delete any files under paths.work_dir or paths.output_dir.[/dim]")

        if not yes:
            typer.confirm("Continue?", abort=True)

        counts = store.reset_video(video_id) if video_id is not None else store.reset_all()
        console.print(f"[green]OK[/green] reset: {counts}")
    finally:
        store.close()


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
