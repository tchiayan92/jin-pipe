"""JinPipe command-line entrypoint."""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console

from jinpipe.config import load_config

app = typer.Typer(add_completion=False, help="Async YouTube/Local audio -> TTS-dataset pipeline")
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
    """Rebuild manifest.jsonl and manifest.csv in output_dir from the per-segment JSON files on disk."""
    from jinpipe.stages.package import write_csv, write_manifest

    cfg = load_config(config)
    manifest_path = cfg.paths.output_dir / "manifest.jsonl"
    count = write_manifest(cfg.paths.output_dir, manifest_path)
    csv_path = cfg.paths.output_dir / "manifest.csv"
    write_csv(cfg.paths.output_dir, csv_path, audio_format=cfg.package.audio_format)
    console.print(f"[green]OK[/green] wrote {count} entries to {manifest_path} and {csv_path}")


@app.command("stats")
def stats(config: Path = ConfigOpt) -> None:
    """Show dataset stats for output_dir: speaker count, total speech hours, per-speaker breakdown."""
    from jinpipe.stages.package import dataset_stats, print_dataset_stats

    cfg = load_config(config)
    result = dataset_stats(cfg.paths.output_dir)
    if result["num_segments"] == 0:
        console.print(f"No packaged segments found in {cfg.paths.output_dir}")
        return
    print_dataset_stats(console, result)


@app.command("filter-speakers")
def filter_speakers_cmd(
    config: Path = ConfigOpt,
    keep: list[str] = typer.Option(
        ..., "--keep", help="Speaker label to retain; repeat for multiple. All other speakers are deleted."
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be removed without deleting anything"),
) -> None:
    """Delete packaged segments (audio+json) whose speaker isn't in --keep, then rebuild manifest.jsonl.

    Only touches output_dir - never the job store - so a video's tracked
    status is unaffected; re-run `package`/`run` to regenerate anything
    deleted here.
    """
    from jinpipe.stages.package import filter_speakers, write_csv, write_manifest

    cfg = load_config(config)
    keep_set = set(keep)
    preview = filter_speakers(cfg.paths.output_dir, keep_set, dry_run=True)
    console.print(f"Speakers to keep: {sorted(keep_set)}")
    console.print(f"{preview['kept']} segment(s) kept, {preview['removed']} segment(s) would be removed")
    if preview["removed"] == 0:
        console.print("[green]OK[/green] nothing to remove")
        return
    if dry_run:
        return

    if not yes:
        typer.confirm("Continue?", abort=True)

    result = filter_speakers(cfg.paths.output_dir, keep_set, dry_run=False)
    manifest_path = cfg.paths.output_dir / "manifest.jsonl"
    count = write_manifest(cfg.paths.output_dir, manifest_path)
    write_csv(cfg.paths.output_dir, cfg.paths.output_dir / "manifest.csv", audio_format=cfg.package.audio_format)
    console.print(f"[green]OK[/green] removed {result['removed']} segment(s); manifest now has {count} entries")


@app.command("transcribe")
def transcribe(
    config: Path = ConfigOpt,
    input_dir: Path = typer.Option(
        ..., "--input-dir", "-i", help="Folder of already-chunked audio files to transcribe directly"
    ),
    output: Path = typer.Option(
        None, "--output", "-o", help="CSV output path (default: <output_dir>/transcriptions.csv)"
    ),
    verbose: bool = VerboseOpt,
) -> None:
    """Transcribe a folder of pre-chunked audio files straight to a CSV - no VAD/rechunk/diarize/package.

    Each file directly inside --input-dir is treated as a single utterance
    already; this loads one Whisper model (from the config's `asr` section)
    and writes a numbered (no, filename, text) row per file.
    """
    from jinpipe.orchestrator import detect_gpu_ids
    from jinpipe.stages.transcribe_only import transcribe_folder, write_csv

    _setup_logging(verbose)
    cfg = load_config(config)

    if not input_dir.is_dir():
        console.print(f"[red]Error[/red] --input-dir is not a directory: {input_dir}")
        raise typer.Exit(code=1)

    device = cfg.asr.device
    if device == "auto":
        device = "cuda" if detect_gpu_ids() else "cpu"

    output_path = output or (cfg.paths.output_dir / "transcriptions.csv")
    rows = transcribe_folder(input_dir, cfg.asr, device=device)
    if not rows:
        console.print(f"[yellow]No audio files found directly inside {input_dir}[/yellow]")
        return
    write_csv(rows, output_path)
    console.print(f"[green]OK[/green] transcribed {len(rows)} file(s) to {output_path}")


@app.command("view")
def view(
    config: Path = ConfigOpt,
    host: str = typer.Option("127.0.0.1", "--host", help="Interface to bind the Gradio server to"),
    port: int = typer.Option(7860, "--port", help="Port to bind the Gradio server to"),
    share: bool = typer.Option(False, "--share", help="Create a public Gradio share link"),
) -> None:
    r"""Launch a Gradio app to browse packaged chunk audio with transcript/speaker/overlap annotations.

    Requires the `viewer` extra: pip install "jinpipe\[viewer]"
    """
    from jinpipe.viewer import launch_viewer

    cfg = load_config(config)
    launch_viewer(cfg.paths.output_dir, server_name=host, server_port=port, share=share)


if __name__ == "__main__":
    app()
