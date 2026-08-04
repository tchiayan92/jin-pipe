import json

import yaml
from typer.testing import CliRunner

from jinpipe.cli import app
from jinpipe.db import JobStore

runner = CliRunner()


def _write_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "sources": {"channels": []},
                "paths": {
                    "work_dir": str(tmp_path / "work"),
                    "output_dir": str(tmp_path / "output"),
                    "db_path": str(tmp_path / "work" / "jobs.sqlite3"),
                },
            }
        )
    )
    return config_path


def test_reset_empty_store_reports_nothing_to_reset(tmp_path):
    config_path = _write_config(tmp_path)
    store = JobStore(tmp_path / "work" / "jobs.sqlite3")
    store.close()

    result = runner.invoke(app, ["reset", "--config", str(config_path), "--yes"])

    assert result.exit_code == 0
    assert "already empty" in result.output


def test_reset_all_with_yes_flag_skips_prompt_and_deletes(tmp_path):
    config_path = _write_config(tmp_path)
    db_path = tmp_path / "work" / "jobs.sqlite3"
    store = JobStore(db_path)
    store.add_video("v1", "https://youtu.be/v1")
    store.update_video("v1", status="DONE")
    store.close()

    result = runner.invoke(app, ["reset", "--config", str(config_path), "--yes"])

    assert result.exit_code == 0
    assert "OK" in result.output
    store = JobStore(db_path)
    try:
        assert store.get_video("v1") is None
    finally:
        store.close()


def test_reset_without_yes_aborts_on_no(tmp_path):
    config_path = _write_config(tmp_path)
    db_path = tmp_path / "work" / "jobs.sqlite3"
    store = JobStore(db_path)
    store.add_video("v1", "https://youtu.be/v1")
    store.close()

    result = runner.invoke(app, ["reset", "--config", str(config_path)], input="n\n")

    assert result.exit_code != 0
    store = JobStore(db_path)
    try:
        assert store.get_video("v1") is not None
    finally:
        store.close()


def test_reset_video_id_targets_only_that_video(tmp_path):
    config_path = _write_config(tmp_path)
    db_path = tmp_path / "work" / "jobs.sqlite3"
    store = JobStore(db_path)
    store.add_video("v1", "https://youtu.be/v1")
    store.add_video("v2", "https://youtu.be/v2")
    store.close()

    result = runner.invoke(app, ["reset", "--config", str(config_path), "--video-id", "v1", "--yes"])

    assert result.exit_code == 0
    store = JobStore(db_path)
    try:
        assert store.get_video("v1") is None
        assert store.get_video("v2") is not None
    finally:
        store.close()


def test_reset_unknown_video_id_reports_nothing_to_reset(tmp_path):
    config_path = _write_config(tmp_path)
    store = JobStore(tmp_path / "work" / "jobs.sqlite3")
    store.close()

    result = runner.invoke(app, ["reset", "--config", str(config_path), "--video-id", "ghost", "--yes"])

    assert result.exit_code == 0
    assert "nothing to reset" in result.output


def _write_segment(output_dir, video_id, idx, duration, speaker):
    output_dir.mkdir(parents=True, exist_ok=True)
    segment_id = f"{video_id}_{idx:05d}"
    (output_dir / f"{segment_id}.json").write_text(
        json.dumps(
            {
                "segment_id": segment_id,
                "video_id": video_id,
                "source_url": f"https://youtu.be/{video_id}",
                "start": 0.0,
                "end": duration,
                "duration": duration,
                "text": "hello",
                "words": [],
                "speaker": speaker,
                "language": "en",
                "dnsmos_ovr": None,
                "exceeds_max_duration": False,
                "has_overlap": False,
            }
        )
    )
    (output_dir / f"{segment_id}.flac").write_bytes(b"fake-audio")


def test_stats_reports_speaker_and_duration_breakdown(tmp_path):
    config_path = _write_config(tmp_path)
    output_dir = tmp_path / "output"
    _write_segment(output_dir, "vid1", 0, 3600.0, "A")
    _write_segment(output_dir, "vid1", 1, 1800.0, "B")

    result = runner.invoke(app, ["stats", "--config", str(config_path)])

    assert result.exit_code == 0
    assert "2 segment(s)" in result.output
    assert "2 speaker(s)" in result.output
    assert "A" in result.output and "B" in result.output


def test_stats_on_empty_output_dir_reports_no_segments(tmp_path):
    config_path = _write_config(tmp_path)

    result = runner.invoke(app, ["stats", "--config", str(config_path)])

    assert result.exit_code == 0
    assert "No packaged segments found" in result.output


def test_filter_speakers_dry_run_reports_without_deleting(tmp_path):
    config_path = _write_config(tmp_path)
    output_dir = tmp_path / "output"
    _write_segment(output_dir, "vid1", 0, 2.0, "A")
    _write_segment(output_dir, "vid1", 1, 3.0, "B")

    result = runner.invoke(app, ["filter-speakers", "--config", str(config_path), "--keep", "A", "--dry-run"])

    assert result.exit_code == 0
    assert "1 segment(s) kept, 1 segment(s) would be removed" in result.output
    assert (output_dir / "vid1_00001.json").exists()


def test_filter_speakers_with_yes_deletes_and_rebuilds_manifest(tmp_path):
    config_path = _write_config(tmp_path)
    output_dir = tmp_path / "output"
    _write_segment(output_dir, "vid1", 0, 2.0, "A")
    _write_segment(output_dir, "vid1", 1, 3.0, "B")

    result = runner.invoke(app, ["filter-speakers", "--config", str(config_path), "--keep", "A", "--yes"])

    assert result.exit_code == 0
    assert "OK" in result.output
    assert not (output_dir / "vid1_00001.json").exists()
    assert not (output_dir / "vid1_00001.flac").exists()
    assert (output_dir / "vid1_00000.json").exists()

    manifest = (output_dir / "manifest.jsonl").read_text().strip().splitlines()
    assert len(manifest) == 1
    assert json.loads(manifest[0])["segment_id"] == "vid1_00000"


def test_filter_speakers_without_yes_aborts_on_no(tmp_path):
    config_path = _write_config(tmp_path)
    output_dir = tmp_path / "output"
    _write_segment(output_dir, "vid1", 0, 2.0, "A")
    _write_segment(output_dir, "vid1", 1, 3.0, "B")

    result = runner.invoke(app, ["filter-speakers", "--config", str(config_path), "--keep", "A"], input="n\n")

    assert result.exit_code != 0
    assert (output_dir / "vid1_00001.json").exists()


def test_filter_speakers_nothing_to_remove_skips_confirmation(tmp_path):
    config_path = _write_config(tmp_path)
    output_dir = tmp_path / "output"
    _write_segment(output_dir, "vid1", 0, 2.0, "A")

    result = runner.invoke(app, ["filter-speakers", "--config", str(config_path), "--keep", "A"])

    assert result.exit_code == 0
    assert "nothing to remove" in result.output


def _fake_transcribe(model, path, cfg):
    return f"transcript of {path.name}"


def test_transcribe_writes_csv_from_input_dir(tmp_path, monkeypatch):
    config_path = _write_config(tmp_path)
    input_dir = tmp_path / "chunks"
    input_dir.mkdir()
    (input_dir / "a.wav").write_bytes(b"fake-audio")
    (input_dir / "b.wav").write_bytes(b"fake-audio")

    monkeypatch.setattr("jinpipe.orchestrator.detect_gpu_ids", lambda: [])
    monkeypatch.setattr(
        "jinpipe.stages.transcribe_only.transcribe_folder",
        lambda input_dir, asr_cfg, device: [
            {"no": i, "filename": p.name, "duration_s": 1.5, "text": _fake_transcribe(None, p, asr_cfg)}
            for i, p in enumerate(sorted(input_dir.iterdir()), start=1)
        ],
    )

    result = runner.invoke(app, ["transcribe", "--config", str(config_path), "--input-dir", str(input_dir)])

    assert result.exit_code == 0
    assert "transcribed 2 file(s)" in result.output
    output_path = tmp_path / "output" / "transcriptions.csv"
    assert output_path.exists()
    lines = output_path.read_text().strip().splitlines()
    assert lines[0] == "no,filename,duration_s,text"
    assert lines[1] == "1,a.wav,1.5,transcript of a.wav"
    assert lines[2] == "2,b.wav,1.5,transcript of b.wav"


def test_transcribe_respects_custom_output_path(tmp_path, monkeypatch):
    config_path = _write_config(tmp_path)
    input_dir = tmp_path / "chunks"
    input_dir.mkdir()
    (input_dir / "a.wav").write_bytes(b"fake-audio")
    custom_output = tmp_path / "custom.csv"

    monkeypatch.setattr("jinpipe.orchestrator.detect_gpu_ids", lambda: [])
    monkeypatch.setattr(
        "jinpipe.stages.transcribe_only.transcribe_folder",
        lambda input_dir, asr_cfg, device: [{"no": 1, "filename": "a.wav", "duration_s": 1.5, "text": "hi"}],
    )

    result = runner.invoke(
        app,
        ["transcribe", "--config", str(config_path), "--input-dir", str(input_dir), "--output", str(custom_output)],
    )

    assert result.exit_code == 0
    assert custom_output.exists()


def test_transcribe_missing_input_dir_errors(tmp_path):
    config_path = _write_config(tmp_path)

    result = runner.invoke(app, ["transcribe", "--config", str(config_path), "--input-dir", str(tmp_path / "ghost")])

    assert result.exit_code != 0
    assert "not a directory" in result.output


def test_transcribe_no_audio_files_warns_without_writing_csv(tmp_path, monkeypatch):
    config_path = _write_config(tmp_path)
    input_dir = tmp_path / "chunks"
    input_dir.mkdir()

    monkeypatch.setattr("jinpipe.orchestrator.detect_gpu_ids", lambda: [])
    monkeypatch.setattr("jinpipe.stages.transcribe_only.transcribe_folder", lambda input_dir, asr_cfg, device: [])

    result = runner.invoke(app, ["transcribe", "--config", str(config_path), "--input-dir", str(input_dir)])

    assert result.exit_code == 0
    assert "No audio files found" in result.output
    assert not (tmp_path / "output" / "transcriptions.csv").exists()
