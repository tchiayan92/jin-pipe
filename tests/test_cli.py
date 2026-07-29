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
