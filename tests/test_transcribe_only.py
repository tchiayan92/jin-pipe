import csv

from jinpipe.config import AsrConfig
from jinpipe.stages.transcribe_only import transcribe_folder, write_csv


def _fake_transcribe(model, path, cfg):
    return {"text": f"transcript of {path.name}", "duration_s": 1.5}


def test_transcribe_folder_numbers_and_transcribes_each_file_in_sorted_order(tmp_path):
    (tmp_path / "b.wav").write_bytes(b"fake-audio")
    (tmp_path / "a.wav").write_bytes(b"fake-audio")
    (tmp_path / "not_audio.txt").write_text("ignore me")

    rows = transcribe_folder(tmp_path, AsrConfig(), model="fake-model", transcribe_fn=_fake_transcribe)

    assert rows == [
        {"no": 1, "filename": "a.wav", "duration_s": 1.5, "text": "transcript of a.wav"},
        {"no": 2, "filename": "b.wav", "duration_s": 1.5, "text": "transcript of b.wav"},
    ]


def test_transcribe_folder_empty_dir_returns_no_rows(tmp_path):
    rows = transcribe_folder(tmp_path, AsrConfig(), model="fake-model", transcribe_fn=_fake_transcribe)
    assert rows == []


def test_transcribe_folder_loads_model_only_when_not_provided(tmp_path, monkeypatch):
    (tmp_path / "a.wav").write_bytes(b"fake-audio")
    calls = []
    monkeypatch.setattr(
        "jinpipe.stages.transcribe_only.load_model",
        lambda cfg, device: calls.append(device) or "loaded-model",
    )

    rows = transcribe_folder(tmp_path, AsrConfig(), device="cpu", transcribe_fn=_fake_transcribe)

    assert calls == ["cpu"]
    assert rows == [{"no": 1, "filename": "a.wav", "duration_s": 1.5, "text": "transcript of a.wav"}]


def test_write_csv_writes_header_and_rows_in_field_order(tmp_path):
    rows = [
        {"no": 1, "filename": "a.wav", "duration_s": 1.5, "text": "hello"},
        {"no": 2, "filename": "b.wav", "duration_s": 2.25, "text": "world, comma"},
    ]
    output_path = tmp_path / "out" / "transcriptions.csv"

    write_csv(rows, output_path)

    with output_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ["no", "filename", "duration_s", "text"]
        read_rows = list(reader)
    assert read_rows == [
        {"no": "1", "filename": "a.wav", "duration_s": "1.5", "text": "hello"},
        {"no": "2", "filename": "b.wav", "duration_s": "2.25", "text": "world, comma"},
    ]


def test_write_csv_with_no_rows_still_writes_header(tmp_path):
    output_path = tmp_path / "transcriptions.csv"
    write_csv([], output_path)
    assert output_path.read_text().strip() == "no,filename,duration_s,text"
