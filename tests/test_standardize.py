import shutil
import subprocess
import wave

import pytest

from jinpipe.stages.standardize import standardize_audio


def test_standardize_audio_raises_on_ffmpeg_failure(tmp_path):
    def failing_runner(args):
        return subprocess.CompletedProcess(args, returncode=1, stdout=b"", stderr=b"boom")

    with pytest.raises(RuntimeError, match="ffmpeg standardize failed"):
        standardize_audio(tmp_path / "in.m4a", tmp_path / "out.wav", runner=failing_runner)


def test_standardize_audio_builds_expected_ffmpeg_args(tmp_path):
    captured = {}

    def fake_runner(args):
        captured["args"] = args
        (tmp_path / "out.wav").write_bytes(b"RIFF....")
        return subprocess.CompletedProcess(args, returncode=0, stdout=b"", stderr=b"")

    out = standardize_audio(tmp_path / "in.m4a", tmp_path / "out.wav", sample_rate=16000, runner=fake_runner)
    assert out == tmp_path / "out.wav"
    args = captured["args"]
    assert args[0] == "ffmpeg"
    assert args[args.index("-ar") + 1] == "16000"
    assert args[args.index("-ac") + 1] == "1"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_standardize_audio_real_ffmpeg_roundtrip(tmp_path):
    raw = tmp_path / "tone.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-ar",
            "44100",
            "-ac",
            "2",
            str(raw),
        ],
        check=True,
        capture_output=True,
    )
    out = tmp_path / "standardized.wav"
    standardize_audio(raw, out, sample_rate=16000)
    assert out.exists()
    with wave.open(str(out), "rb") as wf:
        assert wf.getframerate() == 16000
        assert wf.getnchannels() == 1
