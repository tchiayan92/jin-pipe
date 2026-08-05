import shutil
import subprocess
import wave

import pytest

from jinpipe.stages.standardize import probe_duration, standardize_audio


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


def test_probe_duration_returns_none_on_ffprobe_failure(tmp_path):
    def failing_runner(args):
        return subprocess.CompletedProcess(args, returncode=1, stdout=b"", stderr=b"boom")

    assert probe_duration(tmp_path / "missing.wav", runner=failing_runner) is None


def test_probe_duration_parses_ffprobe_output(tmp_path):
    def fake_runner(args):
        return subprocess.CompletedProcess(args, returncode=0, stdout=b"12.345000\n", stderr=b"")

    assert probe_duration(tmp_path / "out.wav", runner=fake_runner) == pytest.approx(12.345)


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg/ffprobe not installed"
)
def test_probe_duration_real_ffprobe_matches_standardized_wav_length(tmp_path):
    # A source container's own duration metadata (checked here via a 44.1kHz
    # stereo wav standing in for a compressed container) can differ subtly
    # from the standardized wav's true decoded length - probe_duration must
    # read the STANDARDIZED file's own header, which is sample-exact.
    raw = tmp_path / "tone.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
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
    duration = probe_duration(out)
    assert duration == pytest.approx(2.0, abs=0.05)
