# JinPipe

Async, multi-worker pipeline that turns YouTube channels/videos into
TTS-training segments: scrape with `yt-dlp`, segment with Silero-VAD,
transcribe with WhisperX, and pack into sentence-safe audio+JSON segments.

Built as a lighter-weight, YouTube-native alternative to
[Emilia-Pipe](https://github.com/open-mmlab/Amphion/tree/main/preprocessors/Emilia),
with two hard guarantees Emilia-Pipe doesn't make:

- **Output segments never cut mid-sentence.** Cuts only ever land at a word
  boundary between two sentences (see [Sentence-safe rechunking](#sentence-safe-rechunking)).
- **CPU/GPU memory use is actively bounded**, not just implicitly limited by
  concurrency counts (see [Memory / OOM control](#memory--oom-control)).
- **Packaged output audio is not downsampled by the pipeline itself** - the
  16kHz-mono copy VAD/ASR need is kept internal-only (see
  [Output audio quality](#output-audio-quality)).

## Pipeline

```
discover (yt-dlp) -> download (yt-dlp) -> standardize (ffmpeg)
  -> coarse VAD segmentation (Silero-VAD, overlapping super-chunks)
  -> ASR + optional diarization (WhisperX, persistent worker pool)
  -> sentence-safe rechunk (pure logic, per-video barrier)
  -> quality filter (DNSMOS + duration/language)
  -> package (flac + JSON + manifest)
```

The discover/download steps are YouTube-specific and can be swapped out
entirely for a local folder of existing audio files - see
[Choosing a source](#choosing-a-source-youtube-vs-a-local-folder) below.

A single asyncio event loop is the only scheduler and the only SQLite writer
(`jinpipe/db.py`). Worker processes - a `ProcessPoolExecutor` for stateless
CPU stages and a small fixed pool of persistent model-loaded processes for
ASR/diarization (`jinpipe/workers/asr_worker.py`) - only ever return plain
data; the main loop is what persists it. This avoids cross-process SQLite
lock contention entirely instead of retrying around it.

### Sentence-safe rechunking

VAD super-chunks overlap by design (`vad.overlap_s`), because VAD sometimes
has to force a split inside a long, pause-free speech run once it exceeds
`max_superchunk_s`. `jinpipe/stages/rechunk.py` merges word-timestamp streams
across a video's super-chunks, de-duplicating the overlap (preferring
whichever occurrence sits farther from its own chunk's edge, since alignment
is least reliable near chunk boundaries), then splits on sentence-final
punctuation - falling back to the longest inter-word silence gap for
punctuation-less languages. A single sentence longer than `max_segment_s` is
kept whole and tagged `exceeds_max_duration: true` rather than ever being cut
mid-sentence.

### Output audio quality

VAD (Silero) and ASR (WhisperX/Whisper) both require 16kHz mono input, so
`standardize.sample_rate` (default `16000`) controls a separate, *internal*
working copy of each video used only for VAD/ASR/diarization. Packaged output
segments are sliced straight from the original source file instead - not from
that internal copy - so by default they keep the source's native sample rate
and channel count untouched: a 48kHz stereo source produces 48kHz stereo
`.flac` segments, capped only by whatever quality the source (or, for
YouTube, yt-dlp's `download.audio_format`) already had.

Set `package.sample_rate`/`package.channels` explicitly (e.g. `22050`/`1`)
only if you want smaller files and are fine trading quality for it - leave
both `null` (the default) to preserve source quality.

### Memory / OOM control

`jinpipe/resources.py`'s `ResourceGate` gates admission into every stage with
two independent checks: an in-process reservation budget (estimated cost =
audio duration x a per-stage calibrated constant) that catches bursts of
similarly-sized tasks before the OS reports any pressure, and a live
`psutil`/`pynvml` floor check that catches estimation error or memory
pressure from outside the pipeline. `vad.max_superchunk_s` bounds worst-case
per-task audio length, which is what makes the cost estimate tractable.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"                 # core + tests, no heavy ML deps
pip install -e ".[vad,asr,filter,gpu]"  # add these on the box that actually runs the pipeline
```

- `vad`: `silero-vad` (+ CPU torch) - tiny, fine to install anywhere.
- `asr`: `whisperx` (+ torch/torchaudio) - heavy; a CUDA GPU is strongly
  recommended for `large-v3`. Diarization additionally needs a Hugging Face
  token with access to `pyannote/speaker-diarization` (`asr.hf_token`).
- `filter`: `onnxruntime` + `librosa` for DNSMOS scoring. You must separately
  obtain `sig_bak_ovr.onnx` from the
  [DNS-Challenge repo](https://github.com/microsoft/DNS-Challenge) (its
  license doesn't allow bundling it here) and point
  `filter.dnsmos_model_path` at it - or set `filter.min_dnsmos_ovr: null` to
  skip DNSMOS scoring entirely and keep just the duration/language gates.
- `gpu`: `pynvml` for VRAM introspection when a CUDA GPU is present.

## Configure

Copy `configs/default.yaml`, edit `sources`/`paths`, and tune the per-stage
worker counts and `resources.*` floors/budgets for your box.

### Choosing a source: YouTube vs. a local folder

`sources` supports two mutually exclusive modes. Which one runs is decided
by a single field: **is `sources.local_dir` set or not.**

**YouTube mode** (the default) - discover via `channels` and/or list specific
videos in `video_urls`, then download each with `yt-dlp`:

```yaml
sources:
  channels:
    - "https://www.youtube.com/@example/videos"
  video_urls:
    - "https://www.youtube.com/watch?v=xxxxxxxxxxx"
  local_dir: null   # must stay null/omitted for YouTube mode to run
```

**Local-folder mode** - skip YouTube discovery/download entirely and process
audio files already on disk (e.g. a 1-hour interview recording):

```yaml
sources:
  channels: []        # ignored - fine to leave populated too, see note below
  video_urls: []       # ignored - fine to leave populated too, see note below
  local_dir: "./local_audio"   # every audio file directly inside this folder is processed
```

You do **not** need to empty `channels`/`video_urls` for local-folder mode to
work - the orchestrator checks `local_dir` first, and if it's set, YouTube
discovery/download never runs regardless of what's in `channels`/`video_urls`
(it logs a warning at startup if it finds both populated, just so it's
obvious which mode actually ran). Conversely, leave `local_dir: null`
(the default) to get ordinary YouTube mode.

`local_dir` is scanned non-recursively for `.mp3`, `.wav`, `.m4a`, `.flac`,
`.ogg`, `.opus`, `.aac`, `.mp4`, and `.webm` files
(`jinpipe/stages/local.py`); each file is fed straight into standardization
under a `video_id` derived from its filename, so file names should be unique
within the folder. Everything downstream (VAD, ASR/diarization, rechunk,
filter, package, resume) behaves identically to YouTube mode from there on.

```bash
jinpipe check-config --config my-config.yaml
```

## Run

```bash
jinpipe run --config my-config.yaml -v
jinpipe status --config my-config.yaml     # progress counts per stage/status
jinpipe resume --config my-config.yaml -v  # after a crash/interrupt
jinpipe manifest --config my-config.yaml   # rebuild manifest.jsonl from output_dir
jinpipe reset --config my-config.yaml      # clear job-store tracking state (see below)
```

Everything is resumable: the SQLite job store (`paths.db_path`) tracks each
video/super-chunk/segment's status, so `resume` picks up wherever a prior run
left off without re-downloading or re-transcribing completed work.

This cuts both ways: the job store and `output_dir`/`work_dir` are two
independent sources of truth that `jinpipe` assumes stay in sync. If you
manually delete files from `output_dir` (or `work_dir`) without also clearing
the corresponding job-store rows, `run`/`resume` will still see those videos
as `DONE` and silently skip them - nothing will be reprocessed even though
the output is gone. `jinpipe reset` clears job-store rows (never touches
files on disk) so the next run starts those videos over:

```bash
jinpipe reset --config my-config.yaml                     # everything, with a confirmation prompt
jinpipe reset --config my-config.yaml --video-id abc123   # just one video
jinpipe reset --config my-config.yaml --yes               # skip the prompt (scripting)
```

## What's tested where

`tests/` covers everything through the sentence-safe rechunk algorithm, the
job store, the resource gate, and the ASR worker pool's process
lifecycle/crash/stall handling - all with real (but faked/mocked) subprocess
and multiprocessing calls, runnable with no GPU and no heavy ML deps:

```bash
pytest
```

The WhisperX/pyannote model-loading path itself (`jinpipe/workers/asr_worker.py`'s
`_default_model_loader`/`_default_transcribe`) is written against WhisperX's
documented public API but can only be genuinely validated on a box with
`torch`/`whisperx`/a GPU (or patient CPU) installed - that's out of scope for
this repo's local test suite.

## A note on scope

Scraping and reprocessing YouTube audio for training data has copyright/ToS
implications depending on what you do with the output - worth confirming
your use case before scaling channel coverage up.
