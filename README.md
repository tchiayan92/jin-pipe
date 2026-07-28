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

## Pipeline

```
discover (yt-dlp) -> download (yt-dlp) -> standardize (ffmpeg)
  -> coarse VAD segmentation (Silero-VAD, overlapping super-chunks)
  -> ASR + optional diarization (WhisperX, persistent worker pool)
  -> sentence-safe rechunk (pure logic, per-video barrier)
  -> quality filter (DNSMOS + duration/language)
  -> package (flac + JSON + manifest)
```

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

```bash
jinpipe check-config --config my-config.yaml
```

## Run

```bash
jinpipe run --config my-config.yaml -v
jinpipe status --config my-config.yaml     # progress counts per stage/status
jinpipe resume --config my-config.yaml -v  # after a crash/interrupt
jinpipe manifest --config my-config.yaml   # rebuild manifest.jsonl from output_dir
```

Everything is resumable: the SQLite job store (`paths.db_path`) tracks each
video/super-chunk/segment's status, so `resume` picks up wherever a prior run
left off without re-downloading or re-transcribing completed work.

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
