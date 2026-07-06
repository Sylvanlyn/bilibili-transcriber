# Bilibili Transcriber

A local Bilibili transcription tool. It downloads a video's audio track, converts
it with `ffmpeg`, and transcribes it locally with `faster-whisper`.

Use it only for content you are allowed to download/transcribe, and respect the
website's terms and creators' rights.

## Features

- Web UI for entering a Bilibili video link and starting transcription.
- CLI for scripting or batch usage.
- Outputs plain text, SRT subtitles, and structured JSON.
- Caches Whisper models once under `.models/`.
- Writes generated transcripts under `outputs/`.

## Requirements

- Python 3.9+
- `ffmpeg`

Install `ffmpeg` on macOS:

```bash
brew install ffmpeg
```

Install Python dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

## Web UI

```bash
python web_app.py
```

Open:

```text
http://127.0.0.1:8787
```

Paste a Bilibili link, choose a model, and start conversion.

## CLI

```bash
python bilibili_transcribe.py "https://www.bilibili.com/video/BV1D5Ln67EkJ/"
```

By default, output is written to:

```text
outputs/<video title>/
```

Generated files:

- `transcript_raw.txt` - plain text, one recognized segment per line
- `transcript.srt` - subtitle format with indexes and timestamps
- `transcript_segments.json` - structured segment data
- `source_audio_16k.wav` - converted audio used for transcription

Use a custom output directory:

```bash
python bilibili_transcribe.py "https://www.bilibili.com/video/BV1D5Ln67EkJ/" --out my_transcript
```

Transcribe a local media file:

```bash
python bilibili_transcribe.py ./video.mp4 --local
```

## Models

The default model is `small`.

```bash
python bilibili_transcribe.py "https://www.bilibili.com/video/BV1D5Ln67EkJ/" --model medium
```

`medium` is more accurate but slower and downloads a larger model.

Model cache:

```text
.models/whisper
.models/hf_cache
```

These directories are ignored by git.

## Notes

If Hugging Face downloads are unstable, you can try `--hf-mirror`, but recent
`huggingface_hub` versions may fail on mirror redirects. The script currently
falls back to the official Hub in that case.
