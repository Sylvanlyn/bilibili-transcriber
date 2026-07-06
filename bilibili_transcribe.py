#!/usr/bin/env python3
"""
Download a Bilibili video's audio and transcribe it locally with faster-whisper.

Use this only for content you are allowed to download/transcribe, and respect
the website's terms and the creator's rights.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.bilibili.com/",
}

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_CACHE_DIR = PROJECT_ROOT / ".models" / "whisper"
HF_CACHE_DIR = PROJECT_ROOT / ".models" / "hf_cache"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"


def request_json(url: str, headers: dict[str, str] | None = None) -> dict:
    req = urllib.request.Request(url, headers=headers or HEADERS)
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)


def extract_bvid(source: str) -> str:
    match = re.search(r"(BV[0-9A-Za-z]+)", source)
    if not match:
        raise ValueError(f"Could not find a BV id in: {source}")
    return match.group(1)


def sanitize_filename(value: str) -> str:
    value = re.sub(r"[\\/:*?\"<>|]+", "_", value).strip()
    value = re.sub(r"\s+", " ", value)
    return value[:80] or "bilibili_video"


def get_video_info(bvid: str) -> tuple[str, int, str]:
    url = "https://api.bilibili.com/x/web-interface/view?" + urllib.parse.urlencode(
        {"bvid": bvid}
    )
    payload = request_json(url)
    if payload.get("code") != 0:
        raise RuntimeError(f"Bilibili view API failed: {payload}")
    data = payload["data"]
    cid = data["pages"][0]["cid"]
    title = data.get("title") or bvid
    return bvid, cid, title


def get_audio_url(bvid: str, cid: int) -> str:
    params = {
        "bvid": bvid,
        "cid": str(cid),
        "qn": "16",
        "fnval": "16",
        "fourk": "0",
    }
    url = "https://api.bilibili.com/x/player/playurl?" + urllib.parse.urlencode(params)
    headers = dict(HEADERS)
    headers["Referer"] = f"https://www.bilibili.com/video/{bvid}/"
    payload = request_json(url, headers=headers)
    if payload.get("code") != 0:
        raise RuntimeError(f"Bilibili playurl API failed: {payload}")
    audios = payload.get("data", {}).get("dash", {}).get("audio", [])
    if not audios:
        raise RuntimeError("No DASH audio stream found. The video may require login/cookies.")
    best = max(audios, key=lambda item: item.get("bandwidth", 0))
    return best.get("baseUrl") or best.get("base_url")


def download_file(url: str, output_path: Path, referer: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = dict(HEADERS)
    headers["Referer"] = referer
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=90) as response:
            with output_path.open("wb") as file:
                shutil.copyfileobj(response, file)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Download failed with HTTP {exc.code}: {exc.reason}") from exc


def run_ffmpeg(input_path: Path, wav_path: Path) -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is not installed or not on PATH.")
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(wav_path),
        ],
        check=True,
    )


def format_txt_timestamp(seconds: float) -> str:
    seconds = max(0, seconds)
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def format_srt_timestamp(seconds: float) -> str:
    seconds = max(0, seconds)
    millis = int(round(seconds * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_outputs(out_dir: Path, segments: list[dict], meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "transcript_segments.json").write_text(
        json.dumps({"meta": meta, "segments": segments}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with (out_dir / "transcript_raw.txt").open("w", encoding="utf-8") as file:
        for item in segments:
            file.write(f"{item['text']}\n")

    with (out_dir / "transcript.srt").open("w", encoding="utf-8") as file:
        for index, item in enumerate(segments, start=1):
            file.write(
                f"{index}\n"
                f"{format_srt_timestamp(item['start'])} --> "
                f"{format_srt_timestamp(item['end'])}\n"
                f"{item['text']}\n\n"
            )


def transcribe(audio_path: Path, args: argparse.Namespace, out_dir: Path) -> None:
    if args.hf_mirror:
        print(
            "Note: --hf-mirror is ignored because this huggingface_hub version "
            "fails on hf-mirror redirects; using the official Hugging Face Hub."
        )
        if os.environ.get("HF_ENDPOINT") == "https://hf-mirror.com":
            os.environ.pop("HF_ENDPOINT", None)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: faster-whisper. Install with "
            "`python -m pip install faster-whisper`."
        ) from exc

    model = WhisperModel(
        args.model,
        device=args.device,
        compute_type=args.compute_type,
        download_root=str(MODEL_CACHE_DIR),
    )
    segments_iter, info = model.transcribe(
        str(audio_path),
        language=args.language,
        beam_size=args.beam_size,
        vad_filter=not args.no_vad,
        vad_parameters={"min_silence_duration_ms": args.min_silence_ms},
        initial_prompt=args.initial_prompt,
    )

    segments = []
    for segment in segments_iter:
        text = segment.text.strip()
        if not text:
            continue
        item = {"start": segment.start, "end": segment.end, "text": text}
        segments.append(item)
        print(f"[{format_txt_timestamp(segment.start)}] {text}", flush=True)

    meta = {
        "language": info.language,
        "duration": info.duration,
        "model": args.model,
        "device": args.device,
        "compute_type": args.compute_type,
    }
    write_outputs(out_dir, segments, meta)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe Bilibili video audio locally with faster-whisper."
    )
    parser.add_argument(
        "source",
        help="Bilibili URL/BV id, or a local audio/video file when --local is used.",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Treat source as a local media file instead of a Bilibili URL.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory. Defaults to the video title, or local file name with --local.",
    )
    parser.add_argument("--model", default="small", help="Whisper model size/path.")
    parser.add_argument("--language", default="zh", help="ASR language code.")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or auto.")
    parser.add_argument("--compute-type", default="int8", help="int8, float16, float32.")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--min-silence-ms", type=int, default=500)
    parser.add_argument("--no-vad", action="store_true")
    parser.add_argument(
        "--hf-mirror",
        action="store_true",
        help="Try https://hf-mirror.com for model downloads; may fall back if incompatible.",
    )
    parser.add_argument(
        "--initial-prompt",
        default="以下是普通话视频口播内容，请使用简体中文标点。",
    )
    parser.add_argument(
        "--keep-source",
        action="store_true",
        help="Keep downloaded source audio after conversion.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        if args.local:
            source_path = Path(args.source).expanduser().resolve()
            if not source_path.exists():
                raise FileNotFoundError(source_path)
            out_dir = (
                Path(args.out).expanduser().resolve()
                if args.out
                else DEFAULT_OUTPUT_ROOT / sanitize_filename(source_path.stem)
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            wav_path = out_dir / "source_audio_16k.wav"
            print(f"Converting local media: {source_path}")
            run_ffmpeg(source_path, wav_path)
        else:
            bvid = extract_bvid(args.source)
            bvid, cid, title = get_video_info(bvid)
            safe_title = sanitize_filename(title)
            out_dir = (
                Path(args.out).expanduser().resolve()
                if args.out
                else DEFAULT_OUTPUT_ROOT / safe_title
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            source_path = out_dir / f"{safe_title}_{bvid}.m4s"
            wav_path = out_dir / "source_audio_16k.wav"
            print(f"Video: {title} ({bvid}), cid={cid}")
            audio_url = get_audio_url(bvid, cid)
            print("Downloading audio...")
            download_file(audio_url, source_path, f"https://www.bilibili.com/video/{bvid}/")
            print("Converting audio...")
            run_ffmpeg(source_path, wav_path)
            if not args.keep_source:
                source_path.unlink(missing_ok=True)

        print("Transcribing...")
        transcribe(wav_path, args, out_dir)
        print(f"Done. Files written to: {out_dir}")
        print(" - transcript_raw.txt")
        print(" - transcript.srt")
        print(" - transcript_segments.json")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
