#!/usr/bin/env python3
"""
Local web UI for Bilibili transcription.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from bilibili_transcribe import (
    DEFAULT_OUTPUT_ROOT,
    download_file,
    extract_bvid,
    get_audio_url,
    get_video_info,
    run_ffmpeg,
    sanitize_filename,
    transcribe,
)


TASKS: dict[str, dict] = {}
TASKS_LOCK = threading.Lock()
OUTPUT_FILES = {
    "txt": "transcript_raw.txt",
    "srt": "transcript.srt",
    "json": "transcript_segments.json",
    "audio": "source_audio_16k.wav",
}


INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Bilibili Transcriber</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f8;
      --panel: #ffffff;
      --line: #d9dee5;
      --text: #171a1f;
      --muted: #667085;
      --accent: #0f766e;
      --accent-strong: #0b5f59;
      --danger: #b42318;
      --soft: #eef7f6;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
    }
    main {
      width: min(1120px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 28px 0 36px;
    }
    .topbar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      margin-bottom: 18px;
    }
    h1 {
      font-size: 24px;
      line-height: 1.2;
      margin: 0;
      font-weight: 720;
      letter-spacing: 0;
    }
    .status-pill {
      min-width: 108px;
      padding: 7px 10px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: var(--panel);
      color: var(--muted);
      text-align: center;
      font-size: 13px;
      white-space: nowrap;
    }
    .layout {
      display: grid;
      grid-template-columns: minmax(0, 420px) minmax(0, 1fr);
      gap: 18px;
      align-items: start;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
    }
    label {
      display: block;
      margin-bottom: 8px;
      color: #303642;
      font-size: 13px;
      font-weight: 650;
    }
    textarea, select {
      width: 100%;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--text);
      border-radius: 7px;
      font: inherit;
      outline: none;
    }
    textarea {
      min-height: 112px;
      resize: vertical;
      padding: 12px;
      line-height: 1.5;
    }
    select {
      height: 42px;
      padding: 0 10px;
    }
    textarea:focus, select:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(15, 118, 110, .14);
    }
    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-top: 14px;
    }
    .checkline {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-top: 14px;
      color: var(--muted);
      font-size: 13px;
    }
    .checkline input { width: 16px; height: 16px; }
    .actions {
      display: flex;
      gap: 10px;
      margin-top: 16px;
    }
    button, a.file-link {
      height: 42px;
      border: 1px solid transparent;
      border-radius: 7px;
      padding: 0 14px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      font: inherit;
      font-weight: 680;
      text-decoration: none;
      cursor: pointer;
    }
    button.primary {
      background: var(--accent);
      color: white;
      flex: 1;
    }
    button.primary:hover { background: var(--accent-strong); }
    button.secondary {
      background: #fff;
      border-color: var(--line);
      color: var(--text);
    }
    button:disabled {
      opacity: .62;
      cursor: not-allowed;
    }
    .steps {
      display: grid;
      gap: 10px;
      margin-top: 16px;
    }
    .step {
      display: flex;
      gap: 10px;
      align-items: center;
      color: var(--muted);
      font-size: 14px;
    }
    .dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      border: 1px solid var(--line);
      background: #fff;
      flex: none;
    }
    .step.active { color: var(--text); font-weight: 650; }
    .step.active .dot { background: var(--accent); border-color: var(--accent); }
    .step.done .dot { background: #83c5be; border-color: #83c5be; }
    .result-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      margin-bottom: 12px;
    }
    .title-block {
      min-width: 0;
    }
    .video-title {
      margin: 0;
      font-size: 16px;
      font-weight: 720;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .output-dir {
      margin-top: 4px;
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
    }
    .file-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 14px;
    }
    a.file-link {
      height: 36px;
      background: var(--soft);
      border-color: #c8e5e1;
      color: #0b5f59;
      font-size: 13px;
    }
    pre {
      margin: 0;
      min-height: 420px;
      max-height: 62vh;
      overflow: auto;
      padding: 14px;
      border-radius: 7px;
      border: 1px solid var(--line);
      background: #fbfcfd;
      line-height: 1.65;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 13px;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .error {
      color: var(--danger);
      background: #fff1f0;
      border: 1px solid #fecdca;
      padding: 10px 12px;
      border-radius: 7px;
      margin-top: 12px;
      display: none;
      font-size: 13px;
      line-height: 1.45;
    }
    @media (max-width: 860px) {
      main { width: min(100vw - 20px, 680px); padding-top: 16px; }
      .layout { grid-template-columns: 1fr; }
      .row { grid-template-columns: 1fr; }
      .topbar { align-items: flex-start; }
      pre { min-height: 320px; }
    }
  </style>
</head>
<body>
  <main>
    <div class="topbar">
      <h1>Bilibili Transcriber</h1>
      <div class="status-pill" id="statusPill">空闲</div>
    </div>
    <div class="layout">
      <section class="panel">
        <form id="form">
          <label for="url">视频链接</label>
          <textarea id="url" autocomplete="off" spellcheck="false" placeholder="https://www.bilibili.com/video/BV..."></textarea>
          <div class="row">
            <div>
              <label for="model">模型</label>
              <select id="model">
                <option value="small">small</option>
                <option value="medium" selected>medium</option>
                <option value="base">base</option>
                <option value="tiny">tiny</option>
              </select>
            </div>
            <div>
              <label for="format">预览</label>
              <select id="format">
                <option value="txt">TXT</option>
                <option value="srt">SRT</option>
                <option value="json">JSON</option>
              </select>
            </div>
          </div>
          <label class="checkline">
            <input type="checkbox" id="keepSource">
            保留原始音频
          </label>
          <div class="actions">
            <button class="primary" id="startButton" type="submit">▶ 开始转换</button>
            <button class="secondary" id="clearButton" type="button">清空</button>
          </div>
          <div class="steps" id="steps">
            <div class="step" data-step="resolving"><span class="dot"></span><span>解析视频</span></div>
            <div class="step" data-step="downloading"><span class="dot"></span><span>下载音频</span></div>
            <div class="step" data-step="converting"><span class="dot"></span><span>转换音频</span></div>
            <div class="step" data-step="transcribing"><span class="dot"></span><span>识别文本</span></div>
            <div class="step" data-step="done"><span class="dot"></span><span>完成</span></div>
          </div>
          <div class="error" id="errorBox"></div>
        </form>
      </section>
      <section class="panel">
        <div class="result-head">
          <div class="title-block">
            <p class="video-title" id="videoTitle">等待输入链接</p>
            <div class="output-dir" id="outputDir"></div>
          </div>
        </div>
        <div class="file-actions" id="fileActions"></div>
        <pre id="preview"></pre>
      </section>
    </div>
  </main>
  <script>
    const form = document.querySelector('#form');
    const urlInput = document.querySelector('#url');
    const modelInput = document.querySelector('#model');
    const formatInput = document.querySelector('#format');
    const keepSourceInput = document.querySelector('#keepSource');
    const startButton = document.querySelector('#startButton');
    const clearButton = document.querySelector('#clearButton');
    const statusPill = document.querySelector('#statusPill');
    const errorBox = document.querySelector('#errorBox');
    const preview = document.querySelector('#preview');
    const fileActions = document.querySelector('#fileActions');
    const videoTitle = document.querySelector('#videoTitle');
    const outputDir = document.querySelector('#outputDir');
    const steps = Array.from(document.querySelectorAll('.step'));
    let currentTaskId = null;
    let pollTimer = null;

    function setError(message) {
      errorBox.textContent = message || '';
      errorBox.style.display = message ? 'block' : 'none';
    }

    function setStage(stage) {
      const order = ['resolving', 'downloading', 'converting', 'transcribing', 'done'];
      const index = order.indexOf(stage);
      steps.forEach((step) => {
        const stepIndex = order.indexOf(step.dataset.step);
        step.classList.toggle('active', step.dataset.step === stage);
        step.classList.toggle('done', index > stepIndex && index !== -1);
      });
    }

    function setBusy(isBusy) {
      startButton.disabled = isBusy;
      urlInput.disabled = isBusy;
      modelInput.disabled = isBusy;
      keepSourceInput.disabled = isBusy;
      startButton.textContent = isBusy ? '处理中' : '▶ 开始转换';
    }

    function renderFiles(task) {
      fileActions.innerHTML = '';
      if (!task.files) return;
      for (const [key, file] of Object.entries(task.files)) {
        const link = document.createElement('a');
        link.className = 'file-link';
        link.href = `/api/tasks/${task.id}/files/${encodeURIComponent(key)}`;
        link.textContent = file.label;
        link.download = file.name;
        fileActions.appendChild(link);
      }
    }

    async function loadPreview(taskId, format) {
      const response = await fetch(`/api/tasks/${taskId}/files/${encodeURIComponent(format)}`);
      if (!response.ok) return;
      const text = await response.text();
      preview.textContent = text;
    }

    async function pollTask(taskId) {
      const response = await fetch(`/api/tasks/${taskId}`);
      const task = await response.json();
      currentTaskId = task.id;
      statusPill.textContent = task.message || task.status;
      setStage(task.stage || task.status);
      videoTitle.textContent = task.title || '处理中';
      outputDir.textContent = task.output_dir || '';
      if (task.status === 'failed') {
        setBusy(false);
        setError(task.error || '转换失败');
        clearInterval(pollTimer);
        pollTimer = null;
        return;
      }
      if (task.status === 'done') {
        setBusy(false);
        setError('');
        renderFiles(task);
        clearInterval(pollTimer);
        pollTimer = null;
        await loadPreview(task.id, formatInput.value);
      }
    }

    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      const url = urlInput.value.trim();
      if (!url) {
        setError('请输入视频链接');
        return;
      }
      setError('');
      preview.textContent = '';
      fileActions.innerHTML = '';
      videoTitle.textContent = '处理中';
      outputDir.textContent = '';
      setBusy(true);
      setStage('resolving');
      statusPill.textContent = '排队';
      if (pollTimer) clearInterval(pollTimer);
      try {
        const response = await fetch('/api/transcribe', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            url,
            model: modelInput.value,
            keep_source: keepSourceInput.checked
          })
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || '提交失败');
        currentTaskId = payload.id;
        pollTimer = setInterval(() => pollTask(payload.id), 1400);
        await pollTask(payload.id);
      } catch (error) {
        setBusy(false);
        setError(error.message);
      }
    });

    clearButton.addEventListener('click', () => {
      urlInput.value = '';
      preview.textContent = '';
      fileActions.innerHTML = '';
      videoTitle.textContent = '等待输入链接';
      outputDir.textContent = '';
      statusPill.textContent = '空闲';
      setError('');
      setStage('');
    });

    formatInput.addEventListener('change', () => {
      if (currentTaskId) loadPreview(currentTaskId, formatInput.value);
    });
  </script>
</body>
</html>
"""


def set_task(task_id: str, **updates: object) -> None:
    with TASKS_LOCK:
        task = TASKS.setdefault(task_id, {})
        task.update(updates)
        task["updated_at"] = time.time()


def get_task(task_id: str) -> dict | None:
    with TASKS_LOCK:
        task = TASKS.get(task_id)
        return dict(task) if task else None


def output_files_for(task_id: str, out_dir: Path) -> dict[str, dict[str, str]]:
    files = {}
    for key, name in OUTPUT_FILES.items():
        path = out_dir / name
        if path.exists():
            files[key] = {
                "name": name,
                "label": name,
                "url": f"/api/tasks/{task_id}/files/{key}",
            }
    return files


def run_task(task_id: str, source: str, model: str, keep_source: bool) -> None:
    try:
        set_task(task_id, status="running", stage="resolving", message="解析视频")
        bvid = extract_bvid(source)
        bvid, cid, title = get_video_info(bvid)
        safe_title = sanitize_filename(title)
        out_dir = DEFAULT_OUTPUT_ROOT / safe_title
        out_dir.mkdir(parents=True, exist_ok=True)
        set_task(
            task_id,
            bvid=bvid,
            cid=cid,
            title=title,
            output_dir=str(out_dir),
        )

        source_path = out_dir / f"{safe_title}_{bvid}.m4s"
        wav_path = out_dir / "source_audio_16k.wav"

        set_task(task_id, stage="downloading", message="下载音频")
        audio_url = get_audio_url(bvid, cid)
        download_file(audio_url, source_path, f"https://www.bilibili.com/video/{bvid}/")

        set_task(task_id, stage="converting", message="转换音频")
        run_ffmpeg(source_path, wav_path)
        if not keep_source:
            source_path.unlink(missing_ok=True)

        set_task(task_id, stage="transcribing", message=f"识别文本 · {model}")
        transcribe_args = SimpleNamespace(
            model=model,
            language="zh",
            device="cpu",
            compute_type="int8",
            beam_size=5,
            no_vad=False,
            min_silence_ms=500,
            hf_mirror=False,
            initial_prompt="以下是普通话视频口播内容，请使用简体中文标点。",
            keep_source=keep_source,
        )
        transcribe(wav_path, transcribe_args, out_dir)

        set_task(
            task_id,
            status="done",
            stage="done",
            message="完成",
            files=output_files_for(task_id, out_dir),
        )
    except Exception as exc:
        set_task(
            task_id,
            status="failed",
            stage="failed",
            message="失败",
            error=str(exc),
        )


class AppHandler(BaseHTTPRequestHandler):
    server_version = "BilibiliTranscriber/1.0"

    def log_message(self, format: str, *args: object) -> None:
        return

    def send_json(self, payload: object, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_text(self, text: str, content_type: str = "text/html; charset=utf-8") -> None:
        body = text.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            self.send_text(INDEX_HTML)
            return

        parts = [urllib.parse.unquote(part) for part in parsed.path.split("/") if part]
        if len(parts) == 3 and parts[:2] == ["api", "tasks"]:
            task = get_task(parts[2])
            if not task:
                self.send_json({"error": "Task not found"}, HTTPStatus.NOT_FOUND)
                return
            self.send_json(task)
            return

        if len(parts) == 5 and parts[:2] == ["api", "tasks"] and parts[3] == "files":
            task = get_task(parts[2])
            file_key = parts[4]
            if not task:
                self.send_json({"error": "Task not found"}, HTTPStatus.NOT_FOUND)
                return
            if file_key not in OUTPUT_FILES:
                self.send_json({"error": "File not allowed"}, HTTPStatus.NOT_FOUND)
                return
            out_dir = Path(task.get("output_dir", ""))
            path = out_dir / OUTPUT_FILES[file_key]
            if not path.exists():
                self.send_json({"error": "File not found"}, HTTPStatus.NOT_FOUND)
                return
            content_type = "text/plain; charset=utf-8"
            if path.suffix == ".json":
                content_type = "application/json; charset=utf-8"
            elif path.suffix == ".srt":
                content_type = "application/x-subrip; charset=utf-8"
            elif path.suffix == ".wav":
                content_type = "audio/wav"
            body = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/api/transcribe":
            self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            source = str(data.get("url", "")).strip()
            model = str(data.get("model", "small")).strip() or "small"
            keep_source = bool(data.get("keep_source", False))
            if not source:
                self.send_json({"error": "Missing video URL"}, HTTPStatus.BAD_REQUEST)
                return
            if model not in {"tiny", "base", "small", "medium"}:
                self.send_json({"error": "Unsupported model"}, HTTPStatus.BAD_REQUEST)
                return

            task_id = uuid4().hex
            set_task(
                task_id,
                id=task_id,
                status="queued",
                stage="resolving",
                message="排队",
                title="",
                output_dir="",
                files={},
                created_at=time.time(),
            )
            thread = threading.Thread(
                target=run_task,
                args=(task_id, source, model, keep_source),
                daemon=True,
            )
            thread.start()
            self.send_json({"id": task_id}, HTTPStatus.ACCEPTED)
        except json.JSONDecodeError:
            self.send_json({"error": "Invalid JSON"}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local transcription web UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), AppHandler)
    print(f"Open http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
