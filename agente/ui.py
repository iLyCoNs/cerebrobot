from __future__ import annotations

import json
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from . import config
from .agent import Brain
from .llm import LLM
from .memory import Memory

STATIC = Path(__file__).resolve().parent / "static" / "index.html"

STATE: dict = {"job": None, "brain": None, "memory": None, "lock": threading.Lock()}


class Job:
    def __init__(self, job_id: int) -> None:
        self.id = job_id
        self.log: list = []
        self.lock = threading.Lock()
        self.done = False

    def append(self, event: dict) -> None:
        with self.lock:
            self.log.append(event)

    def snapshot(self, since: int) -> dict:
        with self.lock:
            return {"events": self.log[since:], "done": self.done}


def _worker(job: Job, text: str) -> None:
    brain: Brain = STATE["brain"]

    def ev(kind: str, payload: dict) -> None:
        if kind == "answer":
            if brain._depth > 0:
                job.append({"type": "answer", "steps": payload.get("steps")})
        elif kind == "tool":
            job.append(
                {
                    "type": "tool",
                    "tool": payload.get("tool"),
                    "args": payload.get("args"),
                    "result": payload.get("result"),
                }
            )

    def on_retry(**kw) -> None:
        job.append(
            {
                "type": "retry",
                "model": kw.get("model"),
                "key": kw.get("key"),
                "status": kw.get("status"),
                "wait": kw.get("wait"),
            }
        )

    brain.on_event = ev
    brain.llm._rotator.on_retry = on_retry
    try:
        answer = brain.run(text)
        STATE["memory"].remember_turn(text, answer)
        job.append({"type": "final", "content": answer})
    except Exception as exc:
        job.append({"type": "error", "content": str(exc)})
    finally:
        brain.on_event = None
        brain.llm._rotator.on_retry = None
        job.append({"type": "done"})
        job.done = True


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:
        pass

    def _send(self, code: int, body: str, ctype: str) -> None:
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            self._send(200, STATIC.read_text(encoding="utf-8"), "text/html; charset=utf-8")
        elif path == "/api/status":
            brain: Brain = STATE["brain"]
            rot_status = brain.llm._rotator.status()
            payload = {
                "models": list(brain.llm._rotator.models),
                "keys": rot_status.get("keys", []),
                "models_cooling": rot_status.get("models_cooling", {}),
                "memory_summary": STATE["memory"].summary() or "(sin memoria previa)",
                "busy": bool(STATE["job"] and not STATE["job"].done),
            }
            self._send(200, json.dumps(payload, ensure_ascii=False), "application/json; charset=utf-8")
        elif path == "/api/job":
            if STATE["job"] is None:
                self._send(400, '{"error": "sin tarea activa"}', "application/json")
                return
            since = 0
            for part in parsed.query.split("&"):
                if part.startswith("since="):
                    try:
                        since = max(0, int(part[6:]))
                    except ValueError:
                        since = 0
            self._send(
                200,
                json.dumps(STATE["job"].snapshot(since), ensure_ascii=False),
                "application/json; charset=utf-8",
            )
        else:
            self._send(404, '{"error": "no encontrado"}', "application/json")

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0) or 0)
        body = {}
        if length:
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except Exception:
                body = {}

        if path == "/api/chat":
            text = str(body.get("message", "")).strip()
            if not text:
                self._send(400, '{"error": "mensaje vacio"}', "application/json")
                return
            with STATE["lock"]:
                if STATE["job"] and not STATE["job"].done:
                    self._send(409, '{"error": "hay una tarea en curso"}', "application/json")
                    return
                job = Job(int(time.time() * 1000))
                STATE["job"] = job
            threading.Thread(target=_worker, args=(job, text), daemon=True).start()
            self._send(200, json.dumps({"job": job.id}), "application/json")
        elif path == "/api/memory/clear":
            memory: Memory = STATE["memory"]
            memory.data["history"] = []
            memory.data["preferences"] = []
            memory.data["cache"] = {}
            memory.save()
            self._send(200, '{"ok": true}', "application/json")
        else:
            self._send(404, '{"error": "no encontrado"}', "application/json")


def main(port: int = 8000) -> None:
    STATE["memory"] = Memory(config.MEMORY_FILE)
    STATE["brain"] = Brain(LLM(), memory=STATE["memory"])
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}"
    print(f"CEREBRO · {url}")
    print("Ctrl+C para salir")
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8000)
