"""Bot de Discord conectado al cerebro.

Gateway de Discord (websocket) + REST, sin dependencias externas mas alla de
websocket-client y requests (ya instalados). El bot ejecuta tareas con el
cerebro completo (LLM + herramientas) y responde en el chat.

Solo responde al DUEÑO configurado en DISCORD_OWNER_ID — nadie mas puede
ejecutar comandos en tu PC.

Configuracion en .env (raiz del proyecto):
    DISCORD_BOT_TOKEN=MTIz...token del portal de desarrolladores...
    DISCORD_OWNER_ID=123456789012345678  (tu ID de usuario de Discord)

Comandos (mencionando al bot o con prefijo !cerebro):
    !cerebro <tarea>   -> ejecuta la tarea con el cerebro completo
    !estado            -> estado de keys y modelos NVIDIA
    !ping              -> latencia

Uso:  python discord_bot.py
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import requests
import websocket

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agente.agent import Brain  # noqa: E402
from agente.llm import LLM  # noqa: E402
from agente.memory import Memory  # noqa: E402

API = "https://discord.com/api/v10"
INTENTS = 1 | 512 | 32768  # GUILDS | GUILD_MESSAGES | MESSAGE_CONTENT
PREFIX = "!cerebro "


def load_env() -> dict:
    env: dict = {}
    path = _ROOT / ".env"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            env[name.strip()] = value.strip().strip('"').strip("'")
    env.update({k: v for k, v in os.environ.items() if k.startswith("DISCORD_")})
    return env


class DiscordBot:
    def __init__(self, token: str, owner_id: str, auto_channel: str = "") -> None:
        self.token = token
        self.owner_id = str(owner_id)
        self.auto_channel = str(auto_channel)
        self.ws: websocket.WebSocket | None = None
        self.heartbeat_interval = 30.0
        self.last_heartbeat = 0.0
        self.sequence: int | None = None
        self.bot_user_id: str = ""
        self.memory = Memory("agente_memoria.json")
        self.brain = Brain(LLM(), memory=self.memory)
        self.busy = False

    # ------------------------------------------------------------ REST

    def _headers(self) -> dict:
        return {"Authorization": f"Bot {self.token}", "Content-Type": "application/json"}

    def send_message(self, channel_id: str, content: str) -> dict:
        r = requests.post(
            f"{API}/channels/{channel_id}/messages",
            headers=self._headers(),
            json={"content": content[:2000]},
            timeout=20,
        )
        if r.status_code == 429:
            time.sleep(float(r.json().get("retry_after", 1.0)))
            return self.send_message(channel_id, content)
        r.raise_for_status()
        return r.json()

    def edit_message(self, channel_id: str, message_id: str, content: str) -> None:
        r = requests.patch(
            f"{API}/channels/{channel_id}/messages/{message_id}",
            headers=self._headers(),
            json={"content": content[:2000]},
            timeout=20,
        )
        if r.status_code == 429:
            time.sleep(float(r.json().get("retry_after", 1.0)))

    def _chunk_reply(self, text: str, limit: int = 1900) -> list:
        text = str(text)
        chunks = []
        while len(text) > limit:
            cut = text.rfind("\n", 0, limit)
            if cut < limit // 2:
                cut = limit
            chunks.append(text[:cut])
            text = text[cut:].lstrip("\n")
        chunks.append(text)
        return chunks

    # ------------------------------------------------------------ cerebro

    def run_task(self, channel_id: str, task: str) -> None:
        if self.busy:
            self.edit_message(channel_id, self._status_id, "ya hay una tarea en curso, espera.")
            return
        self.busy = True
        status = self.send_message(channel_id, "🧠 pensando · consultando nvidia…")
        self._status_id = status["id"]
        job_events: list = []
        self.brain.on_event = lambda kind, payload: job_events.append((kind, payload))

        def worker() -> None:
            try:
                answer = self.brain.run(task)
                self.memory.remember_turn(task, answer)
                parts = self._chunk_reply(answer)
                self.edit_message(channel_id, self._status_id, parts[0])
                for extra in parts[1:]:
                    self.send_message(channel_id, extra)
            except Exception as exc:
                self.edit_message(channel_id, self._status_id, f"error: {exc}")
            finally:
                self.brain.on_event = None
                self.busy = False

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------ gateway

    def _send_json(self, payload: dict) -> None:
        self.ws.send(json.dumps(payload))

    def _identify(self) -> None:
        self._send_json({
            "op": 2,
            "d": {
                "token": self.token,
                "intents": INTENTS,
                "properties": {"os": "windows", "browser": "cerebro", "device": "cerebro"},
            },
        })

    def _heartbeat(self) -> None:
        if time.time() - self.last_heartbeat >= self.heartbeat_interval:
            self._send_json({"op": 1, "d": self.sequence})
            self.last_heartbeat = time.time()

    def connect(self) -> None:
        url = requests.get(f"{API}/gateway", timeout=15).json()["url"]
        self.ws = websocket.create_connection(
            f"{url}?v=10&encoding=json", timeout=5, sslopt={"cert_reqs": 2}
        )
        hello = json.loads(self.ws.recv())
        if hello.get("op") != 10:
            raise RuntimeError(f"gateway: se esperaba HELLO, llego {hello}")
        self.heartbeat_interval = hello["d"]["heartbeat_interval"] / 1000.0
        self.last_heartbeat = time.time()
        self._identify()

    def run_forever(self) -> None:
        backoff = 2.0
        while True:
            try:
                self.connect()
                backoff = 2.0
                print("bot conectado a Discord. Esperando mensajes del dueño…")
                while True:
                    try:
                        self.ws.settimeout(1.0)
                        raw = self.ws.recv()
                    except websocket.WebSocketTimeoutException:
                        self._heartbeat()
                        continue
                    if not raw:
                        continue
                    event = json.loads(raw)
                    if event.get("s"):
                        self.sequence = event["s"]
                    op = event.get("op")
                    if op == 11:
                        continue
                    if op == 7:
                        print("gateway pide reconexion…")
                        break
                    if op == 0:
                        print(f"[ev] {event.get('t')}")
                    if op == 0 and event.get("t") == "READY":
                        self.bot_user_id = event["d"]["user"]["id"]
                        print(f"READY como {event['d']['user']['username']}")
                    elif op == 0 and event.get("t") == "MESSAGE_CREATE":
                        try:
                            self.handle_message(event["d"])
                        except Exception as exc:
                            print(f"error manejando mensaje: {exc}")
            except Exception as exc:
                print(f"conexion perdida ({exc}); reintentando en {backoff:.0f}s")
                time.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
            finally:
                try:
                    if self.ws:
                        self.ws.close()
                except Exception:
                    pass

    # ------------------------------------------------------------ mensajes

    def handle_message(self, d: dict) -> None:
        author = d.get("author", {})
        content = (d.get("content") or "").strip()
        channel_id = d["channel_id"]
        if author.get("id") == self.bot_user_id or author.get("bot"):
            return
        if not content:
            return
        is_owner = author.get("id") == self.owner_id
        mentioned = f"<@{self.bot_user_id}>" in content
        in_auto_channel = bool(self.auto_channel) and channel_id == self.auto_channel
        if not (is_owner or mentioned or in_auto_channel):
            print(f"[ignorado] author id={author.get('id')} "
                  f"username={author.get('username')} content={content[:60]!r}")
            return  # silencioso: solo el dueño o menciones directas
        task = content
        if mentioned:
            task = content.replace(f"<@{self.bot_user_id}>", "").strip()
        elif task.lower().startswith(PREFIX):
            task = task[len(PREFIX):].strip()
        if task.lower() == "!estado":
            if not is_owner:
                return
            rot = self.brain.llm._rotator
            st = rot.status()
            lines = [f"key#{k['label']} {k['suffix']} · {'OK' if k['cooling_for'] <= 0 else f'en cooldown {k['cooling_for']:.0f}s'}"
                     for k in st["keys"]]
            self.send_message(channel_id, "🧠 estado:\n" + "\n".join(lines))
            return
        if task.lower() == "!ping":
            self.send_message(channel_id, "pong")
            return
        if not task or task.lower() in ("!estado", "!ping"):
            return
        self.run_task(channel_id, task)


def serve_health() -> None:
    """Servidor HTTP minimo para Render: responde 200 en $PORT (autoping lo mantiene vivo)."""
    from http.server import BaseHTTPRequestHandler, HTTPServer

    port = int(os.environ.get("PORT", "10000"))

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write("cerebro vivo 🧠".encode("utf-8"))

        def log_message(self, *args) -> None:
            pass

    try:
        HTTPServer(("0.0.0.0", port), Handler).serve_forever()
    except Exception as exc:
        print(f"servidor health no disponible ({exc}); continuo sin health endpoint")


def main() -> int:
    env = load_env()
    token = env.get("DISCORD_BOT_TOKEN", "").strip()
    owner = env.get("DISCORD_OWNER_ID", "").strip()
    if not token:
        print("Falta configuracion. Agrega al .env del proyecto:")
        print("  DISCORD_BOT_TOKEN=<token del portal de desarrolladores>")
        print("Guia: https://discord.com/developers/applications")
        return 1
    if not owner:
        print("[modo captura] sin DISCORD_OWNER_ID: se registrara el id de todo mensaje recibido")
    auto = env.get("DISCORD_AUTO_CHANNEL", "").strip()
    threading.Thread(target=serve_health, daemon=True).start()
    bot = DiscordBot(token, owner, auto_channel=auto)
    bot.run_forever()


if __name__ == "__main__":
    sys.exit(main())
