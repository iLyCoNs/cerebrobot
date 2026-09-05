from __future__ import annotations

import base64
import ctypes
import io
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List

ToolFn = Callable[..., str]

@dataclass
class Tool:
    name: str
    description: str
    args_schema: Dict[str, Any]
    fn: ToolFn


def _truncate(text: str, limit: int = 5000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncado, {len(text) - limit} chars omitidos]"


def list_dir(path: str = ".", recursive: bool = False) -> str:
    import os

    p = path or "."
    if not recursive:
        entries = []
        for name in sorted(os.listdir(p)):
            full = os.path.join(p, name)
            tag = "DIR " if os.path.isdir(full) else "FILE"
            entries.append(f"{tag}  {full}")
        return "\n".join(entries) or "(vacío)"
    output = []
    for root, dirs, files in os.walk(p):
        dirs[:] = [d for d in dirs if d not in {"__pycache__", ".git", "node_modules", "venv", "comfyui", "sd-webui"}]
        for f in files:
            output.append(os.path.join(root, f))
    return _truncate("\n".join(output) or "(vacío)")


def read_file(path: str) -> str:
    from pathlib import Path

    return _truncate(Path(path).read_text(encoding="utf-8", errors="replace"))


def write_file(path: str, content: str) -> str:
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(content, encoding="utf-8")
    return f"archivo escrito: {path}"


def run_shell(command: str, timeout: int = 60) -> str:
    import subprocess

    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"[timeout tras {timeout}s]"
    out = (proc.stdout or "") + (proc.stderr or "")
    return _truncate(out or "(sin salida)")


def fetch_url(url: str) -> str:
    import requests

    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "agente-cerebro/0.1"})
        return _truncate(r.text, 8000)
    except Exception as e:
        return f"[error fetch]: {e}"


def run_python(code: str) -> str:
    import subprocess
    import tempfile
    import os

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        name = f.name
    try:
        proc = subprocess.run(
            ["python", name],
            capture_output=True,
            text=True,
            timeout=60,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        return _truncate(out or "(sin salida)")
    except subprocess.TimeoutExpired:
        return "[timeout tras 60s]"
    finally:
        try:
            os.unlink(name)
        except OSError:
            pass


# ------------------------------------------------------------- computer use

PANTALLAS_DIR = Path(__file__).resolve().parent.parent / "pantallas"
VISION_MODELS = [
    "meta/llama-3.2-11b-vision-instruct",
    "meta/llama-3.2-90b-vision-instruct",
]


def _dpi_aware() -> None:
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def screenshot(name: str = "") -> str:
    """Captura toda la pantalla virtual y guarda PNG en pantallas/."""
    import mss
    import mss.tools

    _dpi_aware()
    PANTALLAS_DIR.mkdir(exist_ok=True)
    base = (name or f"pant_{time.strftime('%Y%m%d_%H%M%S')}").strip()
    base = "".join(c for c in base if c.isalnum() or c in "-_") or "pantalla"
    path = PANTALLAS_DIR / f"{base}.png"
    counter = 2
    while path.exists():
        path = PANTALLAS_DIR / f"{base}_{counter}.png"
        counter += 1
    with mss.mss() as s:
        mon = s.monitors[0]
        shot = s.grab(mon)
        mss.tools.to_png(shot.rgb, shot.size, output=str(path))
        return f"{path} ({shot.width}x{shot.height}px)"


def see(image_path: str, question: str) -> str:
    """Envia una imagen a un modelo de vision (VLM) y devuelve su respuesta."""
    from PIL import Image

    from nvidia_rotator import NvidiaRotator

    p = Path(image_path)
    if not p.exists():
        return f"[error] no existe: {image_path}"
    img = Image.open(p)
    img.thumbnail((1280, 1280))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=80)
    b64 = base64.b64encode(buf.getvalue()).decode()
    messages = [{"role": "user", "content": [
        {"type": "text", "text": question},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
    ]}]
    rotator = NvidiaRotator(
        model=VISION_MODELS,
        timeout=(10.0, 150.0),
        state_file=str(Path(__file__).resolve().parent.parent / "nvidia_state.json"),
    )
    data = rotator.chat(messages, max_tokens=400, max_attempts=4)
    content = data["choices"][0]["message"]["content"]
    return content or "(el VLM no devolvio texto)"


def _gui():
    import pyautogui

    _dpi_aware()
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.05
    return pyautogui


def click_pct(x_pct: float, y_pct: float, double: bool = False) -> str:
    """Clic en coordenadas porcentuales de la pantalla (x% desde la izquierda, y% desde arriba)."""
    gui = _gui()
    w, h = gui.size()
    x = max(1, min(int(w * float(x_pct) / 100.0), w - 1))
    y = max(1, min(int(h * float(y_pct) / 100.0), h - 1))
    gui.moveTo(x, y, duration=0.25)
    time.sleep(0.1)
    if double:
        gui.doubleClick()
    else:
        gui.click()
    return f"clic en ({x}, {y}) de {w}x{h} ({x_pct}%, {y_pct}%)"


def type_text(text: str) -> str:
    """Escribe texto con el teclado en la ventana enfocada (solo ASCII, sin acentos/enie)."""
    gui = _gui()
    time.sleep(0.2)
    safe = "".join(ch for ch in str(text) if 32 <= ord(ch) < 127)
    gui.write(safe, interval=0.03)
    return f"escrito: {safe!r}"


def press_key(combo: str) -> str:
    """Pulsa una tecla o combinacion (ej: 'enter', 'ctrl+f', 'alt+tab', 'escape', 'win')."""
    gui = _gui()
    time.sleep(0.2)
    keys = [k.strip().lower() for k in str(combo).split("+") if k.strip()]
    for k in keys:
        gui.keyDown(k)
    for k in reversed(keys):
        gui.keyUp(k)
    return f"teclas pulsadas: {combo}"


def scroll(amount: int = -5) -> str:
    """Rueda del mouse: positivo=subir, negativo=bajar."""
    gui = _gui()
    gui.scroll(int(amount))
    return f"scroll {amount}"


def focus_window(title: str) -> str:
    """Trae al frente la primera ventana cuyo titulo contenga el texto dado."""
    import subprocess as sp

    r = sp.run(
        [
            "powershell", "-NoProfile", "-Command",
            "(New-Object -ComObject WScript.Shell).AppActivate("
            + json.dumps(str(title)) + ")",
        ],
        capture_output=True, text=True, timeout=20,
    )
    time.sleep(0.6)
    ok = (r.stdout or "").strip()
    return f"ventana '{title}' -> {'enfocada' if ok == 'True' else f'sin coincidencia ({ok or r.stderr.strip()[:120]})'}"


BUILTIN_TOOLS: List[Tool] = [
    Tool("list_dir", "Lista un directorio. Devuelve las entradas con prefijo DIR/FILE.",
         {"path": {"type": "string"}, "recursive": {"type": "boolean"}},
         lambda path=".", recursive=False: list_dir(path, recursive)),
    Tool("read_file", "Lee el contenido de un archivo de texto.",
         {"path": {"type": "string"}},
         lambda path: read_file(path)),
    Tool("write_file", "Escribe (crea o sobrescribe) un archivo de texto.",
         {"path": {"type": "string"}, "content": {"type": "string"}},
         lambda path, content: write_file(path, content)),
    Tool("run_shell", "Ejecuta un comando de terminal y devuelve su salida.",
         {"command": {"type": "string"}, "timeout": {"type": "integer"}},
         lambda command, timeout=60: run_shell(command, timeout)),
    Tool("run_python", "Ejecuta un fragmento de código Python y devuelve su salida estándar.",
         {"code": {"type": "string"}},
         lambda code: run_python(code)),
    Tool("fetch_url", "Descarga el contenido de una URL (HTML/texto).",
         {"url": {"type": "string"}},
         lambda url: fetch_url(url)),
    Tool(
        "screenshot",
        "Captura la pantalla completa y guarda un PNG en pantallas/. Devuelve la ruta y el tamano. "
        "Usala antes y despues de actuar para ver el resultado de tus acciones.",
        {"name": {"type": "string"}},
        lambda name="": screenshot(name),
    ),
    Tool(
        "see",
        "Mira una imagen con un modelo de vision (VLM): describe que hay, lee textos, ubicar elementos. "
        "Si pides posicion de algo, pide porcentajes (x% desde la izquierda, y% desde arriba) para click_pct.",
        {"image_path": {"type": "string"}, "question": {"type": "string"}},
        lambda image_path, question: see(image_path, question),
    ),
    Tool(
        "click_pct",
        "Hace clic en coordenadas PORCENTUALES de la pantalla (x% desde la izquierda, y% desde arriba). "
        "Ej: click_pct(75.5, 19.8). Con double=True hace doble clic. CUIDADO: clic real en la pantalla del usuario.",
        {"x_pct": {"type": "number"}, "y_pct": {"type": "number"}, "double": {"type": "boolean"}},
        lambda x_pct, y_pct, double=False: click_pct(x_pct, y_pct, double),
    ),
    Tool(
        "type_text",
        "Escribe texto con el teclado en la ventana enfocada (SOLO ascii: sin acentos ni enie).",
        {"text": {"type": "string"}},
        lambda text: type_text(text),
    ),
    Tool(
        "press_key",
        "Pulsa tecla o combinacion: 'enter', 'escape', 'tab', 'ctrl+f', 'ctrl+k', 'alt+tab', 'win', 'flecha abajo' = 'down'.",
        {"combo": {"type": "string"}},
        lambda combo: press_key(combo),
    ),
    Tool(
        "scroll",
        "Rueda del mouse en la ventana bajo el cursor: negativo baja, positivo sube (pasos).",
        {"amount": {"type": "integer"}},
        lambda amount=-5: scroll(amount),
    ),
    Tool(
        "focus_window",
        "Trae al frente la primera ventana cuyo titulo contenga el texto dado (ej: 'Discord', 'Chrome').",
        {"title": {"type": "string"}},
        lambda title: focus_window(title),
    ),
]


def tools_prompt_block(tools: List[Tool]) -> str:
    lines = []
    for t in tools:
        args = ", ".join(f"{k}: {v.get('type', 'string')}" for k, v in t.args_schema.items())
        lines.append(f"- {t.name}({args}): {t.description}")
    return "\n".join(lines)
