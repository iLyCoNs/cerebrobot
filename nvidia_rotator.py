"""Rotador de API keys para NVIDIA NIM (integrate.api.nvidia.com).

Estrategia ante fallos:
- Timeout / error de conexion: la key entra en cooldown (60s por defecto),
  se salta y se reintenta inmediatamente con la siguiente key.
- 429 (rate limit / saturacion): reintento con backoff corto rotando keys.
- 401/403 (key invalida): esa key queda inhabilitada 10 minutos y rota.
- Otros 4xx (404, 400...): error inmediato, no rota keys.

Ademas soporta una CADENA DE MODELOS con fallback: si el modelo principal
agota sus intentos (ej. kimi-k3 saturado con 429), queda en cooldown de
modelo (10 min por defecto) y las siguientes consultas van directas al
modelo de respaldo hasta que expire.

Uso:
    from nvidia_rotator import NvidiaRotator

    rotator = NvidiaRotator(model=[
        "moonshotai/kimi-k3",
        "deepseek-ai/deepseek-v4-pro-0813",
    ])
    respuesta = rotator.ask("Hola")    # texto plano
    data = rotator.chat(messages)      # respuesta completa tipo OpenAI
    for trozo in rotator.chat(mensajes, stream=True):  # streaming
        ...
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Sequence, Tuple, Union

import requests

BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODELS = ["moonshotai/kimi-k3", "deepseek-ai/deepseek-v4-pro-0813"]
DEFAULT_TIMEOUT: Tuple[float, float] = (10.0, 60.0)
DEFAULT_COOLDOWN = 60.0
INVALID_KEY_COOLDOWN = 600.0
DEFAULT_MODEL_COOLDOWN = 600.0
BACKOFF_BASE = 2.0
BACKOFF_CAP = 15.0
DEFAULT_MIN_INTERVAL = 1.5
DEFAULT_MAX_CALLS_PER_MINUTE = 20
DEFAULT_MAX_DISTINCT_429 = 3
DEFAULT_STATE_FILENAME = "nvidia_state.json"

Message = Dict[str, Any]
Messages = Union[List[Message], Tuple[Message, ...]]
ModelSpec = Union[str, Sequence[str]]


class AllKeysCoolingDown(RuntimeError):
    pass


class NvidiaAPIError(RuntimeError):
    def __init__(self, status_code: Optional[int], detail: str) -> None:
        super().__init__(f"HTTP {status_code}: {detail}")
        self.status_code = status_code


class _KeyState:
    __slots__ = ("key", "label", "cooldown_until", "last_error")

    def __init__(self, key: str, label: int) -> None:
        self.key = key
        self.label = label
        self.cooldown_until = 0.0
        self.last_error: Optional[str] = None


def _parse_keys(raw: str) -> List[str]:
    keys: List[str] = []
    for part in raw.replace(";", ",").replace("\n", ",").split(","):
        k = part.strip()
        if k:
            keys.append(k)
    return keys


def load_keys(explicit: Optional[List[str]] = None, env_file: Optional[str] = None) -> List[str]:
    keys: List[str] = []
    if explicit:
        keys = list(explicit)
    else:
        env = os.environ.get("NVIDIA_API_KEYS")
        if env:
            keys = _parse_keys(env)
        else:
            candidates = (
                [Path(env_file)]
                if env_file
                else [Path.cwd() / ".env", Path(__file__).resolve().parent / ".env"]
            )
            for path in candidates:
                if path.exists():
                    for line in path.read_text(encoding="utf-8").splitlines():
                        name, _, value = line.strip().partition("=")
                        if name.strip() == "NVIDIA_API_KEYS":
                            keys = _parse_keys(value.strip().strip('"').strip("'"))
                            break
                    if keys:
                        break

    seen: set = set()
    unique = [k for k in keys if not (k in seen or seen.add(k))]
    if not unique:
        raise ValueError(
            "No se encontraron API keys. Define NVIDIA_API_KEYS en el entorno "
            "o en un archivo .env, o pasalas al constructor: NvidiaRotator(keys=[...])"
        )
    return unique


class NvidiaRotator:
    """Cliente compatible con OpenAI que rota keys de NVIDIA ante fallos."""

    def __init__(
        self,
        keys: Optional[List[str]] = None,
        model: ModelSpec = DEFAULT_MODELS,
        base_url: str = BASE_URL,
        timeout: Tuple[float, float] = DEFAULT_TIMEOUT,
        cooldown: float = DEFAULT_COOLDOWN,
        model_cooldown: float = DEFAULT_MODEL_COOLDOWN,
        env_file: Optional[str] = None,
        state_file: Optional[str] = None,
        min_interval: float = DEFAULT_MIN_INTERVAL,
        max_calls_per_minute: int = DEFAULT_MAX_CALLS_PER_MINUTE,
        max_distinct_429: int = DEFAULT_MAX_DISTINCT_429,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.cooldown = cooldown
        self.model_cooldown = model_cooldown
        requested = model if model is not None else DEFAULT_MODELS
        self._models = [requested] if isinstance(requested, str) else list(requested)
        self.state_file = Path(
            state_file
            or os.environ.get("NVIDIA_STATE_FILE")
            or (Path(__file__).resolve().parent / DEFAULT_STATE_FILENAME)
        )
        self.min_interval = max(0.0, min_interval)
        self.max_calls_per_minute = max(1, max_calls_per_minute)
        self.max_distinct_429 = max(1, max_distinct_429)
        self._lock = threading.RLock()
        self._states = [_KeyState(k, i) for i, k in enumerate(load_keys(keys, env_file))]
        self._cursor = 0
        self._model_cooldown_until: Dict[str, float] = {}
        self.on_retry: Optional[Callable[..., None]] = None
        self._refresh_from_shared()

    @property
    def models(self) -> List[str]:
        return list(self._models)

    def _shared_template(self) -> Dict[str, Any]:
        return {"keys": {}, "models": {}, "calls": {}, "last_request": 0.0}

    def _load_shared(self) -> Dict[str, Any]:
        shared = self._shared_template()
        try:
            if not self.state_file.exists():
                return shared
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
        except Exception:
            return shared
        if not isinstance(data, dict):
            return shared
        for section in ("keys", "models", "calls"):
            value = data.get(section)
            if isinstance(value, dict):
                shared[section] = value
        try:
            shared["last_request"] = float(data.get("last_request", 0.0))
        except Exception:
            shared["last_request"] = 0.0
        return shared

    def _write_shared(self, shared: Dict[str, Any]) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_file.with_name(self.state_file.name + ".tmp")
        tmp.write_text(json.dumps(shared), encoding="utf-8")
        os.replace(tmp, self.state_file)

    def _save_shared(self) -> None:
        with self._lock:
            now = time.time()
            shared = self._load_shared()
            keys = shared["keys"]
            for st in self._states:
                label = f"key#{st.label}"
                keys[label] = max(float(keys.get(label, 0.0)), st.cooldown_until)
            models = shared["models"]
            for model, until in self._model_cooldown_until.items():
                models[model] = max(float(models.get(model, 0.0)), until)
            calls = shared["calls"]
            for label, stamps in list(calls.items()):
                fresh = [t for t in stamps if isinstance(t, (int, float)) and t > now - 120.0]
                if fresh:
                    calls[label] = fresh
                else:
                    calls.pop(label, None)
            self._write_shared(shared)

    def _refresh_from_shared(self) -> None:
        with self._lock:
            shared = self._load_shared()
            for st in self._states:
                try:
                    until = float(shared["keys"].get(f"key#{st.label}", 0.0))
                except Exception:
                    until = 0.0
                if until > st.cooldown_until:
                    st.cooldown_until = until
            for model, until in shared["models"].items():
                try:
                    until_value = float(until)
                except Exception:
                    continue
                if until_value > self._model_cooldown_until.get(model, 0.0):
                    self._model_cooldown_until[model] = until_value

    def _record_request(self, label: str) -> None:
        with self._lock:
            now = time.time()
            shared = self._load_shared()
            history = [
                t for t in shared["calls"].get(label, [])
                if isinstance(t, (int, float)) and t > now - 60.0
            ]
            history.append(now)
            shared["calls"][label] = history
            shared["last_request"] = now
            self._write_shared(shared)

    def _pace_wait(self) -> float:
        if self.min_interval <= 0:
            return 0.0
        with self._lock:
            shared = self._load_shared()
            return max(0.0, shared.get("last_request", 0.0) + self.min_interval - time.time())

    def _available_in(self) -> float:
        with self._lock:
            now = time.time()
            shared = self._load_shared()
            waits: List[float] = []
            for st in self._states:
                if st.cooldown_until > now:
                    waits.append(st.cooldown_until - now)
            calls = shared.get("calls", {})
            for st in self._states:
                history = [
                    t for t in calls.get(f"key#{st.label}", [])
                    if isinstance(t, (int, float)) and t > now - 60.0
                ]
                if len(history) >= self.max_calls_per_minute and history:
                    waits.append(max(0.0, history[0] + 60.0 - now))
            waits.append(max(0.0, float(shared.get("last_request", 0.0)) + self.min_interval - now))
            positive = [w for w in waits if w > 0]
            return min(positive) if positive else 0.0

    def _notify_retry(self, model: str, label: str, status: str, wait: float) -> None:
        callback = self.on_retry
        if callback is None:
            return
        try:
            callback(model=model, key=label, status=status, wait=round(float(wait), 1))
        except Exception:
            pass

    def _cool_key(self, st: _KeyState, seconds: float, reason: str) -> None:
        with self._lock:
            st.cooldown_until = time.time() + seconds
            st.last_error = reason
            self._save_shared()

    def _next_key(self, allow_cooling: bool = False) -> Optional[_KeyState]:
        with self._lock:
            self._refresh_from_shared()
            now = time.time()
            shared = self._load_shared()
            calls = shared.get("calls", {})
            n = len(self._states)
            for off in range(n):
                st = self._states[(self._cursor + off) % n]
                if st.cooldown_until > now:
                    continue
                history = [
                    t for t in calls.get(f"key#{st.label}", [])
                    if isinstance(t, (int, float)) and t > now - 60.0
                ]
                if len(history) >= self.max_calls_per_minute:
                    continue
                self._cursor = (self._cursor + off + 1) % n
                return st
            if allow_cooling and n:
                st = min(self._states, key=lambda s: s.cooldown_until)
                self._cursor = (self._states.index(st) + 1) % n
                return st
            return None

    def _cool_model(self, model: str, seconds: Optional[float] = None) -> None:
        with self._lock:
            self._model_cooldown_until[model] = time.time() + (seconds or self.model_cooldown)
            self._save_shared()

    def _clear_model(self, model: str) -> None:
        with self._lock:
            self._model_cooldown_until.pop(model, None)
            shared = self._load_shared()
            shared["models"].pop(model, None)
            self._write_shared(shared)

    def status(self) -> Dict[str, Any]:
        now = time.time()
        with self._lock:
            return {
                "keys": [
                    {
                        "label": f"key#{st.label}",
                        "suffix": st.key[-6:],
                        "cooling_for": max(0.0, round(st.cooldown_until - now, 1)),
                        "last_error": st.last_error,
                    }
                    for st in self._states
                ],
                "models_cooling": {
                    m: round(until - now, 1)
                    for m, until in self._model_cooldown_until.items()
                    if until > now
                },
            }

    def ask(self, prompt: str, system: Optional[str] = None, **kwargs: Any) -> str:
        messages: Messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        data = self.chat(list(messages), **kwargs)
        return data["choices"][0]["message"]["content"]

    def chat(
        self,
        messages: Messages,
        model: Optional[ModelSpec] = None,
        stream: bool = False,
        max_tokens: int = 1024,
        temperature: Optional[float] = None,
        extra_payload: Optional[Dict[str, Any]] = None,
        max_attempts: Optional[int] = None,
        **kwargs: Any,
    ) -> Union[Dict[str, Any], Generator[str, None, None]]:
        chain = list(model) if isinstance(model, (list, tuple)) else (
            [model] if model else self._default_chain()
        )
        base_payload: Dict[str, Any] = {
            "messages": list(messages),
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if temperature is not None:
            base_payload["temperature"] = temperature
        if extra_payload:
            base_payload.update(extra_payload)
        base_payload.update(kwargs)

        last_error: Optional[BaseException] = None
        for idx, chosen in enumerate(chain):
            try:
                payload = dict(base_payload)
                payload["model"] = chosen
                return self._attempt_model(payload, stream, max_attempts)
            except (AllKeysCoolingDown, NvidiaAPIError) as exc:
                self._cool_model(chosen)
                nxt = chain[idx + 1] if idx + 1 < len(chain) else None
                if nxt:
                    self._notify_retry(
                        chosen, "modelo",
                        f"fallando · cambio a {nxt.split('/')[-1]}",
                        self.model_cooldown,
                    )
                else:
                    self._notify_retry(chosen, "modelo", "sin modelos disponibles", self.model_cooldown)
                last_error = exc
        raise AllKeysCoolingDown(
            f"Ningun modelo de la cadena {chain} respondio. Ultimo error: {last_error!r}"
        ) from last_error

    def _default_chain(self) -> List[str]:
        with self._lock:
            cooling = dict(self._model_cooldown_until)
        ready = [m for m in self._models if cooling.get(m, 0) <= time.time()]
        return ready or list(self._models)

    def _attempt_model(
        self,
        payload: Dict[str, Any],
        stream: bool,
        max_attempts: Optional[int],
    ) -> Union[Dict[str, Any], Generator[str, None, None]]:
        model = payload["model"]
        url = f"{self.base_url}/chat/completions"
        attempts = max_attempts or (2 * len(self._states))
        attempt = 0
        last_error: Optional[BaseException] = None
        distinct_429: set = set()

        while attempt < attempts:
            st = self._next_key(allow_cooling=False)
            if st is None:
                wait = min(self._available_in(), 5.0)
                time.sleep(wait)
                continue

            attempt += 1
            pace = self._pace_wait()
            if pace > 0:
                time.sleep(pace)
            self._record_request(f"key#{st.label}")
            try:
                response = requests.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {st.key}",
                        "Accept": "text/event-stream" if stream else "application/json",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout,
                    stream=stream,
                )
            except (requests.Timeout, requests.ConnectionError) as exc:
                self._cool_key(st, self.cooldown, type(exc).__name__)
                self._notify_retry(model, f"key#{st.label}", type(exc).__name__, self.cooldown)
                last_error = exc
                continue

            if response.status_code == 200:
                self._clear_model(model)
                if stream:
                    return self._stream_content(response)
                return response.json()

            detail = response.text[:500]
            if response.status_code in (401, 403):
                self._cool_key(st, INVALID_KEY_COOLDOWN, f"HTTP {response.status_code}")
                last_error = NvidiaAPIError(response.status_code, detail)
                continue

            if response.status_code == 429 or response.status_code >= 500:
                if response.status_code == 429:
                    distinct_429.add(st.label)
                    if len(distinct_429) >= self.max_distinct_429:
                        last_error = NvidiaAPIError(response.status_code, detail)
                        break
                backoff = min(BACKOFF_CAP, BACKOFF_BASE * (2 ** (attempt - 1)))
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    try:
                        backoff = min(BACKOFF_CAP, float(retry_after))
                    except ValueError:
                        pass
                wait = backoff + random.uniform(0, 1)
                self._cool_key(st, wait, f"HTTP {response.status_code}")
                self._notify_retry(model, f"key#{st.label}", f"HTTP {response.status_code}", wait)
                last_error = NvidiaAPIError(response.status_code, detail)
                time.sleep(wait)
                continue

            raise NvidiaAPIError(response.status_code, detail)

        raise AllKeysCoolingDown(
            f"Modelo {model}: se agotaron los intentos ({attempts}). "
            f"Ultimo error: {last_error!r}"
        ) from last_error

    @staticmethod
    def _stream_content(response: requests.Response) -> Generator[str, None, None]:
        def generator() -> Generator[str, None, None]:
            with response:
                for line in response.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices") or [{}]
                    delta = choices[0].get("delta", {}) if choices else {}
                    piece = delta.get("content") or ""
                    if piece:
                        yield piece
        return generator()
