"""Escucha de voz para el bot de Discord.

Se une al canal de voz del dueño, recibe el audio de los participantes
(RTP + opus), detecta cuando alguien habla (VAD por energia) y transcribe
cada frase con la API de Groq (Whisper, compatible OpenAI).

Solo necesita: PyNaCl (descifrado RTP), av/PyAV (decode opus), numpy
(resampleo). La connexion de voz usa el mismo token del bot.

Uso (desde discord_bot.py):
    sesion = VoiceSession(bot_token, user_id, guild_id, channel_id,
                           session_id, voice_token, endpoint, on_utterance)
    sesion.start()   # hilo daemon; cierra con sesion.stop()
"""

from __future__ import annotations

import io
import json
import queue
import socket
import struct
import threading
import time
import wave
from typing import Callable

import requests
import websocket

try:  # dependencias de voz: si faltan, el modulo queda deshabilitado
    import av
    import numpy as np
    from nacl.secret import SecretBox
    VOICE_LIBS_OK = True
except Exception:  # pragma: no cover
    VOICE_LIBS_OK = False

GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

SILENCE_MS_FINAL = 1200   # de silencio para dar la frase por terminada
MIN_UTTER_MS = 350        # frases mas cortas se descartan
MAX_UTTER_MS = 30000      # cap duro por frase
RMS_ON = 500              # umbral de energia para "esta hablando"
RMS_MIN_PEAK = 1200       # la frase necesita un pico minimo (evita ruido de fondo)


def transcribe(wav_bytes: bytes, api_key: str) -> str:
    r = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        files={"file": ("utt.wav", wav_bytes, "audio/wav")},
        data={"model": "whisper-large-v3", "language": "es", "temperature": "0"},
        timeout=60,
    )
    r.raise_for_status()
    return str(r.json().get("text", "")).strip()


def wav_from_pcm(pcm: bytes, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


class VoiceSession:
    """Conexion de voz de Discord completa: gateway + UDP + VAD + STT."""

    def __init__(
        self,
        bot_token: str,
        bot_user_id: str,
        guild_id: str,
        channel_id: str,
        session_id: str,
        voice_token: str,
        endpoint: str,
        on_utterance: Callable[[str, str, str], None],
        groq_key: str = "",
        owner_id: str = "",
    ) -> None:
        self.bot_token = bot_token
        self.bot_user_id = bot_user_id
        self.guild_id = guild_id
        self.channel_id = channel_id
        self.session_id = session_id
        self.voice_token = voice_token
        self.endpoint = endpoint.replace("wss://", "").strip("/")
        self.on_utterance = on_utterance
        self.groq_key = groq_key
        self.owner_id = owner_id

        self.ws: websocket.WebSocket | None = None
        self.udp: socket.socket | None = None
        self.ssrc = 0
        self.secret_key: bytes | None = None
        self.mode = ""
        self.ip = ""
        self.port = 0
        self.heartbeat_interval = 5.0
        self._hb_nonce = 0
        self._hb_last = 0.0
        self._seq_seen = -1
        self._rtp_ok = False

        self.ssrc_user: dict[int, str] = {}
        self._bufs: dict[int, _SpeakerBuf] = {}
        self._stt_queue: "queue.Queue" = queue.Queue()
        self._stop = threading.Event()
        self.threads: list[threading.Thread] = []
        self.ready = threading.Event()

    # ------------------------------------------------------------ ciclo

    def start(self) -> None:
        if not VOICE_LIBS_OK:
            raise RuntimeError("faltan libs de voz (PyNaCl/av/numpy)")
        for target in (self._gateway_loop, self._udp_loop, self._stt_loop):
            t = threading.Thread(target=target, daemon=True)
            t.start()
            self.threads.append(t)

    def stop(self) -> None:
        self._stop.set()
        try:
            if self.ws:
                self.ws.close()
        except Exception:
            pass
        try:
            if self.udp:
                self.udp.close()
        except Exception:
            pass

    # ------------------------------------------------------------ voice gateway

    def _send_json(self, payload: dict) -> None:
        if self.ws:
            self.ws.send(json_dumps(payload))

    def _gateway_loop(self) -> None:
        try:
            url = f"wss://{self.endpoint}/?v=4&encoding=json"
            print(f"[voz] conectando voice ws a {self.endpoint} ...", flush=True)
            self.ws = websocket.create_connection(url, timeout=5, sslopt={"cert_reqs": 2})
            print("[voz] voice ws conectado", flush=True)
            hello = json_loads(self.ws.recv())
            if hello.get("op") != 8:
                raise RuntimeError(f"voice hello inesperado: {str(hello)[:120]}")
            self.heartbeat_interval = hello["d"].get("heartbeat_interval", 13750) / 1000.0
            print("[voz] voice hello ok, identificando ...", flush=True)
            self._hb_last = time.time()
            self._send_json({
                "op": 0,
                "d": {
                    "server_id": self.guild_id,
                    "user_id": self.bot_user_id,
                    "session_id": self.session_id,
                    "token": self.voice_token,
                },
            })
            self.ws.settimeout(1.0)
            while not self._stop.is_set():
                try:
                    raw = self.ws.recv()
                except websocket.WebSocketTimeoutException:
                    if time.time() - self._hb_last >= self.heartbeat_interval:
                        self._hb_nonce += 1
                        self._send_json({"op": 3, "d": self._hb_nonce})
                        self._hb_last = time.time()
                    continue
                if not raw:
                    continue
                ev = json_loads(raw)
                op = ev.get("op")
                if op == 2:  # ready: ssrc + ip:port internos para descubrimiento
                    d = ev["d"]
                    self.ssrc = d["ssrc"]
                    print(f"[voz] voice ready: ssrc={self.ssrc} udp={d['ip']}:{d['port']}", flush=True)
                    self._udp_discovery(d["ip"], d["port"])
                elif op == 4:  # session description: clave secreta
                    d = ev["d"]
                    self.mode = d.get("mode", "")
                    self.secret_key = bytes(d["secret_key"])
                    self.ready.set()
                    print(f"[voz] lista en {self.channel_id} (modo {self.mode})", flush=True)
                elif op == 5:  # speaking: ssrc -> usuario
                    d = ev["d"]
                    self.ssrc_user[int(d["ssrc"])] = str(d.get("user_id", ""))
                elif op == 13:  # cliente desconectado
                    d = ev["d"]
                    uid = str(d.get("user_id", ""))
                    for ssrc, u in list(self.ssrc_user.items()):
                        if u == uid:
                            self.ssrc_user.pop(ssrc, None)
                            self._bufs.pop(ssrc, None)
        except Exception as exc:
            if not self._stop.is_set():
                print(f"[voz] gateway caida: {exc}", flush=True)

    # ------------------------------------------------------------ UDP / RTP

    def _udp_discovery(self, local_ip: str, local_port: int) -> None:
        self.udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp.settimeout(10)
        pkt = struct.pack(">H", 1) + struct.pack(">H", 70) + struct.pack(">I", self.ssrc) + b"\x00" * 64
        print(f"[voz] udp discovery: enviando a {local_ip}:{local_port} ...", flush=True)
        self.udp.sendto(pkt, (local_ip, local_port))
        data, addr = self.udp.recvfrom(2048)
        ext_ip = data[8:72].split(b"\x00")[0].decode()
        ext_port = struct.unpack(">H", data[72:74])[0]
        self.ip, self.port = ext_ip, ext_port
        print(f"[voz] udp discovery ok: ip externa {ext_ip}:{ext_port}", flush=True)
        self.udp.settimeout(1.0)
        self._send_json({
            "op": 1,
            "d": {
                "protocol": "udp",
                "data": {
                    "address": ext_ip,
                    "port": ext_port,
                    "mode": "xsalsa20_poly1305_lite",
                },
            },
        })

    def _udp_loop(self) -> None:
        dec = None
        resampler = None
        if VOICE_LIBS_OK:
            for name in ("libopus", "opus"):
                try:
                    dec = av.CodecContext.create(name, "r")
                    break
                except Exception:
                    continue
            resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        while not self._stop.is_set():
            if self.secret_key is None or self.udp is None:
                time.sleep(0.2)
                continue
            try:
                data, _ = self.udp.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break
            if not self._rtp_ok:
                self._rtp_ok = True
                print("[voz] rtp: primer paquete de audio recibido", flush=True)
            opus = self._decrypt_rtp(data)
            if not opus:
                continue
            if dec is None:
                continue
            try:
                pcm = _decode_opus(dec, resampler, opus)
            except Exception:
                continue
            if pcm:
                ssrc = struct.unpack(">I", data[8:12])[0]
                self._feed_vad(ssrc, pcm)

    def _decrypt_rtp(self, data: bytes) -> bytes:
        if len(data) < 12 or self.secret_key is None:
            return b""
        b0, b1 = data[0], data[1]
        cc = b0 & 0x0F
        has_ext = bool(b0 & 0x10)
        header_len = 12 + 4 * cc
        if len(data) <= header_len:
            return b""
        if has_ext:
            if len(data) < header_len + 4:
                return b""
            ext_len = struct.unpack(">H", data[header_len + 2:header_len + 4])[0]
            header_len += 4 + 4 * ext_len
            if len(data) <= header_len:
                return b""
        nonce = data[-4:].ljust(24, b"\x00")
        try:
            return SecretBox(self.secret_key).decrypt(data[header_len:-4], nonce)
        except Exception:
            return b""

    # ------------------------------------------------------------ VAD

    def _feed_vad(self, ssrc: int, pcm: bytes) -> None:
        if ssrc not in self._bufs:
            self._bufs[ssrc] = _SpeakerBuf()
        self._bufs[ssrc].feed(pcm)
        for s, buf in list(self._bufs.items()):
            wav_bytes = buf.poll()
            if wav_bytes:
                self._stt_queue.put((ssrc, self.ssrc_user.get(ssrc, ""), wav_bytes))

    # ------------------------------------------------------------ STT

    def _stt_loop(self) -> None:
        while not self._stop.is_set():
            try:
                ssrc, user_id, wav_bytes = self._stt_queue.get(timeout=1.0)
            except Exception:
                continue
            if not self.groq_key:
                continue
            try:
                text = transcribe(wav_bytes, self.groq_key)
            except Exception as exc:
                print(f"[stt] error: {exc}", flush=True)
                continue
            if text:
                print(f"[stt] transcrito: {text}", flush=True)
                try:
                    self.on_utterance(user_id, self.owner_id, text)
                except Exception as exc:
                    print(f"[stt] callback: {exc}", flush=True)


class _SpeakerBuf:
    """Acumula PCM de un hablante y corta la frase al detectar silencio."""

    def __init__(self) -> None:
        self.pcm = bytearray()
        self.speaking = False
        self.silence_ms = 0
        self.total_ms = 0
        self.peak = 0

    def feed(self, pcm: bytes) -> None:
        samples = np.frombuffer(pcm, dtype=np.int16)
        if samples.size == 0:
            return
        ms = int(1000 * samples.size / 16000)
        rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
        self.total_ms += ms
        if self.total_ms > MAX_UTTER_MS and self.speaking:
            self.finalize_now()
            return
        if rms > RMS_ON:
            self.speaking = True
            self.silence_ms = 0
            self.peak = max(self.peak, rms)
        elif self.speaking:
            self.silence_ms += ms
        if self.speaking:
            self.pcm.extend(pcm)

    def poll(self) -> bytes | None:
        """Devuelve la frase terminada (wav) o None."""
        if self.speaking and self.silence_ms >= SILENCE_MS_FINAL:
            return self.finalize_now()
        return None

    def finalize_now(self) -> bytes | None:
        self.speaking = False
        data = bytes(self.pcm)
        self.pcm = bytearray()
        self.silence_ms = 0
        self.total_ms = 0
        peak = self.peak
        self.peak = 0
        if len(data) < 16000 * 2 * MIN_UTTER_MS // 1000 or peak < RMS_MIN_PEAK:
            return None
        return wav_from_pcm(data)


# ------------------------------------------------------------ helpers

def json_dumps(obj: dict) -> str:
    return json.dumps(obj)


def json_loads(raw: bytes) -> dict:
    return json.loads(raw)


def _decode_opus(dec, resampler, opus: bytes) -> bytes:
    out = bytearray()
    for frame in dec.decode(av.Packet(opus)):
        for rf in resampler.resample(frame):
            out.extend(rf.to_ndarray().astype(np.int16).tobytes())
    return bytes(out)
