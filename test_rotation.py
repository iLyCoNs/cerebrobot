"""Prueba sintetica: simula timeouts para verificar la rotacion de keys."""
import time

import requests

import nvidia_rotator as nr

rotator = nr.NvidiaRotator()
llamadas = []
reales = {"post": requests.post}


class FakeTimeout(Exception):
    pass


def post_falso(url, headers=None, **kwargs):
    key = headers["Authorization"].split(" ")[1]
    llamadas.append(key[-6:])
    if len(llamadas) <= 3:
        raise requests.exceptions.Timeout("simulated timeout")
    class R:
        status_code = 200
        headers = {}
        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}
    return R()


requests.post = post_falso
nr.requests = requests
rotator.cooldown = 5.0

resultado = rotator.chat([{"role": "user", "content": "test"}], max_attempts=6)
requests.post = reales["post"]

assert resultado["choices"][0]["message"]["content"] == "ok"
assert len(set(llamadas[:4])) == 4, f"no rotó: {llamadas}"
estado = rotator.status()["keys"]
cooling = [s for s in estado if s["cooling_for"] > 0]
assert len(cooling) == 3, f"deben haber 3 keys en cooldown: {estado}"
print("ROTACION OK: se probaron 4 keys distintas y 3 quedaron en cooldown")
print("estado:", estado)
print("esperando a que expiren cooldowns de prueba...")
time.sleep(5.5)
assert all(s["cooling_for"] == 0 for s in rotator.status()["keys"])
print("COOLDOWN EXPIRA OK")
