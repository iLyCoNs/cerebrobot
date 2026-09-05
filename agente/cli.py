from __future__ import annotations

import sys

from . import config
from .agent import Brain
from .llm import LLM
from .memory import Memory


def _on_event(kind: str, payload: dict) -> None:
    if kind == "tool":
        print(f"   [tool] {payload['tool']}({payload['args']})")
        res = str(payload.get("result", "")).strip()
        if res:
            preview = res.splitlines()[0][:100] if res.splitlines() else res[:100]
            print(f"          -> {preview}")
    elif kind == "progress":
        stage = payload.get("stage", "…")
        pct = payload.get("pct")
        val, mx = payload.get("value"), payload.get("max")
        detalle = f" {val}/{mx}" if val is not None else ""
        print(f"   [img] {stage}{detalle} · {pct}%")
    elif kind == "answer":
        print(f"   [respondido en {payload.get('steps')} pasos]")


def main() -> int:
    llm = LLM()
    memory = Memory(config.MEMORY_FILE)
    brain = Brain(llm, memory=memory, on_event=_on_event)

    print("AGENTE GROK (cerebro + subagentes) sobre NVIDIA")
    print("Modelos:", ", ".join(config.MODEL_CHAIN))
    print("Escribe 'salir' para terminar, 'memoria' para ver lo recordado.\n")

    while True:
        try:
            line = input("tu> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.lower() in {"salir", "exit", "quit", "q"}:
            break
        if line.lower() == "memoria":
            print(memory.summary())
            continue

        print("trabajando...")
        answer = brain.run(line)
        print("\n" + answer + "\n")
        memory.remember_turn(line, answer)

    return 0


if __name__ == "__main__":
    sys.exit(main())