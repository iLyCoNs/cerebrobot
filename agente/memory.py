from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


def _load(path: Path) -> Dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


class Memory:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.data = _load(self.path)
        self.data.setdefault("preferences", [])
        self.data.setdefault("history", [])
        self.data.setdefault("cache", {})

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")

    def remember_turn(self, question: str, answer: str) -> None:
        self.data["history"].append({"question": question, "answer": answer})
        if len(self.data["history"]) > 200:
            self.data["history"] = self.data["history"][-200:]
        self.save()

    def recent_history(self, n: int = 6) -> List[Dict[str, str]]:
        return self.data["history"][-n:]

    def add_preference(self, text: str) -> None:
        self.data["preferences"].append(text)
        self.save()

    def cached(self, key: str, producer: Callable[[], str]) -> str:
        if key in self.data["cache"]:
            hit = self.data["cache"][key]
            self.data["cache"].pop(key)
            self.data["cache"][key] = hit
            self.save()
            return hit
        value = producer()
        self.data["cache"][key] = value
        if len(self.data["cache"]) > 50:
            oldest = next(iter(self.data["cache"]))
            del self.data["cache"][oldest]
        self.save()
        return value

    def summary(self) -> str:
        prefs = self.data.get("preferences", [])
        hist = self.data.get("history", [])
        lines = []
        if prefs:
            lines.append("Preferencias recordadas:")
            lines += [f"- {p}" for p in prefs[-10:]]
        if hist:
            lines.append("Conversaciones anteriores:")
            for h in hist[-6:]:
                lines.append(f"  Q: {h['question'][:80]}")
        return "\n".join(lines) or "(sin memoria previa)"