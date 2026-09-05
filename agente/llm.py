from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_CWD = str(Path(__file__).resolve().parent.parent)
if _CWD not in sys.path:
    sys.path.insert(0, _CWD)

try:
    from nvidia_rotator import NvidiaRotator
except Exception:  # pragma: no cover
    NvidiaRotator = None  # type: ignore

from . import config


class LLM:
    def __init__(self, models: Optional[List[str]] = None) -> None:
        if NvidiaRotator is None:
            raise RuntimeError("No se pudo importar nvidia_rotator.py (NvidiaRotator)")
        self._rotator = NvidiaRotator(model=models or config.MODEL_CHAIN, timeout=(10.0, 180.0))

    def complete(self, messages: List[Dict[str, Any]], max_tokens: int = None) -> str:
        data = self._rotator.chat(
            list(messages),
            max_tokens=max_tokens or config.MAX_TOKENS,
            max_attempts=6,
        )
        return data["choices"][0]["message"]["content"] or ""


def parse_action(content: str) -> Optional[Dict[str, Any]]:
    text = content.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "tool" in obj:
            return obj
    except Exception:
        pass
    brace = re.search(r"\{.*\}", text, re.S)
    if brace:
        try:
            obj = json.loads(brace.group(0))
            if isinstance(obj, dict) and "tool" in obj:
                return obj
        except Exception:
            pass
    return None