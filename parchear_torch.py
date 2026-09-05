"""Parchea torch/_library/infer_schema.py del venv para compatibilidad
torch 2.4.1 + comfy_kitchen (anotaciones list[int] / X | None / anotaciones
en string, y retornos -> None). Idempotente.
"""
from __future__ import annotations

import sys
from pathlib import Path

HELPER = '''import types as _pytypes


def _modernize_annotation(annotation):
    if isinstance(annotation, _pytypes.UnionType):
        return typing.Union[tuple(_modernize_annotation(a) for a in typing.get_args(annotation))]
    if isinstance(annotation, _pytypes.GenericAlias) and annotation.__origin__ in (list, tuple):
        args = typing.get_args(annotation)
        base = typing.List if annotation.__origin__ is list else typing.Tuple
        return base[tuple(_modernize_annotation(a) for a in args)] if args else base
    return annotation
'''

PAIRS = [
    (
        "# mypy: allow-untyped-defs\nimport inspect\nimport typing\n\nfrom .. import device, dtype, Tensor, types\n",
        "# mypy: allow-untyped-defs\nimport inspect\nimport typing\nimport types as _pytypes\n\nfrom .. import device, dtype, Tensor, types\n\n" + HELPER + "\n",
    ),
    (
        "    sig = inspect.signature(prototype_function)\n\n    def error_fn(what):",
        "    sig = inspect.signature(prototype_function)\n\n"
        "    try:\n"
        "        resolved = typing.get_type_hints(prototype_function)\n"
        "    except Exception:\n"
        "        resolved = {}\n\n"
        "    def error_fn(what):",
    ),
    (
        "        if param.annotation is inspect.Parameter.empty:\n"
        "            error_fn(f\"Parameter {name} must have a type annotation.\")\n\n"
        "        if param.annotation not in SUPPORTED_PARAM_TYPES.keys():\n"
        "            error_fn(\n"
        "                f\"Parameter {name} has unsupported type {param.annotation}. \"\n"
        "                f\"The valid types are: {SUPPORTED_PARAM_TYPES.keys()}.\"\n"
        "            )\n\n"
        "        schema_type = SUPPORTED_PARAM_TYPES[param.annotation]",
        "        raw = resolved.get(name, param.annotation)\n"
        "        if raw is inspect.Parameter.empty:\n"
        "            error_fn(f\"Parameter {name} must have a type annotation.\")\n\n"
        "        annotation = _modernize_annotation(raw)\n"
        "        if annotation not in SUPPORTED_PARAM_TYPES.keys():\n"
        "            error_fn(\n"
        "                f\"Parameter {name} has unsupported type {param.annotation}. \"\n"
        "                f\"The valid types are: {SUPPORTED_PARAM_TYPES.keys()}.\"\n"
        "            )\n\n"
        "        schema_type = SUPPORTED_PARAM_TYPES[annotation]",
    ),
    (
        "    ret = parse_return(sig.return_annotation, error_fn)\n",
        "    ret_annotation = resolved.get(\"return\", sig.return_annotation)\n"
        "    if ret_annotation is inspect.Signature.empty:\n"
        "        ret_annotation = None\n"
        "    ret = parse_return(ret_annotation, error_fn)\n",
    ),
    (
        "def parse_return(annotation, error_fn):\n    if annotation is None:\n        return \"()\"\n",
        "def parse_return(annotation, error_fn):\n    if annotation is None or annotation is type(None):\n        return \"()\"\n",
    ),
    (
        "    origin = typing.get_origin(annotation)\n    if origin is not tuple:\n",
        "    annotation = _modernize_annotation(annotation)\n    origin = typing.get_origin(annotation)\n    if origin is not tuple:\n",
    ),
    (
        "    args = typing.get_args(annotation)\n    for arg in args:\n",
        "    args = tuple(a for a in typing.get_args(annotation) if a is not type(None))\n    for arg in args:\n",
    ),
]


def main() -> int:
    target = Path(sys.argv[1])
    src = target.read_text(encoding="utf-8")
    if "_modernize_annotation" in src:
        print("ya parcheado")
        return 0
    for old, new in PAIRS:
        if old not in src:
            print(f"[error] patron no encontrado:\n{old[:80]}...")
            return 1
        src = src.replace(old, new, 1)
    target.write_text(src, encoding="utf-8")
    compile(src, str(target), "exec")
    print("parche aplicado y verificado")
    return 0


if __name__ == "__main__":
    sys.exit(main())
