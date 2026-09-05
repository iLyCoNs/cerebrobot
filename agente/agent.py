from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

from . import config
from . import tools as tools_module
from .llm import LLM, parse_action
from .tools import BUILTIN_TOOLS, Tool, tools_prompt_block

TOOL_CALL_TEMPLATE = (
    "Para REALIZAR una acción (usar una herramienta), responde ÚNICAMENTE con un JSON:\n"
    '{"tool": "<nombre>", "args": {"<arg>": "<valor>"}}\n'
    "No añadas texto fuera del JSON cuando llames una herramienta.\n"
    "Para dar la RESPUESTA FINAL, escribe texto normal (sin JSON)."
)


class Agent:
    def __init__(
        self,
        llm: LLM,
        name: str = "agente",
        tools: Optional[List[Tool]] = None,
        system: Optional[str] = None,
        max_steps: int = None,
        on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> None:
        self.llm = llm
        self.name = name
        self.tools = tools or BUILTIN_TOOLS
        self._tool_map = {t.name: t for t in self.tools}
        self.system = system
        self.max_steps = max_steps or config.MAX_STEPS
        self.on_event = on_event

    def _emit(self, kind: str, payload: Dict[str, Any]) -> None:
        if self.on_event:
            self.on_event(kind, payload)

    def run(self, task: str) -> str:
        changelog = [f"[{self.name}] iniciando"]
        errors = []
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": task},
        ]

        for step in range(self.max_steps):
            content = self.llm.complete(messages) or ""
            action = parse_action(content)

            if action is None:
                self._emit("answer", {"task": task, "content": content, "steps": step + 1})
                return content

            tool_name = action.get("tool")
            args = action.get("args") or {}
            if not isinstance(args, dict):
                args = {}

            if tool_name not in self._tool_map:
                observation = f"[error] herramienta desconocida: {tool_name}"
                errors.append(observation)
            else:
                try:
                    result = self._tool_map[tool_name].fn(**args)
                    observation = str(result)
                except Exception as e:
                    observation = f"[error] {e}"
                    errors.append(observation)

            self._emit("tool", {"tool": tool_name, "args": args, "result": observation})
            changelog.append(f"[{self.name}] llamó {tool_name} -> {observation[:80]}")
            messages.append({"role": "assistant", "content": json.dumps(action)})
            messages.append({"role": "user", "content": f"Resultado de {tool_name}:\n{observation}"})

        final = f"[{self.name}] alcanzó max_steps={self.max_steps} sin respuesta final."
        if errors:
            final += " Errores: " + "; ".join(errors[-3:])
        return final

    def _system_prompt(self) -> str:
        lines = [f"Eres '{self.name}', un agente autónomo que resuelve tareas usando herramientas."]
        if self.system:
            lines.append(self.system)
        lines.append("\nHerramientas disponibles:")
        lines.append(tools_prompt_block(self.tools))
        lines.append("\n" + TOOL_CALL_TEMPLATE)
        lines.append("Trabaja paso a paso. Usa herramientas cuando necesites información o efectos reales.")
        return "\n".join(lines)


class Brain(Agent):
    def __init__(
        self,
        llm: LLM,
        subagent_tools: Optional[List[Tool]] = None,
        memory: Optional["Memory"] = None,
        on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> None:
        self.llm = llm
        self.subagent_tools = subagent_tools or BUILTIN_TOOLS
        self.memory = memory
        self.on_event = on_event
        self._depth = 0
        spawn = Tool(
            "spawn",
            "Crea un subagente especializado para resolver una subtarea y devuelve su resultado. "
            "Úsalo para dividir trabajo grande en partes enfocadas.",
            {"instruction": {"type": "string"}},
            self._spawn,
        )
        super().__init__(
            llm,
            name="cerebro",
            tools=[spawn],
            system=(
                "Eres un orquestador (cerebro) con visión global. Tu única herramienta es 'spawn', "
                "que lanza subagentes especializados dotados de herramientas reales (archivos, "
                "terminal, web, código Python).\n"
                "REGLAS:\n"
                "1. Cuando la tarea necesite información del sistema o efectos reales, DEBES usar "
                "'spawn' con una instrucción específica. Nunca respondas que no puedes: delega.\n"
                "2. Puedes lanzar varios 'spawn' en cadena para dividir trabajo grande; agrega sus "
                "resultados y entrega una respuesta final clara y en español.\n"
                "3. Si la tarea es solo conversacional o de razonamiento, responde directamente."
            ),
            on_event=on_event,
        )

    def _spawn(self, instruction: str) -> str:
        if self._depth >= config.MAX_SUBAGENT_DEPTH:
            return f"[error] profundidad máxima de subagentes alcanzada ({config.MAX_SUBAGENT_DEPTH})"
        self._depth += 1
        try:
            sub = Agent(
                self.llm,
                name=f"subagente[{self._depth}]",
                tools=self.subagent_tools,
                system="Eres un subagente enfocado y autónomo. Tienes herramientas reales "
                "(archivos, terminal, web, código). Resuelve la instrucción asignada usándolas "
                "y devuelve solo el resultado final.",
                max_steps=config.MAX_STEPS,
                on_event=self.on_event,
            )
            if self.memory:
                resultado = self.memory.cached(instruction, lambda: sub.run(instruction))
            else:
                resultado = sub.run(instruction)
            return resultado
        finally:
            self._depth -= 1