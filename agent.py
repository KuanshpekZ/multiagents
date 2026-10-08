"""Базовый агент: цикл «LLM думает -> вызывает инструменты -> отдаёт итоговый JSON».

Здесь собрана вся надёжность: лимит шагов и времени, защита от повторов,
обработка ошибок инструментов, проверка итогового ответа и журнал всех вызовов.
Конкретные агенты (soc/agents/*) задают только роль, инструменты и проверку результата.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from soc import config
from soc.llm import LLM, LLMBadToolCall, LLMError, LLMResponse
from soc.logging_utils import CallLogger, Timer


class AgentError(Exception):
    """Агент не смог выполнить задачу (лимит, сбой LLM, негодный ответ)."""


class OutputRejected(Exception):
    """Итоговый ответ LLM не прошёл проверку. Текст ошибки отправляется модели на исправление."""


class TransientToolError(Exception):
    """Временный сбой инструмента (например, сеть): вызов стоит повторить."""


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict          # JSON Schema аргументов
    func: Callable[..., Any]
    retries: int = 0          # повторы при TransientToolError

    def schema(self) -> dict:
        return {"type": "function",
                "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}


def extract_json(text: str | None) -> dict:
    """Достаёт JSON-объект из ответа модели (допускает обёртку ```json ... ```)."""
    if not text or not text.strip():
        raise ValueError("пустой ответ")
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("в ответе нет JSON-объекта")
    value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("ожидался JSON-объект")
    return value


class Agent:
    name: str = "agent"
    role: str = ""                        # одна чёткая роль (попадает в системный промпт)
    instructions: str = ""                # как действовать
    decision_model: type[BaseModel]       # схема итогового JSON, который пишет LLM
    MAX_REPAIRS = 2                       # сколько раз просим исправить негодный итог
    MAX_SAME_CALL = 2                     # один и тот же вызов инструмента не больше 2 раз

    def __init__(self, llm: LLM, logger: CallLogger, tools: list[Tool],
                 max_steps: int = config.AGENT_MAX_STEPS, timeout_s: float = config.AGENT_TIMEOUT_S):
        if len(tools) > config.MAX_TOOLS_PER_AGENT:
            raise ValueError(f"У агента {self.name} {len(tools)} инструментов, по ТЗ не больше "
                             f"{config.MAX_TOOLS_PER_AGENT}")
        self.llm, self.logger = llm, logger
        self.tools = {t.name: t for t in tools}
        self.max_steps, self.timeout_s = max_steps, timeout_s
        self.tool_history: list[dict] = []   # что агент вызывал и что получил

    # --- то, что переопределяют конкретные агенты ---

    def build_output(self, decision: BaseModel, task: BaseModel) -> BaseModel:
        """Проверяет решение LLM и собирает выходное сообщение. Может поднять OutputRejected."""
        raise NotImplementedError

    # --- общий цикл ---

    def system_prompt(self) -> str:
        schema = json.dumps(self.decision_model.model_json_schema(), ensure_ascii=False)
        return (f"Ты — {self.role}\n\n{self.instructions}\n\n"
                "Общие правила:\n"
                "- Числа и факты бери только из результатов инструментов, ничего не выдумывай.\n"
                "- Не вызывай один и тот же инструмент с теми же аргументами повторно.\n"
                f"- У тебя не больше {self.max_steps} шагов. Когда данных достаточно, заверши работу.\n"
                "- Итоговый ответ: только один JSON-объект по схеме ниже, без пояснений вокруг. "
                "Пиши его обычным текстом сообщения, а не вызовом инструмента: инструмента для итогового "
                "ответа нет. Тексты внутри JSON пиши по-русски.\n\n"
                f"Схема итогового JSON:\n{schema}\n\n"
                "Схема — это описание формата, а не ответ. Верни объект с полями "
                f"{', '.join(self.decision_model.model_fields)} и их значениями.")

    def run(self, task: BaseModel) -> BaseModel:
        started = time.monotonic()
        self.tool_history = []
        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": "Задача (JSON):\n" + task.model_dump_json()},
        ]
        tool_schemas = [t.schema() for t in self.tools.values()]
        repairs, same_calls = 0, {}
        answer_now = False   # следующий шаг без инструментов: модель должна дать итоговый ответ

        for step in range(1, self.max_steps + 1):
            if time.monotonic() - started > self.timeout_s:
                raise self._fail(f"превышен лимит времени {self.timeout_s:.0f} с")

            last_step = step == self.max_steps
            if last_step:   # на последнем шаге инструменты не даём: модель обязана ответить
                messages.append({"role": "user", "content":
                                 "Лимит шагов исчерпан. Верни итоговый JSON по схеме прямо сейчас."})
            no_tools = last_step or answer_now
            answer_now = False
            try:
                response = self._ask_llm(step, messages, None if no_tools else tool_schemas)
            except LLMBadToolCall as e:
                answer = self._answer_from_bad_call(e.generation)
                if answer is None:
                    # Разобрать не удалось: сообщаем модели об ошибке, и следующий шаг идёт без инструментов,
                    # то есть ответить можно только текстом
                    messages.append({"role": "user", "content":
                                     f"Ответ не принят: {e}\nИнструмента для итогового ответа нет. "
                                     "Сейчас верни итоговый JSON по схеме обычным текстом сообщения."})
                    answer_now = True
                    continue
                response = LLMResponse(content=answer)   # дальше проверяется как обычный итоговый ответ
            messages.append(response.as_message())

            if response.tool_calls and not no_tools:
                for call in response.tool_calls:
                    result = self._run_tool(step, call, same_calls)
                    messages.append({"role": "tool", "tool_call_id": call.id,
                                     "content": json.dumps(result, ensure_ascii=False, default=str)})
                continue

            try:
                decision = self.decision_model.model_validate(extract_json(response.content))
                output = self.build_output(decision, task)
            except (ValueError, ValidationError, OutputRejected) as e:
                self.logger.event("output_rejected", self.name, step=step, reason=str(e)[:500])
                repairs += 1
                if repairs > self.MAX_REPAIRS or last_step:
                    raise self._fail(f"итоговый ответ не прошёл проверку: {e}") from e
                messages.append({"role": "user", "content":
                                 f"Ответ не принят: {e}\nИсправь и верни только JSON по схеме."})
                continue

            self.logger.event("agent_done", self.name, steps=step,
                              seconds=round(time.monotonic() - started, 2))
            return output

        raise self._fail(f"исчерпан лимит шагов ({self.max_steps})")

    def _fail(self, reason: str) -> AgentError:
        self.logger.event("agent_failed", self.name, error=reason)
        return AgentError(f"Агент «{self.name}» остановлен: {reason}")

    def _ask_llm(self, step: int, messages: list[dict], tools: list[dict] | None):
        with Timer() as t:
            try:
                response = self.llm.chat(self.name, messages, tools)
            except LLMBadToolCall as e:
                self.logger.llm_call(self.name, step, 0.0, False, model=self.llm.model, error=str(e),
                                     text=e.generation)
                raise
            except LLMError as e:
                self.logger.llm_call(self.name, step, 0.0, False, model=self.llm.model, error=str(e))
                raise self._fail(str(e)) from e
        self.logger.llm_call(self.name, step, t.ms, True, model=self.llm.model,
                             tool_calls=[c.name for c in response.tool_calls], tokens=response.usage,
                             text=response.content)
        return response

    def _answer_from_bad_call(self, generation: str) -> str | None:
        """Модели gpt-oss иногда оформляют итоговый JSON как вызов несуществующего инструмента (например, «JSON»).
        Тогда аргументы этого вызова и есть ответ: берём их как обычный текст. Если это попытка вызвать
        настоящий инструмент или текст не разбирается, возвращаем None."""
        try:
            call = json.loads(generation)
        except ValueError:
            return None
        if not isinstance(call, dict) or call.get("name") in self.tools or not isinstance(call.get("arguments"), dict):
            return None
        return json.dumps(call["arguments"], ensure_ascii=False)

    def _run_tool(self, step: int, call, same_calls: dict) -> Any:
        """Выполняет один вызов инструмента. Ошибки не роняют агента, а возвращаются модели."""
        key = call.name + json.dumps(call.arguments, sort_keys=True, ensure_ascii=False)
        same_calls[key] = same_calls.get(key, 0) + 1

        if call.name not in self.tools:
            return self._tool_error(step, call, f"нет инструмента «{call.name}». Доступны: {sorted(self.tools)}")
        if call.parse_error:
            return self._tool_error(step, call, call.parse_error)
        if same_calls[key] > self.MAX_SAME_CALL:
            return self._tool_error(step, call, "этот вызов уже выполнялся с теми же аргументами. "
                                                "Используй полученный результат и переходи к итоговому ответу.")
        tool = self.tools[call.name]
        error = ""
        for attempt in range(tool.retries + 1):
            with Timer() as t:
                try:
                    result = tool.func(**call.arguments)
                    error = ""
                except TransientToolError as e:
                    error = f"временный сбой: {e}"
                except TypeError as e:          # модель передала не те аргументы
                    error = f"неверные аргументы: {e}"
                except Exception as e:          # noqa: BLE001 — любой сбой инструмента возвращаем модели
                    error = f"{type(e).__name__}: {e}"
            if not error:
                self.logger.tool_call(self.name, step, call.name, call.arguments, t.ms, True, result=result)
                self.tool_history.append({"tool": call.name, "args": call.arguments, "result": result})
                return result
            if not error.startswith("временный сбой") or attempt == tool.retries:
                break
            time.sleep(min(2 ** attempt, 10))
        return self._tool_error(step, call, error)

    def _tool_error(self, step: int, call, error: str) -> dict:
        self.logger.tool_call(self.name, step, call.name, call.arguments, 0.0, False, error=error)
        return {"error": error}

    def last_result(self, tool_name: str) -> Any | None:
        for record in reversed(self.tool_history):
            if record["tool"] == tool_name:
                return record["result"]
        return None
