"""Клиент LLM. Работает с любым OpenAI-совместимым API (Groq, Gemini, OpenRouter, OpenAI, Ollama).

Адрес, ключ и модель задаются в .env. Сбои сети и лимиты запросов повторяются с паузой,
после исчерпания попыток поднимается LLMError с понятным текстом.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from soc import config


class LLMError(Exception):
    """LLM недоступна или вернула непригодный ответ."""


class LLMBadToolCall(LLMError):
    """Модель сломала вызов инструмента (например, «вызвала» несуществующий), и провайдер отклонил ответ.
    Это ошибка модели, а не сети, поэтому запрос не повторяется: агент разбирается с ней сам."""

    def __init__(self, message: str, generation: str = ""):
        super().__init__(message)
        self.generation = generation   # что сгенерировала модель (Groq присылает это в поле failed_generation)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    parse_error: str | None = None   # если модель прислала аргументы не в виде JSON


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] | None = None

    def as_message(self) -> dict:
        """Ответ модели в формате истории диалога."""
        message: dict[str, Any] = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            message["tool_calls"] = [
                {"id": c.id, "type": "function",
                 "function": {"name": c.name, "arguments": json.dumps(c.arguments, ensure_ascii=False)}}
                for c in self.tool_calls]
        return message


class LLM(Protocol):
    model: str

    def chat(self, agent: str, messages: list[dict], tools: list[dict] | None) -> LLMResponse: ...


class OpenAICompatLLM:
    """Обращение к LLM через OpenAI-совместимый Chat Completions API с вызовом инструментов."""

    def __init__(self, base_url: str, api_key: str, model: str,
                 timeout_s: float = config.LLM_TIMEOUT_S, max_retries: int = config.LLM_MAX_RETRIES):
        from openai import OpenAI  # импорт здесь, чтобы offline-режим не требовал пакета

        if not api_key or not model:
            raise LLMError("Не заданы LLM_API_KEY или LLM_MODEL. Скопируйте .env.example в .env и заполните.")
        self.model = model
        self.max_retries = max_retries
        self._client = OpenAI(base_url=base_url or None, api_key=api_key, timeout=timeout_s, max_retries=0)

    def chat(self, agent: str, messages: list[dict], tools: list[dict] | None) -> LLMResponse:
        import openai

        kwargs: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": 0}
        if tools:
            kwargs["tools"] = tools
        last_error = ""
        for attempt in range(1, self.max_retries + 1):
            try:
                completion = self._client.chat.completions.create(**kwargs)
                return self._parse(completion)
            except openai.BadRequestError as e:
                # Некоторые модели не принимают параметр temperature: убираем его и пробуем ещё раз
                if "temperature" in kwargs and "temperature" in str(e).lower():
                    kwargs.pop("temperature")
                    last_error = f"BadRequestError: {e}"
                    continue
                # Groq проверяет вызовы инструментов на своей стороне и отклоняет ответ с ошибкой
                # tool_use_failed, если модель вызвала инструмент, которого нет в списке
                if getattr(e, "code", None) == "tool_use_failed":
                    body = e.body if isinstance(e.body, dict) else {}
                    raise LLMBadToolCall(body.get("message") or str(e), body.get("failed_generation") or "") from e
                raise LLMError(f"LLM отклонила запрос (BadRequestError): {e}") from e
            except (openai.AuthenticationError, openai.PermissionDeniedError, openai.NotFoundError) as e:
                # Неверный ключ, адрес или имя модели: повтор не поможет
                raise LLMError(f"LLM отклонила запрос ({type(e).__name__}): {e}") from e
            except openai.RateLimitError as e:
                # Лимит запросов или токенов в минуту: ждём столько, сколько просит сервер
                last_error = f"RateLimitError: {e}"
                if attempt < self.max_retries:
                    time.sleep(self._retry_after(e, default=15 * attempt))
            except openai.APIError as e:
                # Сеть, тайм-аут, ошибка 5xx: короткая пауза и повтор
                last_error = f"{type(e).__name__}: {e}"
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 20))   # 2, 4, 8… секунд
        raise LLMError(f"LLM недоступна после {self.max_retries} попыток. Последняя ошибка: {last_error}")

    @staticmethod
    def _retry_after(error: Any, default: float) -> float:
        """Пауза перед повтором: из заголовка Retry-After, но не дольше минуты."""
        try:
            return min(float(error.response.headers.get("retry-after")), 60.0)
        except (AttributeError, TypeError, ValueError):
            return min(default, 60.0)

    @staticmethod
    def _parse(completion: Any) -> LLMResponse:
        if not completion.choices:
            raise LLMError("LLM вернула пустой ответ (нет choices).")
        message = completion.choices[0].message
        calls = []
        for i, call in enumerate(message.tool_calls or []):
            raw = call.function.arguments or "{}"
            try:
                args, problem = json.loads(raw), None
                if not isinstance(args, dict):
                    args, problem = {}, f"аргументы не являются объектом JSON: {raw[:200]}"
            except json.JSONDecodeError as e:
                args, problem = {}, f"аргументы не являются JSON ({e}): {raw[:200]}"
            calls.append(ToolCall(id=call.id or f"call_{i}", name=call.function.name, arguments=args,
                                  parse_error=problem))
        usage = None
        if getattr(completion, "usage", None):
            usage = {"prompt": completion.usage.prompt_tokens or 0,
                     "completion": completion.usage.completion_tokens or 0}
        return LLMResponse(content=message.content, tool_calls=calls, usage=usage)


def make_llm(mode: str = "auto") -> LLM:
    """mode: api — настоящая LLM из .env; offline — без сети, решения по фиксированным правилам;
    auto — api, если ключ задан, иначе offline."""
    if mode == "auto":
        mode = "api" if config.LLM_API_KEY else "offline"
    if mode == "api":
        return OpenAICompatLLM(config.LLM_BASE_URL, config.LLM_API_KEY, config.LLM_MODEL)
    if mode == "offline":
        from soc.offline_llm import OfflineLLM

        return OfflineLLM()
    raise ValueError(f"Неизвестный режим LLM: {mode}")
