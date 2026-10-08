"""Надёжность цикла агента: лимиты, защита от зацикливания, исправление ответа, сбои LLM."""
import json

import pytest
from pydantic import BaseModel

from soc.agent import Agent, AgentError, Tool
from soc.llm import LLMBadToolCall, LLMError, LLMResponse, ToolCall


class Answer(BaseModel):
    value: int


class EchoAgent(Agent):
    name = "echo"
    role = "тестовый агент."
    decision_model = Answer

    def build_output(self, decision, task):
        return decision


class Task(BaseModel):
    text: str = "go"


class ScriptedLLM:
    """Возвращает заранее заданные ответы по очереди; последний повторяется."""
    model = "scripted"

    def __init__(self, *responses):
        self.responses, self.calls = list(responses), 0
        self.tools_sent = []   # были ли инструменты в каждом запросе

    def chat(self, agent, messages, tools):
        self.calls += 1
        self.tools_sent.append(bool(tools))
        response = self.responses[min(self.calls - 1, len(self.responses) - 1)]
        if isinstance(response, Exception):
            raise response
        return response


def tool_call(name="ping", **args):
    return LLMResponse(content=None, tool_calls=[ToolCall(id="c1", name=name, arguments=args)])


def final(obj):
    return LLMResponse(content=json.dumps(obj))


PING = Tool("ping", "тест", {"type": "object", "properties": {}}, lambda: {"pong": True})


def events(logger, kind):
    return [json.loads(line) for line in logger.path.read_text(encoding="utf-8").splitlines()
            if json.loads(line)["kind"] == kind]


def test_stops_at_step_limit_when_llm_loops(logger):
    llm = ScriptedLLM(tool_call())                       # модель бесконечно зовёт один инструмент
    agent = EchoAgent(llm, logger, [PING], max_steps=5)
    with pytest.raises(AgentError):
        agent.run(Task())
    assert llm.calls == 5                                # ровно лимит, не больше
    tools = events(logger, "tool_call")
    assert sum(e["ok"] for e in tools) == 2              # повторный вызов выполнен не больше 2 раз
    assert any("уже выполнялся" in (e["error"] or "") for e in tools)


def test_repairs_invalid_answer(logger):
    llm = ScriptedLLM(LLMResponse(content="это не JSON"), final({"value": "x"}), final({"value": 7}))
    assert EchoAgent(llm, logger, [PING]).run(Task()).value == 7
    assert llm.calls == 3
    assert len(events(logger, "output_rejected")) == 2   # оба отклонённых ответа записаны в журнал


def test_gives_up_after_too_many_bad_answers(logger):
    llm = ScriptedLLM(LLMResponse(content="мусор"))
    with pytest.raises(AgentError, match="не прошёл проверку"):
        EchoAgent(llm, logger, [PING]).run(Task())


def test_tool_errors_do_not_crash_agent(logger):
    def broken():
        raise RuntimeError("сломался")

    llm = ScriptedLLM(tool_call("no_such_tool"), tool_call("broken"), final({"value": 1}))
    agent = EchoAgent(llm, logger, [Tool("broken", "тест", {"type": "object", "properties": {}}, broken)])
    assert agent.run(Task()).value == 1
    assert [e["ok"] for e in events(logger, "tool_call")] == [False, False]


def test_llm_failure_becomes_agent_error(logger):
    with pytest.raises(AgentError, match="недоступна"):
        EchoAgent(ScriptedLLM(LLMError("LLM недоступна")), logger, [PING]).run(Task())
    assert events(logger, "llm_call")[0]["ok"] is False


def test_answer_sent_as_fake_tool_call_is_accepted(logger):
    # модель оформила итог как вызов несуществующего инструмента «JSON»: аргументы и есть ответ
    generation = json.dumps({"name": "JSON", "arguments": {"value": 5}})
    llm = ScriptedLLM(LLMBadToolCall("attempted to call tool 'JSON'", generation))
    assert EchoAgent(llm, logger, [PING]).run(Task()).value == 5
    assert llm.calls == 1                    # лишнего запроса не было


def test_answer_from_fake_tool_call_is_still_checked(logger):
    # ответ из такого вызова проходит ту же проверку: негодный отклоняется и модель исправляется
    generation = json.dumps({"name": "JSON", "arguments": {"value": "не число"}})
    llm = ScriptedLLM(LLMBadToolCall("bad", generation), final({"value": 4}))
    assert EchoAgent(llm, logger, [PING]).run(Task()).value == 4
    assert len(events(logger, "output_rejected")) == 1


def test_recovers_after_broken_tool_call(logger):
    # модель сломала вызов, и ответ из него не извлечь; на следующем шаге она исправилась
    llm = ScriptedLLM(LLMBadToolCall("attempted to call tool 'JSON'"), final({"value": 3}))
    assert EchoAgent(llm, logger, [PING]).run(Task()).value == 3
    assert [e["ok"] for e in events(logger, "llm_call")] == [False, True]
    assert llm.tools_sent == [True, False]   # после сбоя модель отвечает без инструментов


def test_tool_limit_from_assignment(logger):
    with pytest.raises(ValueError, match="не больше"):
        EchoAgent(ScriptedLLM(final({"value": 1})), logger, [PING] * 6)
