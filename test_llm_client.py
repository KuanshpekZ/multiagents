"""Клиент LLM через настоящий HTTP: локальный сервер отвечает в формате OpenAI Chat Completions.

Так проверяется весь путь режима api (запрос, вызовы инструментов, разбор ответа, повторы)
без ключа и без внешней сети. Решения на сервере принимают те же правила, что в offline-режиме.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from soc import config
from soc.llm import LLMBadToolCall, LLMError, OpenAICompatLLM
from soc.offline_llm import OfflineLLM
from soc.pipeline import check_expected, run_pipeline, scenario_index


class FakeOpenAIServer:
    def __init__(self, fail_first: int = 0, status_on_fail: int = 500, code_on_fail: str | None = None):
        self.requests, self.fail_first, self.status_on_fail = [], fail_first, status_on_fail
        self.code_on_fail = code_on_fail
        rules = OfflineLLM()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(body)
                if len(outer.requests) <= outer.fail_first:
                    return self._send(outer.status_on_fail, {"error": {"message": "сбой", "type": "server_error",
                                                                       "code": outer.code_on_fail,
                                                                       "failed_generation": "{}"}})
                system = body["messages"][0]["content"]
                agent = "triage" if "первой линии" in system else "explainer"
                answer = rules.chat(agent, body["messages"], body.get("tools"))
                message = {"role": "assistant", "content": answer.content}
                if answer.tool_calls:
                    message["tool_calls"] = [
                        {"id": c.id, "type": "function",
                         "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                        for c in answer.tool_calls]
                self._send(200, {"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                                 "choices": [{"index": 0, "message": message,
                                              "finish_reason": "tool_calls" if answer.tool_calls else "stop"}],
                                 "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})

            def _send(self, status, payload):
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


@pytest.fixture
def server():
    s = FakeOpenAIServer()
    yield s
    s.close()


def test_full_pipeline_through_http_api(server, tmp_path):
    llm = OpenAICompatLLM(server.url, "test-key", "test-model")
    index = scenario_index()
    state, _ = run_pipeline(config.SCENARIO_DIR / "mixed.csv", scenario="mixed", llm=llm, runs_dir=tmp_path)
    assert state.status == "completed" and state.llm_model == "test-model"
    assert check_expected(state, index["mixed"]["expected_incidents"])["passed"]
    first = server.requests[0]
    assert first["temperature"] == 0 and len(first["tools"]) == 4          # инструменты триажа ушли в запрос
    assert any(m["role"] == "tool" for m in server.requests[-1]["messages"])  # результаты вернулись модели


def test_retries_on_server_error(monkeypatch, tmp_path):
    monkeypatch.setattr("time.sleep", lambda s: None)
    server = FakeOpenAIServer(fail_first=2)
    try:
        llm = OpenAICompatLLM(server.url, "k", "m", max_retries=3)
        response = llm.chat("triage", [{"role": "system", "content": "первой линии"},
                                       {"role": "user", "content": "{}"}], None)
        assert response.content and len(server.requests) == 3
    finally:
        server.close()


def test_gives_clear_error_when_api_is_down(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    server = FakeOpenAIServer(fail_first=99)
    try:
        with pytest.raises(LLMError, match="после 3 попыток"):
            OpenAICompatLLM(server.url, "k", "m", max_retries=3).chat("triage", [{"role": "user", "content": "x"}], None)
    finally:
        server.close()


def test_wrong_key_fails_without_retries():
    server = FakeOpenAIServer(fail_first=99, status_on_fail=401)
    try:
        with pytest.raises(LLMError, match="отклонила запрос"):
            OpenAICompatLLM(server.url, "bad", "m").chat("triage", [{"role": "user", "content": "x"}], None)
        assert len(server.requests) == 1
    finally:
        server.close()


def test_broken_tool_call_is_reported_to_agent():
    # так Groq отвечает, если модель вызвала инструмент, которого нет в запросе
    server = FakeOpenAIServer(fail_first=99, status_on_fail=400, code_on_fail="tool_use_failed")
    try:
        with pytest.raises(LLMBadToolCall) as error:
            OpenAICompatLLM(server.url, "k", "m").chat("triage", [{"role": "user", "content": "x"}], None)
        assert error.value.generation == "{}"   # текст, который сгенерировала модель, передан агенту
        assert len(server.requests) == 1   # ошибка модели, а не сети: запрос не повторяется
    finally:
        server.close()


def test_missing_key_is_reported():
    with pytest.raises(LLMError, match="LLM_API_KEY"):
        OpenAICompatLLM("", "", "")
