"""Журнал вызовов: каждая строка файла calls.jsonl — одно событие.

Пишем все вызовы LLM и инструментов, сообщения между агентами и ошибки.
По журналу считается распределение нагрузки между агентами (требование ТЗ: не больше 40%).
"""
from __future__ import annotations

import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

MAX_FIELD_CHARS = 2000  # длинные аргументы и результаты в журнале обрезаем


def _short(value: Any) -> Any:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= MAX_FIELD_CHARS else text[:MAX_FIELD_CHARS] + f"… [+{len(text) - MAX_FIELD_CHARS} симв.]"


class CallLogger:
    def __init__(self, path: Path, run_id: str):
        self.path = Path(path)
        self.run_id = run_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seq = 0

    def event(self, kind: str, agent: str = "system", **fields: Any) -> None:
        self._seq += 1
        record = {"seq": self._seq, "ts": datetime.now().isoformat(timespec="milliseconds"),
                  "run_id": self.run_id, "kind": kind, "agent": agent, **fields}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def llm_call(self, agent: str, step: int, duration_ms: float, ok: bool, *, model: str,
                 tool_calls: list[str] | None = None, tokens: dict | None = None, error: str | None = None,
                 text: str | None = None) -> None:
        self.event("llm_call", agent, step=step, model=model, duration_ms=round(duration_ms, 1), ok=ok,
                   requested_tools=tool_calls or [], tokens=tokens, error=error,
                   text=_short(text) if text else None)

    def tool_call(self, agent: str, step: int, name: str, args: dict, duration_ms: float, ok: bool,
                  *, result: Any = None, error: str | None = None) -> None:
        self.event("tool_call", agent, step=step, tool=name, args=args, duration_ms=round(duration_ms, 1),
                   ok=ok, result=_short(result) if result is not None else None, error=error)

    def message(self, sender: str, receiver: str, msg_type: str, payload: Any) -> None:
        self.event("message", sender, to=receiver, msg_type=msg_type, payload=_short(payload))


class Timer:
    def __enter__(self):
        self.started = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.ms = (time.perf_counter() - self.started) * 1000


def load_events(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_stats(path: Path) -> dict:
    """Сколько вызовов LLM и инструментов сделал каждый агент и какова его доля."""
    llm, tools = Counter(), Counter()
    for e in load_events(path):
        if e["kind"] == "llm_call":
            llm[e["agent"]] += 1
        elif e["kind"] == "tool_call":
            tools[e["agent"]] += 1
    agents = sorted(set(llm) | set(tools))
    total = sum(llm.values()) + sum(tools.values())
    rows = [{"agent": a, "llm_calls": llm[a], "tool_calls": tools[a], "total": llm[a] + tools[a],
             "share": round((llm[a] + tools[a]) / total, 3) if total else 0.0} for a in agents]
    return {"total_calls": total, "agents": rows,
            "max_share": max((r["share"] for r in rows), default=0.0)}


def format_stats(stats: dict) -> str:
    lines = [f"{'Агент':<12} {'LLM':>4} {'Инстр.':>7} {'Всего':>6} {'Доля':>7}"]
    for r in stats["agents"]:
        lines.append(f"{r['agent']:<12} {r['llm_calls']:>4} {r['tool_calls']:>7} {r['total']:>6} {r['share'] * 100:>6.1f}%")
    lines.append(f"{'Итого':<12} {'':>4} {'':>7} {stats['total_calls']:>6}")
    return "\n".join(lines)
