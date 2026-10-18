"""Test doubles for the Anthropic client. No test ever calls the real API."""

import json
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import anthropic
import duckdb
import httpx2  # the HTTP client anthropic 1.x is built on

Script = list[Any]  # items: a message, an exception to raise, or a callable(request) -> message


def usage(input_tokens: int = 100, output_tokens: int = 50, cache_read: int = 0, cache_write: int = 0) -> Any:
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_write,
    )


def text_message(text: str, *, stop_reason: str = "end_turn") -> Any:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)], stop_reason=stop_reason, usage=usage()
    )


def tool_use_message(*calls: tuple[str, dict[str, Any]]) -> Any:
    blocks = [
        SimpleNamespace(type="tool_use", id=f"toolu_{i}", name=name, input=args)
        for i, (name, args) in enumerate(calls)
    ]
    return SimpleNamespace(content=blocks, stop_reason="tool_use", usage=usage())


def final_answer(
    text: str, numbers: list[dict[str, Any]], proposed: list[dict[str, Any]] | None = None
) -> Any:
    payload = {"answer_text": text, "numbers": numbers, "assumptions": [], "proposed_actions": proposed or []}
    return text_message(json.dumps(payload))


def last_tool_numbers(request: dict[str, Any]) -> list[dict[str, Any]]:
    """Numbers from the most recent tool_result in a request (unwrapping the <data> fence)."""
    for message in reversed(request["messages"]):
        content = message["content"]
        if message["role"] == "user" and isinstance(content, list):
            for block in content:
                if block.get("type") == "tool_result" and not block.get("is_error"):
                    body = block["content"].split("\n", 1)[1].rsplit("\n", 1)[0]
                    return json.loads(body)["numbers"]
    return []


def rate_limit_error() -> Exception:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.RateLimitError("rate limited", response=httpx2.Response(429, request=request), body=None)


def fetch_one(conn: duckdb.DuckDBPyConnection, sql: str, params: list[Any] | None = None) -> tuple[Any, ...]:
    """Run a query that must return exactly one row."""
    row = conn.execute(sql, params or []).fetchone()
    assert row is not None, f"no row for: {sql}"
    return row


class FakeMessages:
    def __init__(self, script: Script) -> None:
        self.script = list(script)
        self.requests: list[dict[str, Any]] = []

    def create(self, **request: Any) -> Any:
        # Snapshot messages: the loop keeps appending to the same list after this call.
        self.requests.append({**request, "messages": list(request.get("messages", []))})
        if not self.script:
            raise AssertionError("fake client script exhausted")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        if callable(item):
            return item(request)
        return item


class FakeClient:
    def __init__(self, script: Script) -> None:
        self.messages = FakeMessages(script)


def answer_citing_last_tool(text_fmt: str) -> Callable[[dict[str, Any]], Any]:
    """Final answer whose text and numbers[] come from the previous tool result."""

    def make(request: dict[str, Any]) -> Any:
        numbers = last_tool_numbers(request)
        return final_answer(text_fmt.format(*[n["value"] for n in numbers]), numbers)

    return make
