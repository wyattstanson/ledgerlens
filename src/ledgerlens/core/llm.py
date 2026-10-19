"""Claude client wrapper (BUILD_PROMPT section 9.5): routing, caching, retries, budget and usage logging.

- Model and effort come from config/models.yaml, never from code.
- The static prefix (tools + system prompt) carries one cache breakpoint; per-request content follows.
- 429 and 529 are retried with jittered exponential backoff; the SDK's own retries are turned off so
  attempts are counted in one place.
- Every call writes an ``audit.llm_calls`` row from ``response.usage`` and refuses to start once the
  month's spend reaches the soft cap.
"""

import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import anthropic
import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from ledgerlens.core.db import connect, utcnow
from ledgerlens.core.guards import DATA_BLOCK_RULE
from ledgerlens.core.lineage import new_id

Effort = Literal["low", "medium", "high", "xhigh", "max"]

# 5-minute cache writes bill at 1.25x base input (Claude API pricing, checked 2026-10-19).
CACHE_WRITE_MULTIPLIER = 1.25
# Message Batches bill at half the standard price (Claude API pricing, checked 2026-10-19).
BATCH_DISCOUNT = 0.5
RETRYABLE: tuple[type[Exception], ...] = (anthropic.RateLimitError, anthropic.OverloadedError)


class ModelPrice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_per_mtok: float = Field(ge=0)
    output_per_mtok: float = Field(ge=0)
    cache_read_ratio: float = Field(ge=0, le=1)


class TaskRoute(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    effort: Effort
    batch: bool = False
    offline_only: bool = False


class Budget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    monthly_soft_cap_usd: float = Field(gt=0)


class ModelsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    models: dict[str, ModelPrice]
    tasks: dict[str, TaskRoute]
    budget: Budget

    @classmethod
    def from_yaml(cls, path: Path) -> "ModelsConfig":
        config = cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        unknown = {route.model for route in config.tasks.values()} - set(config.models)
        if unknown:
            raise ValueError(f"tasks route to models without prices: {sorted(unknown)}")
        return config

    def route(self, task: str) -> TaskRoute:
        try:
            return self.tasks[task]
        except KeyError:
            raise KeyError(f"no model route for task {task!r}; add it to config/models.yaml") from None


def cost_usd(
    price: ModelPrice,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int,
    cache_write_tokens: int,
) -> float:
    per_token_in = price.input_per_mtok / 1_000_000
    return (
        input_tokens * per_token_in
        + output_tokens * price.output_per_mtok / 1_000_000
        + cache_read_tokens * per_token_in * price.cache_read_ratio
        + cache_write_tokens * per_token_in * CACHE_WRITE_MULTIPLIER
    )


class BudgetExceededError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMResponse:
    message: Any  # anthropic.types.Message
    call_id: str
    cost_usd: float


def build_request(
    route: TaskRoute,
    *,
    system: str,
    messages: Sequence[dict[str, Any]],
    tools: Sequence[dict[str, Any]] | None,
    output_schema: dict[str, Any] | None,
    max_tokens: int,
) -> dict[str, Any]:
    """Assemble a Messages API request with the cacheable prefix first.

    Render order is tools -> system -> messages, so one breakpoint at the end of the system prompt
    caches both the tool definitions and the system text.
    """
    output_config: dict[str, Any] = {"effort": route.effort}
    if output_schema is not None:
        output_config["format"] = {"type": "json_schema", "schema": output_schema}
    request: dict[str, Any] = {
        "model": route.model,
        "max_tokens": max_tokens,
        "system": [
            {"type": "text", "text": f"{system}\n\n{DATA_BLOCK_RULE}", "cache_control": {"type": "ephemeral"}}
        ],
        "messages": list(messages),
        "output_config": output_config,
    }
    if tools:
        request["tools"] = list(tools)
    return request


class LLMClient:
    def __init__(
        self,
        config: ModelsConfig,
        db_path: Path,
        *,
        client: Any | None = None,
        api_key: SecretStr | None = None,
        offline: bool = False,
        max_attempts: int = 4,
        base_delay_s: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
    ) -> None:
        self.config = config
        self.db_path = db_path
        self.offline = offline
        self.max_attempts = max_attempts
        self.base_delay_s = base_delay_s
        self._sleep = sleep
        self._rng = rng or random.Random()  # noqa: S311 (jitter, not security)
        self._client = client
        self._api_key = api_key

    @property
    def client(self) -> Any:
        if self._client is None:
            key = self._api_key.get_secret_value() if self._api_key else None
            self._client = anthropic.Anthropic(api_key=key, max_retries=0)
        return self._client

    def month_to_date_spend(self) -> float:
        now = utcnow()
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        with connect(self.db_path) as conn:
            row = conn.execute(
                "select coalesce(sum(cost_usd), 0) from audit.llm_calls where created_at >= ?", [month_start]
            ).fetchone()
        return float(row[0]) if row else 0.0

    def create(
        self,
        *,
        task: str,
        system: str,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None,
        output_schema: dict[str, Any] | None = None,
        max_tokens: int = 16000,
        agent_run_id: str | None = None,
    ) -> LLMResponse:
        route = self._check_budget(task)
        request = build_request(
            route,
            system=system,
            messages=messages,
            tools=tools,
            output_schema=output_schema,
            max_tokens=max_tokens,
        )
        started = time.perf_counter()
        message = self._create_with_retries(request)
        ms = (time.perf_counter() - started) * 1000
        call_id, cost = self._log_usage(task, route, message.usage, ms, agent_run_id)
        return LLMResponse(message=message, call_id=call_id, cost_usd=cost)

    def _check_budget(self, task: str) -> TaskRoute:
        route = self.config.route(task)
        if route.offline_only and not self.offline:
            raise RuntimeError(f"task {task!r} is offline-only and cannot run on the request path")
        cap, spent = self.config.budget.monthly_soft_cap_usd, self.month_to_date_spend()
        if spent >= cap:
            raise BudgetExceededError(
                f"month-to-date LLM spend ${spent:.2f} has reached the ${cap:.2f} soft cap"
            )
        return route

    def _log_usage(
        self,
        task: str,
        route: TaskRoute,
        usage: Any,
        ms: float,
        agent_run_id: str | None,
        multiplier: float = 1.0,
    ) -> tuple[str, float]:
        tokens = {
            "input_tokens": int(usage.input_tokens or 0),
            "output_tokens": int(usage.output_tokens or 0),
            "cache_read_tokens": int(getattr(usage, "cache_read_input_tokens", 0) or 0),
            "cache_write_tokens": int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
        }
        cost = cost_usd(self.config.models[route.model], **tokens) * multiplier
        call_id = new_id("llm")
        with connect(self.db_path) as conn:
            conn.execute(
                "insert into audit.llm_calls values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    call_id,
                    agent_run_id,
                    task,
                    route.model,
                    route.effort,
                    tokens["input_tokens"],
                    tokens["output_tokens"],
                    tokens["cache_read_tokens"],
                    tokens["cache_write_tokens"],
                    cost,
                    ms,
                    utcnow(),
                ],
            )
        return call_id, cost

    def create_batch(
        self,
        *,
        task: str,
        system: str,
        items: dict[str, list[dict[str, Any]]],
        output_schema: dict[str, Any] | None = None,
        max_tokens: int = 1024,
        poll_s: float = 30.0,
        timeout_s: float = 24 * 3600,
    ) -> dict[str, Any]:
        """Run many independent requests through the Message Batches API (billed at a discount).

        ``items`` maps a custom id to its messages. Returns custom id -> message for each succeeded item;
        failed, expired or canceled items are simply absent. Usage is logged per succeeded item.
        """
        route = self._check_budget(task)
        if not items:
            return {}
        requests = [
            {
                "custom_id": cid,
                "params": build_request(
                    route,
                    system=system,
                    messages=msgs,
                    tools=None,
                    output_schema=output_schema,
                    max_tokens=max_tokens,
                ),
            }
            for cid, msgs in items.items()
        ]
        started = time.perf_counter()
        batch = self.client.messages.batches.create(requests=requests)
        waited = 0.0
        while batch.processing_status != "ended":
            if waited >= timeout_s:
                raise TimeoutError(f"batch {batch.id} did not finish within {timeout_s:.0f}s")
            self._sleep(poll_s)
            waited += poll_s
            batch = self.client.messages.batches.retrieve(batch.id)
        ms = (time.perf_counter() - started) * 1000
        out: dict[str, Any] = {}
        for entry in self.client.messages.batches.results(batch.id):
            if entry.result.type == "succeeded":
                message = entry.result.message
                self._log_usage(task, route, message.usage, ms / len(items), None, multiplier=BATCH_DISCOUNT)
                out[entry.custom_id] = message
        return out

    def _create_with_retries(self, request: dict[str, Any]) -> Any:
        for attempt in range(1, self.max_attempts + 1):
            try:
                return self.client.messages.create(**request)
            except RETRYABLE:
                if attempt == self.max_attempts:
                    raise
                delay = self.base_delay_s * 2 ** (attempt - 1) * (0.5 + self._rng.random())
                self._sleep(delay)
        raise AssertionError("unreachable")  # pragma: no cover
