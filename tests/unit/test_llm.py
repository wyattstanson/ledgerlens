import random
from pathlib import Path

import pytest
from fakes import FakeClient, fetch_one, rate_limit_error, text_message, usage

from ledgerlens.core.db import connect, utcnow
from ledgerlens.core.guards import DATA_BLOCK_RULE
from ledgerlens.core.llm import (
    BudgetExceededError,
    LLMClient,
    ModelPrice,
    ModelsConfig,
    build_request,
    cost_usd,
)


def make_client(
    models_config: ModelsConfig, db: Path, script: list, sleeps: list[float] | None = None
) -> LLMClient:
    return LLMClient(
        models_config,
        db,
        client=FakeClient(script),
        sleep=(sleeps.append if sleeps is not None else lambda _s: None),
        rng=random.Random(0),
    )


def test_shipped_routing_matches_the_spec(models_config: ModelsConfig):
    assert models_config.route("title_mapping").model == "claude-haiku-5-5"
    assert models_config.route("pack_agent").model == "claude-sonnet-5-5"
    assert models_config.route("pack_agent").effort == "medium"
    assert models_config.route("eval_judge").offline_only is True
    with pytest.raises(KeyError, match="no model route"):
        models_config.route("nope")


def test_routes_to_unpriced_models_are_rejected(tmp_path: Path):
    path = tmp_path / "models.yaml"
    path.write_text(
        "models: {}\ntasks: {t: {model: claude-x, effort: low}}\nbudget: {monthly_soft_cap_usd: 1}\n"
    )
    with pytest.raises(ValueError, match="without prices"):
        ModelsConfig.from_yaml(path)


def test_cost_uses_list_prices_cache_reads_and_cache_writes():
    sonnet = ModelPrice(input_per_mtok=2.0, output_per_mtok=10.0, cache_read_ratio=0.05)
    base = cost_usd(
        sonnet, input_tokens=1_000_000, output_tokens=1_000_000, cache_read_tokens=0, cache_write_tokens=0
    )
    assert base == pytest.approx(12.0)
    cached = cost_usd(
        sonnet, input_tokens=0, output_tokens=0, cache_read_tokens=1_000_000, cache_write_tokens=0
    )
    assert cached == pytest.approx(0.10)
    written = cost_usd(
        sonnet, input_tokens=0, output_tokens=0, cache_read_tokens=0, cache_write_tokens=1_000_000
    )
    assert written == pytest.approx(2.5)


def test_request_puts_cacheable_prefix_first_with_effort_and_format(models_config: ModelsConfig):
    schema = {"type": "object", "properties": {}, "additionalProperties": False}
    request = build_request(
        models_config.route("pack_agent"),
        system="You are an analyst.",
        messages=[{"role": "user", "content": "q"}],
        tools=[{"name": "t", "description": "d", "input_schema": schema, "strict": True}],
        output_schema=schema,
        max_tokens=1000,
    )
    [system_block] = request["system"]
    assert system_block["cache_control"] == {"type": "ephemeral"}
    assert DATA_BLOCK_RULE in system_block["text"]
    assert request["output_config"] == {
        "effort": "medium",
        "format": {"type": "json_schema", "schema": schema},
    }
    assert request["model"] == "claude-sonnet-5-5"
    assert "thinking" not in request  # budget_tokens thinking is rejected by current models


def test_create_logs_usage_and_cost_from_the_response(models_config: ModelsConfig, db: Path):
    message = text_message("hi")
    message.usage = usage(input_tokens=1000, output_tokens=200, cache_read=3000, cache_write=500)
    client = make_client(models_config, db, [message])
    response = client.create(task="pack_agent", system="s", messages=[{"role": "user", "content": "q"}])
    expected = (1000 * 2 + 200 * 10 + 3000 * 2 * 0.05 + 500 * 2 * 1.25) / 1_000_000
    assert response.cost_usd == pytest.approx(expected)
    with connect(db) as conn:
        row = fetch_one(
            conn,
            "select task, model, effort, input_tokens, output_tokens, cache_read_tokens, "
            "cache_write_tokens, cost_usd from audit.llm_calls",
        )
    assert row[:7] == ("pack_agent", "claude-sonnet-5-5", "medium", 1000, 200, 3000, 500)
    assert row[7] == pytest.approx(expected)


def test_rate_limits_are_retried_with_growing_jittered_backoff(models_config: ModelsConfig, db: Path):
    sleeps: list[float] = []
    client = make_client(
        models_config, db, [rate_limit_error(), rate_limit_error(), text_message("ok")], sleeps
    )
    client.create(task="pack_agent", system="s", messages=[{"role": "user", "content": "q"}])
    assert len(sleeps) == 2
    assert 0.5 <= sleeps[0] <= 1.5  # base 1s x jitter [0.5, 1.5)
    assert 1.0 <= sleeps[1] <= 3.0  # doubled


def test_retries_give_up_after_max_attempts(models_config: ModelsConfig, db: Path):
    import anthropic  # noqa: PLC0415

    client = make_client(models_config, db, [rate_limit_error() for _ in range(4)])
    with pytest.raises(anthropic.RateLimitError):
        client.create(task="pack_agent", system="s", messages=[{"role": "user", "content": "q"}])
    with connect(db) as conn:
        assert conn.execute("select count(*) from audit.llm_calls").fetchone() == (0,)


def test_budget_cap_refuses_new_calls(models_config: ModelsConfig, db: Path):
    with connect(db) as conn:
        conn.execute(
            "insert into audit.llm_calls values ('llm_x', null, 't', 'm', 'low', 0, 0, 0, 0, 20.0, 1, ?)",
            [utcnow()],
        )
    fake = FakeClient([text_message("never sent")])
    client = LLMClient(models_config, db, client=fake)
    with pytest.raises(BudgetExceededError, match="soft cap"):
        client.create(task="pack_agent", system="s", messages=[{"role": "user", "content": "q"}])
    assert fake.messages.requests == []


def test_offline_only_tasks_stay_off_the_request_path(models_config: ModelsConfig, db: Path):
    online = make_client(models_config, db, [text_message("x")])
    with pytest.raises(RuntimeError, match="offline-only"):
        online.create(task="eval_judge", system="s", messages=[{"role": "user", "content": "q"}])
    offline = LLMClient(models_config, db, client=FakeClient([text_message("x")]), offline=True)
    offline.create(task="eval_judge", system="s", messages=[{"role": "user", "content": "q"}])


def test_client_construction_needs_no_key_until_first_call(models_config: ModelsConfig, db: Path):
    client = LLMClient(models_config, db)  # must not raise without ANTHROPIC_API_KEY
    assert client.month_to_date_spend() == 0.0
