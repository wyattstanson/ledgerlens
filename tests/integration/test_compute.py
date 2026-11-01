"""Client-supplied figures: the same functions, and none of the claims that need rows.

The governance rule these routes must not break is the one about lineage. A number the warehouse
produced resolves to its rows; a number four typed figures produced does not, so these answers carry
a disclaimer and no lineage id, and nothing here reads or writes the warehouse.
"""

import io

import duckdb
import pytest
from fastapi.testclient import TestClient

from ledgerlens.api.main import app
from ledgerlens.api.routes_compute import MAX_CSV_BYTES, MAX_SEGMENTS

HEADER = "segment,q_actual,amount_actual,q_plan,amount_plan\n"


@pytest.fixture
def client() -> TestClient:
    # No settings override and no db fixture: a compute route that needed either would fail here.
    return TestClient(app)


def csv_file(body: str) -> dict[str, tuple[str, io.BytesIO, str]]:
    return {"file": ("segments.csv", io.BytesIO(body.encode("utf-8")), "text/csv")}


def test_the_bridge_parts_sum_to_the_variance(client: TestClient):
    response = client.post(
        "/api/compute/fpa.variance_bridge",
        json={
            "segments": [
                {
                    "key": "auto",
                    "q_actual": 1200,
                    "amount_actual": 240000,
                    "q_plan": 1000,
                    "amount_plan": 190000,
                },
                {
                    "key": "home",
                    "q_actual": 800,
                    "amount_actual": 200000,
                    "q_plan": 900,
                    "amount_plan": 216000,
                },
            ],
            "cash_actual": 430000,
            "cash_plan": 400000,
        },
    )
    assert response.status_code == 200
    body = response.json()["result"]
    assert body["variance"] == 30000.0
    assert sum(body["components"].values()) == pytest.approx(body["variance"], abs=0.01)


def test_an_answer_from_typed_figures_carries_no_lineage_id(client: TestClient):
    body = client.post(
        "/api/compute/fpa.materiality",
        json={"measure": "claims_paid", "kind": "expense", "actual": 1_100_000, "plan": 1_000_000},
    ).json()
    assert "lineage_id" not in str(body), "a figure with no rows behind it must not claim lineage"
    assert "no lineage id" in body["disclaimer"]


def test_materiality_uses_the_same_thresholds_as_the_pack(client: TestClient):
    # Operating level: 3% or $25,000. 100,000 over a 1,000,000 plan is 10%, so material and a FLAG.
    body = client.post(
        "/api/compute/fpa.materiality",
        json={"measure": "claims_paid", "kind": "expense", "actual": 1_100_000, "plan": 1_000_000},
    ).json()["result"]
    assert body["status"] == "material"
    assert body["level"] == "operating"
    assert body["label"] == "FLAG:"
    assert body["favourable"] is False, "an expense above plan is adverse"


def test_net_operating_cash_is_judged_at_the_statement_level(client: TestClient):
    # 30,000 over a 1,000,000 plan is 3%: material at the operating level, only watch at 5%/$50,000.
    body = client.post(
        "/api/compute/fpa.materiality",
        json={"measure": "net_operating_cash", "kind": "net", "actual": 1_030_000, "plan": 1_000_000},
    ).json()["result"]
    assert body["level"] == "financial_statement"
    assert body["status"] == "watch"
    assert body["label"] is None


def test_a_documented_one_time_item_is_not_flagged(client: TestClient):
    payload = {"measure": "opex", "kind": "expense", "actual": 1_100_000, "plan": 1_000_000}
    assert client.post("/api/compute/fpa.materiality", json=payload).json()["result"]["status"] == "material"
    body = client.post("/api/compute/fpa.materiality", json={**payload, "one_time": True}).json()["result"]
    assert body["status"] == "one_time"
    assert body["label"] is None


def test_an_outcome_rate_is_normalised_on_total_cost(client: TestClient):
    # The spec example: $110 with a 20% commitment is $88 a baseline hour, under a $100 T&M rate.
    body = client.post(
        "/api/compute/rc.card_rate",
        json={"pricing_model": "outcome", "rate": 110, "productivity_commitment": 0.2},
    ).json()["result"]
    assert body["sticker_usd_per_hour"] == pytest.approx(110.0)
    assert body["effective_usd_per_hour"] == pytest.approx(88.0)


def test_a_fixed_price_sow_says_why_it_has_no_hourly_rate(client: TestClient):
    body = client.post("/api/compute/rc.card_rate", json={"pricing_model": "fixed", "rate": 20000}).json()
    assert body["result"]["effective_usd_per_hour"] is None
    assert "hours" in body["result"]["note"]


@pytest.mark.parametrize(
    ("payload", "why"),
    [
        ({"segments": [], "cash_actual": 1, "cash_plan": 1}, "no segments at all"),
        ({"cash_actual": 1, "cash_plan": 1}, "no segments key"),
        (
            {
                "segments": [{"key": "a", "q_actual": -1, "amount_actual": 1, "q_plan": 1, "amount_plan": 1}],
                "cash_actual": 1,
                "cash_plan": 1,
            },
            "a negative quantity",
        ),
    ],
)
def test_bad_input_is_refused_rather_than_computed(client: TestClient, payload: dict, why: str):
    assert client.post("/api/compute/fpa.variance_bridge", json=payload).status_code == 422, why


def test_an_unknown_field_is_refused_so_a_typo_is_never_silently_ignored(client: TestClient):
    payload = {
        "measure": "opex",
        "kind": "expense",
        "actual": 1,
        "plan": 1,
        "one_tyme": True,
    }
    assert client.post("/api/compute/fpa.materiality", json=payload).status_code == 422


def test_a_csv_of_segments_gives_the_same_bridge_as_the_form(client: TestClient):
    body = HEADER + "auto,1200,240000,1000,190000\nhome,800,200000,900,216000\n"
    uploaded = client.post(
        "/api/compute/fpa.variance_bridge/csv",
        params={"cash_actual": 430000, "cash_plan": 400000},
        files=csv_file(body),
    )
    assert uploaded.status_code == 200
    assert uploaded.json()["segments"] == 2

    typed = client.post(
        "/api/compute/fpa.variance_bridge",
        json={
            "segments": [
                {
                    "key": "auto",
                    "q_actual": 1200,
                    "amount_actual": 240000,
                    "q_plan": 1000,
                    "amount_plan": 190000,
                },
                {
                    "key": "home",
                    "q_actual": 800,
                    "amount_actual": 200000,
                    "q_plan": 900,
                    "amount_plan": 216000,
                },
            ],
            "cash_actual": 430000,
            "cash_plan": 400000,
        },
    )
    assert uploaded.json()["result"] == typed.json()["result"]


def test_a_csv_missing_a_column_says_which_one(client: TestClient):
    response = client.post(
        "/api/compute/fpa.variance_bridge/csv",
        params={"cash_actual": 1, "cash_plan": 1},
        files=csv_file("segment,q_actual,amount_actual,q_plan\nauto,1,2,3\n"),
    )
    assert response.status_code == 400
    assert "amount_plan" in response.json()["detail"]


def test_a_csv_with_an_unreadable_figure_names_the_line(client: TestClient):
    response = client.post(
        "/api/compute/fpa.variance_bridge/csv",
        params={"cash_actual": 1, "cash_plan": 1},
        files=csv_file(HEADER + "auto,1,2,3,4\nhome,1,2,3,not-a-number\n"),
    )
    assert response.status_code == 400
    assert response.json()["detail"].startswith("line 3:")


def test_a_csv_with_only_a_header_is_refused(client: TestClient):
    response = client.post(
        "/api/compute/fpa.variance_bridge/csv",
        params={"cash_actual": 1, "cash_plan": 1},
        files=csv_file(HEADER),
    )
    assert response.status_code == 400
    assert "no segments" in response.json()["detail"]


def test_an_oversized_upload_is_refused_before_it_is_parsed(client: TestClient):
    row = "auto,1,2,3,4\n"
    body = HEADER + row * ((MAX_CSV_BYTES // len(row)) + 200)
    response = client.post(
        "/api/compute/fpa.variance_bridge/csv",
        params={"cash_actual": 1, "cash_plan": 1},
        files=csv_file(body),
    )
    assert response.status_code == 413


def test_too_many_segments_are_refused(client: TestClient):
    body = HEADER + "auto,1,2,3,4\n" * (MAX_SEGMENTS + 1)
    response = client.post(
        "/api/compute/fpa.variance_bridge/csv",
        params={"cash_actual": 1, "cash_plan": 1},
        files=csv_file(body),
    )
    assert response.status_code == 400
    assert str(MAX_SEGMENTS) in response.json()["detail"]


def test_a_compute_route_never_opens_the_warehouse(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    """Nothing here reads rows, so nothing here may take the warehouse lock or write an audit run.

    One that quietly opened it would queue behind every pack panel on a shared instance, and would
    put rows into a trail that is meant to record warehouse reads.
    """

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("a compute route opened the warehouse")

    monkeypatch.setattr(duckdb, "connect", refuse)
    for path, payload in (
        ("/api/compute/rc.card_rate", {"pricing_model": "tm", "rate": 100}),
        (
            "/api/compute/fpa.materiality",
            {"measure": "opex", "kind": "expense", "actual": 2, "plan": 1},
        ),
    ):
        assert client.post(path, json=payload).status_code == 200, path
