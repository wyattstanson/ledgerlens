"""FP&A forecasts from the models the owner reviewed in ``ml_pipeline/`` (BUILD_PROMPT 10.5, Gate C).

Each series gets the model chosen in ``ml_pipeline/model_choice.json`` (SeasonalNaive, AutoETS or AutoARIMA,
per the Gate B plan) in the exact form it had when it was reviewed and scored in the final test
(``ml_pipeline/model_forms.json``). It is refit on every closed period, re-estimating coefficients only, and
forecast from the latest closed month: 6 months for the monthly drivers and 13 weeks for weekly cash, with 80%
and 95% intervals. Left to search again on the longer history, AutoARIMA dropped the yearly pattern of weekly
collections, so a new form needs a new review.

Totals are not forecast directly. Collections, claims paid, LAE and net operating cash are rebuilt from the
driver forecasts through the driver tree, with profiles measured on 12 closed months that are old enough to be
fully paid (they end three months before the latest close):

- collection profile: share of a month's billed premium collected 1 month early, in the month, or 1-3+ later;
- payout profile: share of a month's approved claim amount paid, net of returns, 0-3+ months later;
- LAE profile: handling cost per approved claim, by the same lags.

Totals carry point values only; intervals belong to the driver and weekly-cash forecasts themselves.
statsforecast is imported inside the functions that fit: it needs statsmodels, whose compiled module Windows
App Control blocks on the dev machine, so forecasts are computed on Linux (CI, WSL) and stored, and the tools
only read the stored tables.
"""

import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
from pandas.tseries.frequencies import to_offset

from ledgerlens.core.db import connect, utcnow
from ledgerlens.fpa.forecast_series import (
    FREQ,
    LINES,
    complete_weeks,
    load_raw,
    monthly_long,
    timestamp,
    to_statsforecast,
    weekly_long,
)
from ledgerlens.settings import REPO_ROOT

MODEL_CHOICE = REPO_ROOT / "ml_pipeline" / "model_choice.json"
MODEL_FORMS = REPO_ROOT / "ml_pipeline" / "model_forms.json"
FINAL_TEST = REPO_ROOT / "ml_pipeline" / "final_test.json"
LEVELS = (80, 95)
HORIZON = {"monthly": 6, "weekly": 13}
# Month 6's early collections come from month 7's billing, so the monthly drivers run one month further.
DRIVER_LEAD = 1
LAGS = (-1, 0, 1, 2, 3)  # months between the driver's month and the cash month; 3 holds 3 or more
PROFILE_MONTHS = 12
SETTLE_MONTHS = 3  # profiles use months at least this old, so their cash has arrived


# A pick is a model name, whose form AutoETS or AutoARIMA searches on each fit, or a fixed form from
# ``fixed_form``, which is refit as it is.
Pick = str | dict[str, Any]


def load_choice(path: Path = MODEL_CHOICE) -> dict[str, dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _model(pick: Pick, freq: str) -> Any:
    from statsforecast.models import ARIMA, AutoARIMA, AutoETS, SeasonalNaive  # noqa: PLC0415 (module docstring)

    season = 12 if freq == "monthly" else 52
    # statsforecast's ETS supports seasons of at most 24 periods, so weekly ETS is non-seasonal (Gate B).
    ets_season = season if season <= 24 else 1  # noqa: PLR2004
    name = pick if isinstance(pick, str) else pick["model"]
    fixed = None if isinstance(pick, str) else pick
    if name == "SeasonalNaive":
        return SeasonalNaive(season_length=season)
    if name == "AutoETS" and fixed:
        return AutoETS(season_length=ets_season, model=fixed["ets"], damped=fixed["damped"])
    if name == "AutoETS":
        return AutoETS(season_length=ets_season)
    if name == "AutoARIMA" and fixed:
        return ARIMA(
            order=tuple(fixed["order"]),
            season_length=season,
            seasonal_order=tuple(fixed["seasonal_order"]),
            include_mean=fixed["include_mean"],
            include_drift=fixed["include_drift"],
        )
    if name == "AutoARIMA":
        return AutoARIMA(season_length=season)
    raise ValueError(f"unknown model {name!r}")


def _arima_form(arma: Any) -> str:
    """statsforecast's ``arma`` (p, q, P, Q, m, d, D) written the usual way: ARIMA(p,d,q)(P,D,Q)[m]."""
    p, q, sp, sq, m, d, sd = (int(x) for x in arma)
    return f"ARIMA({p},{d},{q})({sp},{sd},{sq})[{m}]"


def fixed_form(name: str, fitted: Any) -> dict[str, Any]:
    """The form a fitted ``name`` model settled on, with what ``_model`` needs to refit exactly that form."""
    spec = getattr(fitted, "model_", None) or {}
    if name == "AutoETS":
        error, trend, season = str(spec["method"])[4:-1].split(",")  # "ETS(M,Ad,N)"
        return {
            "model": name,
            "form": str(spec["method"]),
            "ets": error + trend[0] + season,
            "damped": trend.endswith("d"),
        }
    if name == "AutoARIMA":
        p, q, sp, sq, _, d, sd = (int(x) for x in spec["arma"])
        coef = list(spec["coef"])
        return {
            "model": name,
            "form": _arima_form(spec["arma"]),
            "order": [p, d, q],
            "seasonal_order": [sp, sd, sq],
            "include_mean": "intercept" in coef,
            "include_drift": "drift" in coef,
        }
    return {"model": name, "form": name}


def reviewed_form(spec: dict[str, Any]) -> str:
    """The form the chosen model had on the training rows at Gate C, written like ``fixed_form``."""
    form = str(spec.get("forms_on_training_rows", {}).get(spec["chosen"], spec["chosen"]))
    return _arima_form(json.loads(form)) if form.startswith("[") else form


def load_forms(choice: dict[str, dict[str, Any]], path: Path = MODEL_FORMS) -> dict[str, dict[str, Any]]:
    """The fixed form of every reviewed series, refused unless each is the model and form reviewed at Gate C.

    The file is written by ``ml_pipeline/04_eval.py`` (step 15) from the training rows.
    """
    forms = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    for uid, spec in choice.items():
        form, reviewed = forms.get(uid, {}), reviewed_form(spec)
        if form.get("model") != spec["chosen"] or form.get("form") != reviewed:
            raise ValueError(f"{path.name} does not hold the reviewed form of {uid} ({reviewed})")
    return {uid: forms[uid] for uid in choice}


def _fitted(history: pd.DataFrame, freq: str, picks: Mapping[str, Pick]) -> Iterator[tuple[str, Any]]:
    """Fit every series in ``history`` with its pick, one engine per distinct pick: (model name, engine)."""
    from statsforecast import StatsForecast  # noqa: PLC0415 (see module docstring)

    ids = sorted(history["unique_id"].unique())
    missing = [uid for uid in ids if uid not in picks]
    if missing:
        raise KeyError(f"no reviewed model for {missing}")
    groups: dict[str, list[str]] = {}
    for uid in ids:
        pick = picks[uid]
        groups.setdefault(pick if isinstance(pick, str) else json.dumps(pick, sort_keys=True), []).append(uid)
    for key in sorted(groups):
        pick = picks[groups[key][0]]
        engine = StatsForecast(models=[_model(pick, freq)], freq=FREQ[freq], n_jobs=1)
        engine.fit(df=history.loc[history["unique_id"].isin(groups[key])])
        yield (pick if isinstance(pick, str) else pick["model"]), engine


def fit_forms(history: pd.DataFrame, freq: str, picks: Mapping[str, Pick]) -> dict[str, dict[str, Any]]:
    """The fixed form each series' pick settles on when fit on ``history`` (``unique_id, ds, y``)."""
    forms = {}
    for name, engine in _fitted(history, freq, picks):
        uids: Any = engine.uids  # pandas or polars, depending on the input frame
        forms |= {str(uid): fixed_form(name, engine.fitted_[i, 0]) for i, uid in enumerate(uids)}
    return forms


def forecast_chosen(history: pd.DataFrame, freq: str, choice: Mapping[str, Pick], h: int) -> pd.DataFrame:
    """Fit each series' pick on ``history`` (``unique_id, ds, y``) and forecast ``h`` periods.

    Returns ``unique_id, ds, model, form, mean, lo-80, hi-80, lo-95, hi-95``. ``form`` is the form that was
    fit: a fixed pick's own, or whatever AutoETS or AutoARIMA settled on for this history.
    """
    frames = []
    for name, engine in _fitted(history, freq, choice):
        out = pd.DataFrame(engine.predict(h=h, level=list(LEVELS)))
        uids: Any = engine.uids  # pandas or polars, depending on the input frame
        forms = {str(uid): fixed_form(name, engine.fitted_[i, 0])["form"] for i, uid in enumerate(uids)}
        alias = next(
            col
            for col in out.columns
            if col not in ("unique_id", "ds") and "-lo-" not in col and "-hi-" not in col
        )
        renamed = out.rename(
            columns={
                alias: "mean",
                **{f"{alias}-{s}-{lv}": f"{s}-{lv}" for s in ("lo", "hi") for lv in LEVELS},
            }
        )
        frames.append(renamed.assign(model=name, form=[forms[str(uid)] for uid in renamed["unique_id"]]))
    columns = ["unique_id", "ds", "model", "form", "mean", "lo-80", "hi-80", "lo-95", "hi-95"]
    result = pd.DataFrame(pd.concat(frames, ignore_index=True)[columns])
    # Every target is positive (premium, counts, claim size, cash), so an interval never reaches below zero.
    for low in ("lo-80", "lo-95"):
        result[low] = result[low].clip(lower=0.0)
    return result.sort_values(by=["unique_id", "ds"], kind="mergesort", ignore_index=True)


def predict_rows(history: pd.DataFrame, rows: pd.DataFrame, choice: Mapping[str, Pick]) -> pd.DataFrame:
    """Forecasts for every row of ``rows`` (``ds, series, freq``) from ``history`` (the same plus ``value``).

    Rows must continue each series straight after its history. Returns, in the order of ``rows``:
    ``unique_id, ds, freq, chosen`` (the reviewed model's mean), its ``lo-80, hi-80, lo-95, hi-95``, and
    ``baseline`` (SeasonalNaive), so a holdout can be scored against both in one pass.
    """
    wanted = pd.DataFrame(
        {"unique_id": rows["series"], "ds": rows["ds"], "freq": rows["freq"], "row": range(len(rows))}
    )
    frames = []
    for freq in sorted(wanted["freq"].unique()):
        part = wanted.loc[wanted["freq"] == freq]
        past = to_statsforecast(history, freq)
        step = to_offset(FREQ[freq])
        last = past.groupby("unique_id")["ds"].max()
        for uid, dates in part.groupby("unique_id")["ds"]:
            expected = pd.date_range(last[uid] + step, periods=len(dates), freq=FREQ[freq])
            if not pd.DatetimeIndex(dates).sort_values().equals(expected):
                raise ValueError(f"{freq} rows for {uid} do not continue straight after the history")
        h = int(part.groupby("unique_id").size().max())
        chosen = forecast_chosen(past, freq, choice, h)
        baseline = forecast_chosen(past, freq, dict.fromkeys(choice, "SeasonalNaive"), h)
        merged = part.merge(
            chosen.drop(columns=["model", "form"]).rename(columns={"mean": "chosen"}), on=["unique_id", "ds"]
        )
        base = pd.DataFrame(
            {"unique_id": baseline["unique_id"], "ds": baseline["ds"], "baseline": baseline["mean"]}
        )
        merged = merged.merge(base, on=["unique_id", "ds"])
        if len(merged) != len(part):
            raise ValueError(f"{freq} rows do not continue straight after the history")
        frames.append(merged)
    out = pd.concat(frames, ignore_index=True).sort_values(by="row", kind="mergesort", ignore_index=True)
    return pd.DataFrame(out.drop(columns=["row"]))


# --- driver tree --------------------------------------------------------------------------------------------

PROFILE_SQL = """
with window_months as (select ?::date as first_month, ?::date as last_month),
dues as (
    select lob, date_trunc('month', due_date) as m, amount, paid_date
    from fpa.stg_fpa__instalments, window_months
    where date_trunc('month', due_date) between first_month and last_month
),
collection as (
    select lob, 'collection' as kind,
           least(greatest(date_diff('month', m, date_trunc('month', paid_date)), -1), 3) as lag,
           sum(amount) as amount
    from dues where paid_date is not null group by all
),
billed as (select lob, sum(amount) as base from dues group by all),
approved as (
    select lob, sum(severity) as amount, count(*) as claims
    from fpa.stg_fpa__claims, window_months
    where date_trunc('month', approved_date) between first_month and last_month
    group by all
),
legs as (
    select p.lob, date_trunc('month', p.approved_date) as m, p.amount, p.handling_cost, p.paid_ts,
           p.returned_ts
    from fpa.stg_fpa__payouts p, window_months
    where date_trunc('month', p.approved_date) between first_month and last_month
),
payout as (
    select lob, 'payout' as kind,
           least(greatest(date_diff('month', m, date_trunc('month', paid_ts)), 0), 3) as lag,
           sum(amount) as amount
    from legs where paid_ts is not null group by all
    union all
    select lob, 'payout', least(greatest(date_diff('month', m, date_trunc('month', returned_ts)), 0), 3),
           -sum(amount)
    from legs where returned_ts is not null group by all
),
lae as (
    select lob, 'lae' as kind,
           least(greatest(date_diff('month', m, date_trunc('month', paid_ts)), 0), 3) as lag,
           sum(handling_cost) as amount
    from legs where paid_ts is not null group by all
)
select c.lob, c.kind, c.lag, sum(c.amount) / any_value(b.base) as share
from collection c join billed b using (lob) group by all
union all
select p.lob, p.kind, p.lag, sum(p.amount) / any_value(a.amount)
from payout p join approved a using (lob) group by all
union all
select l.lob, l.kind, l.lag, sum(l.amount) / any_value(a.claims)
from lae l join approved a using (lob) group by all
"""


def profiles(warehouse: Path, latest_closed: str) -> pd.DataFrame:
    """``lob, kind, lag, share``: collection and payout shares per unit billed or approved; LAE per claim."""
    last = timestamp(f"{latest_closed}-01") - pd.DateOffset(months=SETTLE_MONTHS)
    first = last - pd.DateOffset(months=PROFILE_MONTHS - 1)
    with connect(warehouse, read_only=True) as conn:
        out = conn.execute(PROFILE_SQL, [first.date(), last.date()]).df()
    return out.sort_values(by=["lob", "kind", "lag"], kind="mergesort", ignore_index=True)


def rebuild_totals(drivers: pd.DataFrame, profile: pd.DataFrame, months: list[pd.Timestamp]) -> pd.DataFrame:
    """Totals for ``months`` from monthly driver values (actuals before the origin, forecasts after).

    ``drivers`` has ``unique_id`` (``driver|lob``), ``ds`` and ``value`` and must cover the lags behind
    each month. Returns ``ds, measure, value`` for collected_premium, claims_paid, lae, opex and
    net_operating_cash.
    """
    value = {
        (str(uid), timestamp(ds)): float(v)
        for uid, ds, v in zip(drivers["unique_id"], drivers["ds"], drivers["value"], strict=True)
    }
    share = {
        (str(lob), str(kind), int(lag)): float(s)
        for lob, kind, lag, s in zip(
            profile["lob"], profile["kind"], profile["lag"], profile["share"], strict=True
        )
    }

    def driver(name: str, lob: str, month: pd.Timestamp) -> float:
        key = (f"{name}|{lob}", month)
        if key not in value:
            raise KeyError(f"no {name} value for {lob} in {month:%Y-%m}")
        return value[key]

    rows = []
    for month in months:
        collected = claims = lae = 0.0
        for lob in LINES:
            for lag in LAGS:
                source = month - pd.DateOffset(months=lag)
                if (paid := share.get((lob, "collection", lag), 0.0)) != 0.0:
                    collected += paid * driver("due_premium", lob, source)
                payout, handling = share.get((lob, "payout", lag), 0.0), share.get((lob, "lae", lag), 0.0)
                if payout != 0.0 or handling != 0.0:
                    count = driver("approved_claims", lob, source)
                    claims += payout * count * driver("severity", lob, source)
                    lae += handling * count
        opex = value[("opex|all", month)]
        for measure, amount in (
            ("collected_premium", collected),
            ("claims_paid", claims),
            ("lae", lae),
            ("opex", opex),
            ("net_operating_cash", collected - claims - lae - opex),
        ):
            rows.append({"ds": month, "measure": measure, "value": round(amount, 2)})
    return pd.DataFrame(rows)


# --- the job ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ForecastRun:
    origin: str
    forecasts: pd.DataFrame
    totals: pd.DataFrame
    profile: pd.DataFrame
    models: pd.DataFrame  # per series: the reviewed choice, its reviewed and deployed form, and its scores


FORECAST_WORLD = "insurer"  # the reviewed models cover the insurer's series only (ADR 0005, decision 12)


def loaded_world(warehouse: Path) -> str | None:
    """The world the pipeline last ran on (``fpa.run_meta``); None if it has not run on the loaded one."""
    with duckdb.connect(str(warehouse), read_only=True) as con:
        try:
            row = con.execute("select variant from fpa.run_meta").fetchone()
        except duckdb.CatalogException:
            return None
    return None if row is None else str(row[0])


def run_forecast(
    warehouse: Path,
    choice_path: Path = MODEL_CHOICE,
    final_test: Path = FINAL_TEST,
    forms_path: Path = MODEL_FORMS,
) -> ForecastRun:
    """Refit the reviewed models in their reviewed forms on every closed period, forecast, rebuild totals and
    store everything."""
    world = loaded_world(warehouse)
    if world != FORECAST_WORLD:
        raise RuntimeError(
            f"the reviewed forecast models cover the {FORECAST_WORLD}'s series, and the pipeline last ran on "
            f"{world or 'no world'}: load the insurer and run `pipeline fpa` first (ADR 0005, decision 12)"
        )
    choice = load_choice(choice_path)
    picks = load_forms(choice, forms_path)
    raw = load_raw(warehouse)
    frame = pd.concat([monthly_long(raw), weekly_long(raw)], ignore_index=True)
    steps = {"monthly": HORIZON["monthly"] + DRIVER_LEAD, "weekly": HORIZON["weekly"]}
    forecasts = pd.concat(
        [
            forecast_chosen(to_statsforecast(frame, freq), freq, picks, steps[freq]).assign(freq=freq)
            for freq in ("monthly", "weekly")
        ],
        ignore_index=True,
    )
    forecasts["h"] = forecasts.groupby("unique_id").cumcount() + 1
    forecasts["in_horizon"] = forecasts["h"] <= forecasts["freq"].map(lambda f: HORIZON[str(f)])
    origin = raw.latest_closed
    profile = profiles(warehouse, origin)

    actual = monthly_long(raw)
    monthly_actual = pd.DataFrame(
        {"unique_id": actual["series"], "ds": actual["ds"], "value": actual["value"]}
    )
    fc = forecasts.loc[forecasts["freq"] == "monthly"]
    monthly_fc = pd.DataFrame({"unique_id": fc["unique_id"], "ds": fc["ds"], "value": fc["mean"]})
    future = sorted({timestamp(d) for d in fc.loc[fc["in_horizon"], "ds"]})
    totals = rebuild_totals(pd.concat([monthly_actual, monthly_fc], ignore_index=True), profile, future)

    tested = json.loads(final_test.read_text(encoding="utf-8")) if final_test.is_file() else {}
    deployed = dict(zip(forecasts["unique_id"], forecasts["form"], strict=True))
    unreviewed = sorted(uid for uid, spec in choice.items() if deployed.get(uid) != reviewed_form(spec))
    if unreviewed:
        raise ValueError(f"the refit settled on a form nobody reviewed for {unreviewed}; nothing was stored")
    models = pd.DataFrame(
        [
            {
                "row_key": uid,
                "unique_id": uid,
                "freq": spec["freq"],
                "chosen": spec["chosen"],
                "reviewed_form": reviewed_form(spec),
                "deployed_form": deployed.get(uid),
                "cv_mase": spec["cv_mase"],
                "baseline_cv_mase": spec["baseline_cv_mase"],
                "test_mase": tested.get("series", {}).get(uid, {}).get("chosen_mase"),
                "baseline_test_mase": tested.get("series", {}).get(uid, {}).get("baseline_mase"),
            }
            for uid, spec in choice.items()
        ]
    )
    stamp = utcnow()
    weeks = complete_weeks(raw)
    meta = pd.DataFrame(
        [
            {
                "row_key": "meta",
                "origin": origin,
                "last_complete_week": str(timestamp(weeks["week_start"].max()).date()),
                "generated_at": stamp,
                "model_choice_sha256": hashlib.sha256(choice_path.read_bytes()).hexdigest(),
                "model_forms_sha256": hashlib.sha256(forms_path.read_bytes()).hexdigest(),
            }
        ]
    )
    tables = {
        "forecasts": forecasts.assign(
            origin=origin, row_key=forecasts["unique_id"] + "|" + forecasts["ds"].dt.strftime("%Y-%m-%d")
        ),
        "forecast_totals": totals.assign(
            origin=origin, row_key=totals["measure"] + "|" + totals["ds"].dt.strftime("%Y-%m")
        ),
        "forecast_profiles": profile.assign(
            row_key=profile["lob"] + "|" + profile["kind"] + "|" + profile["lag"].astype(str)
        ),
        "forecast_models": models,
        "forecast_meta": meta,
    }
    with connect(warehouse) as conn:
        _write(conn, tables)
    return ForecastRun(origin, forecasts, totals, profile, models)


def _write(conn: duckdb.DuckDBPyConnection, tables: dict[str, pd.DataFrame]) -> None:
    # History kept from before forecasts carried their model form gets the column first: DuckDB will not
    # alter a table and then insert into it in one transaction.
    conn.execute("alter table if exists fpa.forecast_history add column if not exists form varchar")
    conn.execute("begin transaction")
    try:
        conn.execute("create schema if not exists fpa")
        for name, frame in tables.items():
            conn.register("_forecast_frame", frame)
            conn.execute(f"create or replace table fpa.{name} as select * from _forecast_frame")  # noqa: S608
            conn.unregister("_forecast_frame")
        # One row per origin and forecast, kept across runs so later actuals can be checked against them.
        conn.execute(
            "create table if not exists fpa.forecast_history as select * from fpa.forecasts where false"
        )
        conn.execute(
            "delete from fpa.forecast_history where origin = (select any_value(origin) from fpa.forecasts)"
        )
        conn.execute("insert into fpa.forecast_history by name select * from fpa.forecasts")
    except Exception:
        conn.execute("rollback")
        raise
    conn.execute("commit")
