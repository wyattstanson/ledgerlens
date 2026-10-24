# ADR 0005: FP&A bridge, materiality choices and the ledger guard

- Status: proposed (Phase 4, for the owner's review in PR #4; decision 12 is Phase 4b, reviewed in its own PR)
- Affects: BUILD_PROMPT section 10 (Phase 4), ADR 0001 (where this ADR departs from its wording), section 9.2
  (the answer lineage check, shared by every pack).

## Context

Phase 4 builds the payments-aware FP&A pack on a synthetic insurer with a NetSuite-shaped ledger. ADR 0001
fixes the finance conventions (materiality, sign, aging, labels, answer shape, connector tool order). Building
it raised choices ADR 0001 does not settle, and three places where the code does something other than its
wording. The phase review (correctness, conventions, security red team) also found a gap in the shared
answer lineage check.

## Decisions

1. **Bridge basis.** Volume, mix and rate are computed on the event basis (premium billed, claims approved,
   payouts made); timing is the gap between cash and events, actual against plan: `(cash_a - event_a) -
   (cash_p - event_p)`. Each part is rounded to cents and the residual takes the rounding, so every bridge
   sums to its variance exactly. Because the residual absorbs whatever the other parts miss, the sum alone
   proves nothing: the Dagster check `bridge_sums_to_variance` also fails a residual above 2 cents (8 for net
   operating cash, which combines four measures). Claims timing also carries ordinary month-end noise (claims
   are lumpy), so it is reported, not read as the planted check lag; that lag shows in the check
   `days_to_pay` driver.
2. **Materiality levels and labels.** Net operating cash is the statement-level total and uses the financial
   statement thresholds; premium collected, claims paid, LAE and opex use the operating thresholds. A
   material variance is labelled `CRITICAL:` at the financial statement level and `FLAG:` at the operating
   level (`materiality.risk_label`). Thresholds compare by multiplication, so exactly 3% of plan reaches 3%;
   with no plan the percentage is undefined and only the dollar amounts apply.
3. **Seasonality (departs from ADR 0001).** ADR 0001 asks compute to check known seasonality before flagging.
   Here the plan itself is seasonal (claim frequency by month, billing by days in the month), so a variance
   is already measured against a seasonal expectation and no separate seasonality screen runs. One-time items
   (`one_time_items` in `config/fpa.yaml`) and reclassifications (an equal and opposite expense variance in
   the same month) are screened as ADR 0001 asks.
4. **Confidence.** High for a closed, reconciled period; medium for the open period; low when the plan or
   the actuals for a period are missing (ADR 0001's "missing or inconsistent data"), with the reason stored
   beside it.
5. **Raw ledger query (departs from ADR 0001's tool order).** The ledger adapter has the connector's raw
   query, last in its order, but it is not an agent tool: the report-shaped tools and saved searches answer
   every Phase 4 question. Should it become one, it is already fenced: DuckDB's own parse tree must show one
   SELECT, no `*`, no table functions, only built-in functions that describe data (not settings, secrets,
   sequences or macros), and only ledger tables, comma joins and subqueries included. It returns at most
   1,000 rows and is interrupted after ten seconds.
6. **Read-only, no external access.** Every FP&A read opens the warehouse read-only with
   `enable_external_access` off (`core.db.NO_EXTERNAL_ACCESS`), so no query can read files, environment
   variables or attach other databases. DuckDB refuses a second connection to one file with a different
   configuration while the first is open, so dbt runs in its own process (as under Dagster), with any
   environment variable named like a key, token, secret, password or credential removed.
7. **Adapters carry their source.** The payments feed and the ledger return `Rows`: the data plus the query
   and tables behind it. Tools get them from `payments_feed()` and `ledger()`, the one place to swap in a real
   feed, and lineage cites what that feed actually ran.
8. **Answer shape.** The four FP&A agents share an ADR 0001 brief (headline, scope, drivers, risks with the
   tools' labels, action with owner role and timing in words, "Not financial advice.") appended to the shared
   system prompt. The shape is asked for, not validated: the core answer schema is shared by every pack.
   What is enforced is the lineage check, which every answer must pass.
9. **Answer lineage check, hardened (all packs).** The check now finds digits glued to words ("USD9400000")
   and numbers written with a scale word ("9.4 million"), refuses numbers above ten written in words, and
   requires a percentage to cite a ratio or share and a dollar amount to cite a USD value. A single digit glued
   to a word ("Q3", "v2") is still read as an identifier.
10. **Forecasting through the review gates.** Models are chosen only in `ml_pipeline/` under the ml-pipeline
    skill, with the owner's approval at each gate (`ml_pipeline/PIPELINE.md`). Production never re-chooses:
    `fpa/forecast.py` refits each reviewed model (`ml_pipeline/model_choice.json`) in the exact form that was
    reviewed and scored in the final test (`ml_pipeline/model_forms.json`) on every closed period,
    re-estimating coefficients only, and refuses to store a forecast from any other form. Left to search again
    on the longer history, AutoARIMA dropped the yearly pattern of weekly collections; a new form is a
    candidate for the next review, not applied directly. Data preparation lives in `fpa/forecast_series.py` so
    the notebooks and production cannot diverge. Totals are rebuilt from driver forecasts through the driver
    tree and carry point values only.
11. **Where forecasts are computed.** statsforecast imports statsmodels, whose compiled module Windows App
    Control blocks (intermittently) on the dev machine. Forecasts are therefore computed on Linux (CI, WSL,
    later the hosted job), stored in the warehouse, and only read by the `fpa.forecast` tool, which loads no
    model code. A forecast history keyed by origin lets the Dagster check compare each newly closed month with
    the forecast made a month earlier.
12. **Processor variant (Phase 4b, BUILD_PROMPT 10.1).** `generate --pack fpa --variant processor` writes a
    second synthetic world, a card processor: merchants in three verticals, authorization attempts by card type
    and initiation (a person, or an AI agent buying for one), fees, interchange, T+1 settlement batches, and
    chargebacks that merchants repay or that are written off after 120 days. It reuses everything after the
    marts: the bridge, materiality, driver table shape, tools, run tables and the four agents, unchanged.
    - *One world at a time.* `raw_fpa.world_meta` names the loaded world; the other world's own tables are
      written empty, so one dbt project builds either and every test runs. `pipeline fpa --variant X` refuses
      another world. Loading a world drops `fpa.run_meta`, so every tool refuses to answer until the pipeline
      has run on it (the staging views would otherwise show the new world under the old world's scope); the
      pipeline also stops if another world is loaded while dbt runs, and a scenario approval records its
      world, so it cannot be executed on the other world's scenario.
    - *Same ledger, same tests.* Settlements post as NetSuite `deposit`s, recoveries as `custpymt`. The
      reconciliation test now signs events by the feed's credit/debit indicator (`direction`), the receivable
      test ties every receivable account to its open items, and two tests tie settlements to captures and
      chargeback events to disputes.
    - *Bases.* Fee revenue and interchange: segments vertical by card type, quantity payment volume, event basis
      captured, cash as settled (timing is settlement across month ends). Chargebacks: segments vertical by
      initiation, quantity disputes raised, event basis the amount merchants have not repaid, cash debited less
      recovered. In the open month a recovery still in flight counts as not repaid until it lands, so that
      month's chargeback split leans to rate and away from timing; closed months have had their recoveries.
      The income statement shows fees and interchange as captured and chargeback losses as written off.
    - *Materiality unchanged.* The world is sized so gross fee revenue (about $37M) and net revenue after
      interchange (about $16M) both sit inside ADR 0001's band, and the run checks both. ADR 0001's aging
      buckets apply to all receivables: unsettled batches are not yet due; merchant chargebacks age from the
      dispute.
    - *Plants.* More digital-goods merchants (fee revenue volume), debit traffic moving to premium credit
      (interchange mix), a premium interchange increase (interchange rate), faster growth in agent-initiated
      traffic, which is disputed more (chargeback volume), and issuers declining more of it (agent
      authorization rate). The plan is the closed-form expectation of the same day-level inputs, starting from
      the opening book exactly, so plan and actual differ only by the plants and by chance.
    - *Scenario.* Send 30% of agent-initiated attempts with network tokens, assumed to be approved at the rate,
      and disputed at the rate, observed for human-initiated purchases in the same vertical over the latest 13
      weeks; a chargeback costs the vertical's observed dollars not repaid per chargeback, counted only on
      disputes old enough for their repayment to have landed.
    - *Golden cases* live in `tests/evals/golden/fpa_processor/`, apart from the insurer's, and CI runs the
      offline evals once per world.

## Consequences

- Deferred, each named in PR #4: the waterfall chart and the scenario demo in the UI (Phase 5 builds the
  web UI). For the processor (decision 12): a weekly cash mart and forecasting, which go through the same
  ml-pipeline gates; the reviewed forecast models cover the insurer's series only.
- Known forecast weakness: homeowners claim size keeps SeasonalNaive, which repeats last year's large losses
  into the same months a year on. A robust alternative would be a model change and needs a new review.
- Residual risks: the answer shape is prompt-level only; lineage for bridge numbers stops at the pipeline's
  `fpa.bridge` and `fpa.bridge_totals` rows, which the pipeline rebuilds from the marts on every run; a real
  feed behind the adapters must report its own source honestly.
