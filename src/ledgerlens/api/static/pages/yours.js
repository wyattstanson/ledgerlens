/* Your numbers: a client's own figures through the functions the packs run.

   Every panel here posts to /api/compute, which runs the tested function and returns a result with
   no lineage id, because figures typed into a form have no rows behind them. The disclaimer the API
   sends back is shown with every answer rather than written into the page, so the page cannot
   soften it. */

import { card, el, fmt, postJson, section } from "../lib.js";

export const title = "Your numbers";

const MODELS = [
  ["tm", "Time and materials"],
  ["fte", "Monthly FTE"],
  ["outcome", "Outcome based"],
  ["fixed", "Fixed-price SOW"],
];

const KINDS = [
  ["expense", "Expense"],
  ["revenue", "Revenue"],
  ["net", "Net"],
];

const SEED_SEGMENTS = [
  { key: "Auto", q_actual: 1200, amount_actual: 240000, q_plan: 1000, amount_plan: 190000 },
  { key: "Home", q_actual: 800, amount_actual: 200000, q_plan: 900, amount_plan: 216000 },
];

export async function render(view) {
  view.replaceChildren(
    el("div", { class: "page-head" }, [
      el("h1", { text: "Your numbers" }),
      el("p", {
        text:
          "Put your own figures through the same tested functions the pack pages use. Nothing is stored, " +
          "and nothing here reads the warehouse, so these answers carry no lineage id.",
      }),
    ]),
    section("Variance bridge", [bridgePanel()]),
    section("Is it material?", [materialityPanel()]),
    section("Vendor rate, normalised", [ratePanel()]),
  );
}

/* ---------- Shared pieces ---------- */

function field(label, input) {
  return el("div", { class: "field" }, [el("label", { text: label }), input]);
}

function numberInput(value, { step = "any", min } = {}) {
  return el("input", { type: "number", step, value: String(value), ...(min === undefined ? {} : { min }) });
}

function select(options, value) {
  const node = el(
    "select",
    {},
    options.map(([key, label]) => el("option", { value: key, text: label })),
  );
  node.value = value;
  return node;
}

/** The disclaimer the API returned, under the answer it belongs to. */
function answer(children, disclaimer) {
  return el("div", { class: "yours-answer" }, [
    ...children,
    el("p", { class: "yours-caveat", text: disclaimer }),
  ]);
}

function failure(error) {
  const message = error.status === 422 ? "Check the figures: one of them is missing or out of range." : null;
  return el("p", { class: "note note--error", text: message || error.message || "That did not compute." });
}

/** A button that disables itself while its work is in flight, so a double click cannot double post. */
function runner(label, out, work) {
  const button = el("button", { type: "submit", class: "btn" }, [
    el("span", { text: label }),
    el("span", { class: "btn-orb", "aria-hidden": "true", text: "↗" }),
  ]);
  const form = el("form", { class: "yours-run" }, [button]);
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    button.disabled = true;
    out.replaceChildren(el("p", { class: "note note--advice", text: "Computing" }));
    try {
      out.replaceChildren(await work());
    } catch (error) {
      out.replaceChildren(failure(error));
    } finally {
      button.disabled = false;
    }
  });
  return form;
}

/* ---------- Variance bridge ---------- */

function bridgePanel() {
  const rows = [];
  const body = el("tbody");
  const out = el("div", { class: "yours-out" });

  const addRow = (seed) => {
    const inputs = {
      key: el("input", { type: "text", value: seed.key }),
      q_actual: numberInput(seed.q_actual, { min: "0" }),
      amount_actual: numberInput(seed.amount_actual),
      q_plan: numberInput(seed.q_plan, { min: "0" }),
      amount_plan: numberInput(seed.amount_plan),
    };
    const row = el("tr", {}, [
      ...Object.values(inputs).map((input) => el("td", {}, [input])),
      el("td", {}, [
        el("button", {
          type: "button",
          class: "yours-drop",
          text: "Remove",
          "aria-label": "Remove this segment",
          onclick: () => {
            if (rows.length === 1) return; // the bridge needs at least one segment to mean anything
            rows.splice(rows.indexOf(entry), 1);
            row.remove();
          },
        }),
      ]),
    ]);
    const entry = { inputs, row };
    rows.push(entry);
    body.append(row);
  };

  SEED_SEGMENTS.forEach(addRow);

  const cashActual = numberInput(430000);
  const cashPlan = numberInput(400000);

  const table = el("div", { class: "table-wrap" }, [
    el("table", {}, [
      el("thead", {}, [
        el("tr", {}, [
          el("th", { text: "Segment" }),
          el("th", { text: "Quantity, actual" }),
          el("th", { text: "Amount, actual" }),
          el("th", { text: "Quantity, plan" }),
          el("th", { text: "Amount, plan" }),
          el("th", { text: "" }),
        ]),
      ]),
      body,
    ]),
  ]);

  const run = runner("Build the bridge", out, async () => {
    const payload = {
      segments: rows.map(({ inputs }) => ({
        key: inputs.key.value || "segment",
        q_actual: Number(inputs.q_actual.value),
        amount_actual: Number(inputs.amount_actual.value),
        q_plan: Number(inputs.q_plan.value),
        amount_plan: Number(inputs.amount_plan.value),
      })),
      cash_actual: Number(cashActual.value),
      cash_plan: Number(cashPlan.value),
    };
    const body = await postJson("/api/compute/fpa.variance_bridge", payload);
    return answer([bridgeResult(body.result)], body.disclaimer);
  });

  return card([
    el("p", { class: "kpi-note", text: "Actual against plan, split the way the FP&A page splits it." }),
    table,
    el("div", { class: "controls" }, [
      field("Cash, actual", cashActual),
      field("Cash, plan", cashPlan),
      el("button", {
        type: "button",
        class: "btn btn--quiet",
        text: "Add a segment",
        onclick: () => addRow({ key: "", q_actual: 0, amount_actual: 0, q_plan: 0, amount_plan: 0 }),
      }),
    ]),
    run,
    csvDrop(cashActual, cashPlan, out),
    out,
  ]);
}

function bridgeResult(result) {
  const parts = Object.entries(result.components);
  return el("dl", { class: "formula-out" }, [
    ...parts.flatMap(([name, value]) => [
      el("dt", { text: name[0].toUpperCase() + name.slice(1) }),
      el("dd", { text: fmt(value, "USD") }),
    ]),
    el("dt", { class: "is-strong", text: "Variance" }),
    el("dd", { class: "is-strong", text: fmt(result.variance, "USD") }),
  ]);
}

/** The same bridge from a file, for a client who has the segments in a ledger extract already. */
function csvDrop(cashActual, cashPlan, out) {
  const picker = el("input", { type: "file", accept: ".csv,text/csv" });
  const form = el("form", { class: "yours-upload" }, [
    field("Or upload segments as CSV", picker),
    el("button", { type: "submit", class: "btn btn--quiet", text: "Upload and compute" }),
  ]);
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const file = picker.files?.[0];
    if (!file) {
      out.replaceChildren(el("p", { class: "note note--error", text: "Choose a CSV file first." }));
      return;
    }
    const data = new FormData();
    data.append("file", file);
    const query = new URLSearchParams({ cash_actual: cashActual.value, cash_plan: cashPlan.value });
    out.replaceChildren(el("p", { class: "note note--advice", text: "Reading the file" }));
    try {
      const response = await fetch(`/api/compute/fpa.variance_bridge/csv?${query}`, {
        method: "POST",
        body: data,
      });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(body.detail || "That file could not be read.");
      out.replaceChildren(
        answer(
          [
            el("p", { class: "kpi-note", text: `${body.segments} segments read from the file.` }),
            bridgeResult(body.result),
          ],
          body.disclaimer,
        ),
      );
    } catch (error) {
      out.replaceChildren(failure(error));
    }
  });
  return el("details", { class: "yours-more" }, [
    el("summary", { text: "Upload a segment extract instead" }),
    el("p", {
      class: "note note--advice",
      text: "Columns: segment, q_actual, amount_actual, q_plan, amount_plan. The file is parsed and dropped, never stored.",
    }),
    form,
  ]);
}

/* ---------- Materiality ---------- */

function materialityPanel() {
  const measure = el("input", { type: "text", value: "claims_paid" });
  const kind = select(KINDS, "expense");
  const actual = numberInput(1100000);
  const plan = numberInput(1000000);
  const oneTime = el("input", { type: "checkbox" });
  const out = el("div", { class: "yours-out" });

  const run = runner("Check it", out, async () => {
    const body = await postJson("/api/compute/fpa.materiality", {
      measure: measure.value || "measure",
      kind: kind.value,
      actual: Number(actual.value),
      plan: Number(plan.value),
      one_time: oneTime.checked,
    });
    const r = body.result;
    const pct = r.pct_of_plan === null ? "no plan" : `${(r.pct_of_plan * 100).toFixed(1)}% of plan`;
    return answer(
      [
        el("p", { class: `badge badge--${r.status === "material" ? "flag" : "ok"}`, text: r.status }),
        el("p", {
          class: "kpi-value",
          text: `${fmt(r.variance, "USD")} ${r.favourable ? "favourable" : "adverse"}`,
        }),
        el("p", {
          class: "kpi-note",
          text:
            `${pct}, judged at the ${r.level.replace("_", " ")} level: ` +
            `${(r.threshold_pct * 100).toFixed(0)}% or ${fmt(r.threshold_usd, "USD")}.` +
            (r.label ? ` Labelled ${r.label}` : ""),
        }),
      ],
      body.disclaimer,
    );
  });

  return card([
    el("p", { class: "kpi-note", text: "One variance against the thresholds in ADR 0001." }),
    el("div", { class: "controls" }, [
      field("Measure", measure),
      field("Kind", kind),
      field("Actual", actual),
      field("Plan", plan),
      field("Documented one-time item", oneTime),
    ]),
    run,
    out,
  ]);
}

/* ---------- Vendor rate ---------- */

function ratePanel() {
  const model = select(MODELS, "outcome");
  const rate = numberInput(110, { min: "0" });
  const fx = numberInput(1, { min: "0" });
  const commitment = numberInput(0.2, { min: "0" });
  const hours = numberInput(160, { min: "1" });
  const out = el("div", { class: "yours-out" });

  const run = runner("Normalise it", out, async () => {
    const body = await postJson("/api/compute/rc.card_rate", {
      pricing_model: model.value,
      rate: Number(rate.value),
      usd_per_unit: Number(fx.value),
      productivity_commitment: Number(commitment.value),
      standard_hours: Number(hours.value),
    });
    const r = body.result;
    if (r.effective_usd_per_hour === null) {
      return answer([el("p", { class: "kpi-note", text: r.note })], body.disclaimer);
    }
    return answer(
      [
        el("p", { class: "kpi-value", text: `${fmt(r.effective_usd_per_hour, "USD")} per baseline hour` }),
        el("p", {
          class: "kpi-note",
          text: `Sticker rate ${fmt(r.sticker_usd_per_hour, "USD")} per hour before the commitment.`,
        }),
      ],
      body.disclaimer,
    );
  });

  return card([
    el("p", {
      class: "kpi-note",
      text: "A rate card line on one basis, so two vendors priced differently can be compared.",
    }),
    el("div", { class: "controls" }, [
      field("Pricing model", model),
      field("Rate", rate),
      field("USD per currency unit", fx),
      field("Productivity commitment", commitment),
      field("Standard hours a month", hours),
    ]),
    run,
    out,
  ]);
}
