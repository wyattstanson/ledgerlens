/* The method page: the arithmetic behind every figure, with the inputs live.

   A consultancy shows you the conclusion. This shows the equation, lets you move its inputs, and
   recomputes in front of you. The worked examples run in the browser on the numbers you set; the
   figures on the pack pages come from the governed tools and carry lineage, which the panels say. */

import { card, el, fmt, section } from "../lib.js";

export const title = "Method";

/* ---------- Formula definitions ----------
   Each has the equation as it appears in the spec, the inputs you can move, and the arithmetic. */

const MODELS = [
  {
    id: "net-cash",
    group: "FP&A",
    name: "Net operating cash",
    equation: "net_operating_cash = collected_premium - claims_paid - lae - opex",
    why: "The driver tree rebuilds cash from the payment events underneath it, so a miss can be attributed rather than just reported.",
    inputs: [
      { key: "collected", label: "Collected premium", value: 3797090, step: 10000, unit: "USD" },
      { key: "claims", label: "Claims paid", value: 1795905, step: 10000, unit: "USD" },
      { key: "lae", label: "Loss adjustment expense", value: 4270, step: 500, unit: "USD" },
      { key: "opex", label: "Operating expense", value: 873704, step: 10000, unit: "USD" },
    ],
    compute: (v) => [
      { label: "Net operating cash", value: v.collected - v.claims - v.lae - v.opex, unit: "USD" },
    ],
  },
  {
    id: "collections",
    group: "FP&A",
    name: "Collected premium",
    equation: "collected = Σ over methods ( due × method_mix × on_time_rate ) + late_collections",
    why: "Collections depend on how customers pay, not just on what was billed. Change the mix and the cash moves.",
    inputs: [
      { key: "due", label: "Due premium", value: 4000000, step: 100000, unit: "USD" },
      { key: "cardMix", label: "Card share", value: 0.34, step: 0.01, unit: "share", max: 1 },
      { key: "cardRate", label: "Card paid on time", value: 0.93, step: 0.01, unit: "share", max: 1 },
      { key: "achMix", label: "ACH share", value: 0.3, step: 0.01, unit: "share", max: 1 },
      { key: "achRate", label: "ACH paid on time", value: 0.97, step: 0.01, unit: "share", max: 1 },
      { key: "late", label: "Late cash from prior months", value: 180000, step: 10000, unit: "USD" },
    ],
    compute: (v) => {
      const card = v.due * v.cardMix * v.cardRate;
      const ach = v.due * v.achMix * v.achRate;
      return [
        { label: "From card", value: card, unit: "USD" },
        { label: "From ACH", value: ach, unit: "USD" },
        { label: "Collected (these two methods plus late cash)", value: card + ach + v.late, unit: "USD" },
      ];
    },
  },
  {
    id: "bridge",
    group: "FP&A",
    name: "Variance bridge",
    equation: "volume = (ΣQa - ΣQp) × P̄p    mix = Σ (share_a - share_p) × ΣQa × Pp    rate = Σ (Pa - Pp) × Qa",
    why: "Actual minus plan split into parts that sum to the whole. A property test proves the parts add to the variance to the cent.",
    inputs: [
      { key: "qa", label: "Actual quantity", value: 1200, step: 25, unit: "count" },
      { key: "qp", label: "Plan quantity", value: 1000, step: 25, unit: "count" },
      { key: "pa", label: "Actual rate", value: 92, step: 1, unit: "USD" },
      { key: "pp", label: "Plan rate", value: 100, step: 1, unit: "USD" },
    ],
    compute: (v) => {
      const volume = (v.qa - v.qp) * v.pp;
      const rate = (v.pa - v.pp) * v.qa;
      const total = v.qa * v.pa - v.qp * v.pp;
      return [
        { label: "Volume", value: volume, unit: "USD" },
        { label: "Rate", value: rate, unit: "USD" },
        { label: "Residual (one segment, so mix is zero)", value: total - volume - rate, unit: "USD" },
        { label: "Total variance", value: total, unit: "USD", strong: true },
      ];
    },
  },
  {
    id: "mase",
    group: "FP&A",
    name: "Forecast error, MASE",
    equation: "MASE = mean|forecast - actual| ÷ mean|seasonal_naive - actual|",
    why: "Scaled against the naive forecast, so 1.00 means no better than repeating last season. Below 1 earns its place.",
    inputs: [
      { key: "err", label: "Mean absolute error of the model", value: 64000, step: 1000, unit: "USD" },
      { key: "naive", label: "Mean absolute error of seasonal naive", value: 100000, step: 1000, unit: "USD" },
    ],
    compute: (v) => {
      const mase = v.naive === 0 ? null : v.err / v.naive;
      return [
        { label: "MASE", value: mase, unit: "ratio", decimals: 2, strong: true },
        { label: mase !== null && mase < 1 ? "Beats the baseline" : "Does not beat the baseline", value: null },
      ];
    },
  },
  {
    id: "impact",
    group: "HR",
    name: "Impact ratio, the four-fifths rule",
    equation: "impact_ratio = selection_rate_group ÷ selection_rate_highest_group",
    why: "Below 0.80 is the threshold US guidance uses. Measured per stage, on the process, never on a person.",
    inputs: [
      { key: "passA", label: "Group A passed", value: 76, step: 1, unit: "count" },
      { key: "totalA", label: "Group A assessed", value: 171, step: 1, unit: "count" },
      { key: "passB", label: "Group B passed", value: 145, step: 1, unit: "count" },
      { key: "totalB", label: "Group B assessed", value: 202, step: 1, unit: "count" },
    ],
    compute: (v) => {
      const a = v.totalA ? v.passA / v.totalA : 0;
      const b = v.totalB ? v.passB / v.totalB : 0;
      const highest = Math.max(a, b);
      const ratio = highest ? Math.min(a, b) / highest : 0;
      const small = v.totalA < 5 || v.totalB < 5;
      return [
        { label: "Group A selection rate", value: a, unit: "rate" },
        { label: "Group B selection rate", value: b, unit: "rate" },
        { label: "Impact ratio", value: ratio, unit: "ratio", decimals: 2, strong: true },
        {
          label: small
            ? "Suppressed: a group under five people is never published"
            : ratio < 0.8
              ? "Below four fifths, so this cell is flagged"
              : "At or above four fifths",
          value: null,
        },
      ];
    },
  },
  {
    id: "bh",
    group: "HR",
    name: "Benjamini-Hochberg correction",
    equation: "keep test i (ranked by p) while p(i) ≤ (i ÷ m) × α",
    why: "Run 469 tests at 5% and about 23 look significant by chance. One correction across every test in the run, registered before it runs.",
    inputs: [
      { key: "m", label: "Tests in the family", value: 469, step: 1, unit: "count" },
      { key: "alpha", label: "Alpha", value: 0.05, step: 0.01, unit: "ratio", max: 1 },
      { key: "rank", label: "Rank of this test", value: 5, step: 1, unit: "count" },
    ],
    compute: (v) => {
      const threshold = v.m ? (v.rank / v.m) * v.alpha : 0;
      return [
        { label: "False positives expected with no correction", value: v.m * v.alpha, unit: "count", decimals: 1 },
        { label: "This test's threshold", value: threshold, unit: "ratio", decimals: 6, strong: true },
      ];
    },
  },
  {
    id: "normalise",
    group: "Procurement",
    name: "Rate normalisation",
    equation: "FTE: monthly ÷ standard_hours      Outcome: rate × (1 - productivity_commitment)",
    why: "The cheapest hourly rate is often not the cheapest work. Normalising to one basis is what exposes that.",
    inputs: [
      { key: "monthly", label: "Monthly FTE rate", value: 14400, step: 200, unit: "USD" },
      { key: "hours", label: "Standard hours a month", value: 160, step: 5, unit: "count" },
      { key: "outcome", label: "Outcome-based rate", value: 110, step: 5, unit: "USD" },
      { key: "commitment", label: "Productivity commitment", value: 0.2, step: 0.05, unit: "share", max: 1 },
      { key: "tm", label: "Time and materials rate", value: 100, step: 5, unit: "USD" },
    ],
    compute: (v) => {
      const fte = v.hours ? v.monthly / v.hours : 0;
      const effective = v.outcome * (1 - v.commitment);
      const rates = [
        { name: "the FTE deal", rate: fte },
        { name: "the outcome deal", rate: effective },
        { name: "time and materials", rate: v.tm },
      ].sort((x, y) => x.rate - y.rate);
      return [
        { label: "FTE, per hour", value: fte, unit: "USD", decimals: 2 },
        { label: "Outcome deal, effective", value: effective, unit: "USD", decimals: 2 },
        { label: `Cheapest is ${rates[0].name}`, value: null, strong: true },
      ];
    },
  },
  {
    id: "panel",
    group: "Procurement",
    name: "Panel optimiser",
    equation: "minimise Σ p(v,r) × h(v,r)   subject to   Σ_v h(v,r) = D(r),   h(v,r) ≤ D(r) × keep(v),   Σ keep(v) ≤ N",
    why: "Every role's demand must be met, hours only go to vendors you keep, and the panel has a size limit. Solved exactly, not by ranking.",
    inputs: [
      { key: "demand", label: "Hours needed for this role", value: 10000, step: 500, unit: "count" },
      { key: "cheap", label: "Cheapest kept vendor rate", value: 33.42, step: 0.5, unit: "USD" },
      { key: "incumbent", label: "Incumbent rate", value: 41.1, step: 0.5, unit: "USD" },
      { key: "cap", label: "Most of the demand one vendor may hold", value: 0.6, step: 0.05, unit: "share", max: 1 },
    ],
    compute: (v) => {
      const atCheap = Math.min(1, v.cap) * v.demand;
      const rest = v.demand - atCheap;
      const cost = atCheap * v.cheap + rest * v.incumbent;
      return [
        { label: "Cost at the concentration cap", value: cost, unit: "USD" },
        { label: "Cost if the incumbent kept it all", value: v.demand * v.incumbent, unit: "USD" },
        { label: "Saving", value: v.demand * v.incumbent - cost, unit: "USD", strong: true },
      ];
    },
  },
];

/* ---------- Rendering ---------- */

export async function render(view) {
  view.replaceChildren(head());
  const groups = ["FP&A", "HR", "Procurement"];
  for (const group of groups) {
    const models = MODELS.filter((m) => m.group === group);
    view.append(section(group, models.map((model, i) => panel(model, i))));
  }
  view.append(
    el("p", {
      class: "note note--advice",
      text: "These panels recompute in your browser from the inputs you set, so you can see how each equation behaves. The figures on the pack pages are computed by the same equations on the server, where each one carries a lineage id back to its rows.",
    }),
  );
}

function head() {
  return el("div", { class: "page-head" }, [
    el("h1", { text: "The arithmetic" }),
    el("p", {
      text: "Every number this platform shows comes from one of these. Move an input and watch the result change, then go to the pack page and click the same figure to see the rows it came from.",
    }),
  ]);
}

function panel(model, index) {
  const state = Object.fromEntries(model.inputs.map((i) => [i.key, i.value]));
  const results = el("dl", { class: "formula-out" });

  const recompute = () => {
    const rows = model.compute(state);
    results.replaceChildren(
      ...rows.flatMap((row) => {
        const value =
          row.value === null || row.value === undefined
            ? null
            : row.decimals !== undefined
              ? row.value.toLocaleString("en-US", {
                  minimumFractionDigits: row.decimals,
                  maximumFractionDigits: row.decimals,
                })
              : fmt(row.value, row.unit);
        const dt = el("dt", { class: row.strong ? "is-strong" : null, text: row.label });
        const dd = el("dd", { class: row.strong ? "is-strong" : null, text: value ?? "" });
        return [dt, dd];
      }),
    );
  };

  const controls = model.inputs.map((input) => {
    const id = `${model.id}-${input.key}`;
    const output = el("output", { for: id, class: "field-value", text: display(state[input.key], input) });
    const slider = el("input", {
      id,
      type: "range",
      min: "0",
      max: String(input.max ?? (input.value * 2.5 || 1)),
      step: String(input.step),
      value: String(input.value),
      oninput: (event) => {
        state[input.key] = Number(event.target.value);
        output.textContent = display(state[input.key], input);
        recompute();
      },
    });
    return el("div", { class: "field field--slider" }, [
      el("label", { for: id, text: input.label }),
      slider,
      output,
    ]);
  });

  recompute();

  return card(
    [
      el("h3", { class: "formula-name", text: model.name }),
      el("code", { class: "formula", text: model.equation }),
      el("p", { class: "kpi-note", text: model.why }),
      el("div", { class: "formula-body" }, [el("div", { class: "formula-inputs" }, controls), results]),
    ],
    { delay: index * 60 },
  );
}

function display(value, input) {
  if (input.unit === "share" || input.unit === "rate") return `${(value * 100).toFixed(0)}%`;
  if (input.unit === "ratio") return value.toFixed(2);
  return fmt(value, input.unit);
}
