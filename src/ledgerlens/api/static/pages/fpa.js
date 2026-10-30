/* FP&A: what the month did against plan, what the bridge says, and what the reviewed models expect. */

import {
  callTool, card, chartTheme, chartBox, drawChart, el, findNumber, fmt,
  matchNumbers, note, onAttached, section, titleCase, traced,
} from "../lib.js";

const COMPONENTS = ["volume", "mix", "rate", "timing", "residual"];

/** The theme colours are hex, so an alpha pair makes the interval band. */
const translucent = (hex) => (hex.startsWith("#") && hex.length === 7 ? `${hex}26` : hex);

export const title = "FP&A";

export async function render(view, meta) {
  const fpa = meta.fpa;
  if (!meta.packs.fpa || !fpa) {
    view.replaceChildren(head(null), note("The FP&A pipeline has not run on this warehouse yet.", "error"));
    return;
  }
  const period = fpa.latest_closed;
  view.replaceChildren(head(fpa));

  const [flags, bridge, income, aging, forecast, scenario] = await Promise.all([
    callTool("fpa.variance_flags", { period }),
    callTool("fpa.bridge", { measure: "net_operating_cash", period }),
    callTool("ledger.income_statement", { period }),
    callTool("ledger.ar_aging", {}),
    callTool("fpa.forecast", { view: "weekly_cash" }),
    callTool("fpa.scenario", {}),
  ]);

  view.append(
    section(`Variances, ${period}`, [flagCards(flags, period)]),
    section("Net operating cash: plan to actual", [bridgeChart(bridge, period)]),
    section(`Income statement, ${period}`, [incomeTable(income)]),
    section("Receivables", [agingCards(aging)]),
    section("13-week cash forecast", [weeklyChart(forecast)]),
    section("Scenario: 30% of check payouts moved to instant", [scenarioTable(scenario)]),
    el("p", { class: "note note--advice", text: bridge.data?.scope?.note || "" }),
  );
}

function head(fpa) {
  return el("div", { class: "page-head" }, [
    el("h1", { text: "Payments-aware FP&A" }),
    el("p", {
      text: "Collections, claims and cash rebuilt from the payment events behind them. Click any number to see the query and rows it came from.",
    }),
    fpa
      ? el("div", { class: "scope" }, [
          el("span", { text: fpa.entity }),
          el("span", { text: `closed through ${fpa.latest_closed}` }),
          el("span", { text: `${fpa.open_period} still open` }),
          el("span", { text: fpa.book }),
        ])
      : null,
  ]);
}

function flagCards(flags, period) {
  const rows = flags.data?.flags || [];
  if (!rows.length) return note("No material variances in this period.");
  return el(
    "div",
    { class: "cards" },
    rows.map((flag, i) => {
      const number = findNumber(flags, `variance ${flag.measure} ${period}`);
      const direction = flag.favourable ? "favourable" : "adverse";
      return card(
        [
          el("span", { class: "kpi-label", text: titleCase(flag.measure) }),
          el("span", { class: "kpi-value" }, [traced(number)]),
          el("span", {
            class: `badge badge--${flag.favourable ? "ok" : "flag"}`,
            text: `${flag.status}, ${direction}`,
          }),
          el("span", { class: "kpi-note", text: `${titleCase(flag.level)} level` }),
        ],
        { delay: i * 60 },
      );
    }),
  );
}

function bridgeChart(bridge, period) {
  if (!bridge.data?.found) return note(`No bridge for ${period}.`);
  const theme = chartTheme();
  const get = (prefix) => findNumber(bridge, `${prefix} net_operating_cash ${period}`);
  const plan = get("plan");
  const actual = get("actual");
  const parts = COMPONENTS.map((name) => ({ name, number: get(name) })).filter((p) => p.number);

  const { plot, node } = chartBox(`Plan to actual, ${period}`);
  const labels = ["Plan", ...parts.map((p) => titleCase(p.name)), "Actual"];
  const measures = ["absolute", ...parts.map(() => "relative"), "total"];
  const values = [plan?.value ?? 0, ...parts.map((p) => p.number.value), 0];

  onAttached(() =>
    drawChart(
      plot,
      [
        {
          type: "waterfall",
          orientation: "v",
          x: labels,
          y: values,
          measure: measures,
          text: values.map((v, i) => (measures[i] === "total" ? fmt(actual?.value ?? 0, "USD") : fmt(v, "USD"))),
          textposition: "outside",
          connector: { line: { color: theme.ink, width: 1 } },
          increasing: { marker: { color: theme.accent } },
          decreasing: { marker: { color: theme.warn } },
          totals: { marker: { color: theme.ink } },
          hovertemplate: "%{x}: %{text}<extra></extra>",
        },
      ],
      { yaxis: { title: { text: "USD" }, tickformat: "$,.0s" }, xaxis: { title: { text: "" } } },
    ),
  );

  const summary = el("ul", { class: "scope" }, [
    el("li", {}, ["Plan ", traced(plan)]),
    el("li", {}, ["Actual ", traced(actual)]),
    el("li", {}, ["Variance ", traced(get("variance"))]),
    el("li", { text: `${bridge.data.confidence} confidence: ${bridge.data.confidence_reason}` }),
  ]);
  return el("div", {}, [node, summary]);
}

function incomeTable(income) {
  if (!income.data?.found) return note("No income statement for this period.");
  const period = income.data.scope.period;
  const body = (income.data.lines || []).map((line) =>
    el("tr", {}, [el("td", { text: line }), el("td", { class: "num" }, [traced(findNumber(income, `${line} ${period}`))])]),
  );
  return card([
    el("div", { class: "table-wrap" }, [
      el("table", {}, [
        el("thead", {}, [el("tr", {}, [el("th", { text: "Line" }), el("th", { class: "num", text: "Amount" })])]),
        el("tbody", {}, body),
      ]),
    ]),
    el("p", { class: "note note--advice", text: income.data.basis }),
  ]);
}

function agingCards(aging) {
  const buckets = matchNumbers(aging, "receivable ").filter((n) => n.rest.endsWith(" amount"));
  if (!buckets.length) return note("No receivables recorded.");
  return el(
    "div",
    { class: "cards" },
    buckets.map((bucket, i) => {
      const name = bucket.rest.replace(" amount", "");
      const items = findNumber(aging, `receivable ${name} items`);
      return card(
        [
          el("span", { class: "kpi-label", text: titleCase(name) }),
          el("span", { class: "kpi-value" }, [traced(bucket)]),
          el("span", { class: "kpi-note" }, [items ? traced(items, `${fmt(items.value, "count")} invoices`) : "no count"]),
        ],
        { delay: i * 60 },
      );
    }),
  );
}

function weeklyChart(forecast) {
  if (!forecast.data?.found) return note("No forecast has been written to this warehouse yet.");
  const theme = chartTheme();
  const { plot, node } = chartBox(`Weekly cash from ${forecast.data.origin}, with 80% intervals`);

  const series = (prefix, label) => {
    const points = matchNumbers(forecast, `${prefix} cash_weekly|`)
      .map((n) => ({ week: n.rest.split(" ").pop(), value: n.value, series: n.rest.split(" ")[0] }))
      .filter((p) => p.series === label);
    return { x: points.map((p) => p.week), y: points.map((p) => p.value) };
  };

  onAttached(() => {
    const traces = [];
    for (const [key, name, colour] of [
      ["collected_premium", "Premium collected", theme.accent],
      ["claims_paid", "Claims paid", theme.warn],
    ]) {
      const mid = series("forecast", key);
      const low = series("80% low", key);
      const high = series("80% high", key);
      traces.push(
        { type: "scatter", mode: "lines", x: high.x, y: high.y, line: { width: 0 }, hoverinfo: "skip", showlegend: false },
        {
          type: "scatter", mode: "lines", x: low.x, y: low.y, line: { width: 0 }, fill: "tonexty",
          fillcolor: translucent(colour), hoverinfo: "skip", showlegend: false,
        },
        {
          type: "scatter", mode: "lines+markers", name, x: mid.x, y: mid.y,
          line: { color: colour, width: 2 }, marker: { size: 5 },
          hovertemplate: `${name}, %{x}: %{y:$,.0f}<extra></extra>`,
        },
      );
    }
    drawChart(plot, traces, {
      showlegend: true,
      legend: { orientation: "h", y: -0.2 },
      yaxis: { title: { text: "USD per week" }, tickformat: "$,.0s" },
      xaxis: { title: { text: "Week beginning" } },
    });
  });

  const models = Object.entries(forecast.data.models || {})
    .map(([series, model]) => `${series}: ${model}`)
    .join(" · ");
  return el("div", {}, [node, el("p", { class: "note note--advice", text: `Chosen models, ${models}` })]);
}

function scenarioTable(scenario) {
  const metrics = scenario.data?.metrics || [];
  if (!metrics.length) return note("No scenario has been computed.");
  const rows = metrics.map((metric) =>
    el("tr", {}, [
      el("td", { text: titleCase(metric) }),
      el("td", { class: "num" }, [traced(findNumber(scenario, `baseline ${metric}`))]),
      el("td", { class: "num" }, [traced(findNumber(scenario, `scenario ${metric}`))]),
      el("td", { class: "num" }, [traced(findNumber(scenario, `change ${metric}`))]),
    ]),
  );
  return card([
    el("div", { class: "table-wrap" }, [
      el("table", {}, [
        el("thead", {}, [
          el("tr", {}, [
            el("th", { text: "Metric" }),
            el("th", { class: "num", text: "Baseline" }),
            el("th", { class: "num", text: "Scenario" }),
            el("th", { class: "num", text: "Change" }),
          ]),
        ]),
        el("tbody", {}, rows),
      ]),
    ]),
    el("p", {
      class: "note note--advice",
      text: `Changed drivers: ${(scenario.data.changed_drivers || []).map((d) => `${d.driver} (${d.segment})`).join(", ")}`,
    }),
  ]);
}
