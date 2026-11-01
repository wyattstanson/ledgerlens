/* Rate cards: what the panel would cost, where the savings come from, and what is billed above contract. */

import {
  callTool, card, countUp, chartTheme, chartBox, compactMoney, drawChart, el, findNumber, fmt,
  note, onAttached, section, titleCase, traced,
} from "../lib.js";

const STEPS = [
  ["current cost USD", "Current", "absolute"],
  ["leakage recovery USD", "Leakage", "relative"],
  ["fx retranslation USD", "FX", "relative"],
  ["rate harmonization USD", "Rate", "relative"],
  ["volume shift USD", "Volume", "relative"],
  ["vendor exits USD", "Exits", "relative"],
  ["new cost USD", "New cost", "total"],
];

export const title = "Rate cards";

export async function render(view, meta) {
  if (!meta.packs.ratecards) {
    view.replaceChildren(head(), note("The rate cards pipeline has not run on this warehouse yet.", "error"));
    return;
  }
  view.replaceChildren(head());

  const [bridge, panel, leakage, mapping] = await Promise.all([
    callTool("rc.savings_bridge", {}),
    callTool("rc.optimize_panel", {}),
    callTool("rc.leakage", {}),
    callTool("rc.mapping_status", {}),
  ]);

  view.append(
    section("Headline", [headline(bridge, panel)]),
    section("Where the saving comes from", [bridgeChart(bridge)]),
    section("The proposed panel", [panelCard(panel)]),
    section("Billed above contract, and billed twice", [leakageCards(leakage)]),
    section("Title mapping", [mappingCards(mapping)]),
    el("p", {
      class: "note note--advice",
      text: "A proposed panel is a proposal. It becomes real only when a named person approves it on the approvals page.",
    }),
  );
}

function head() {
  return el("div", { class: "page-head" }, [
    el("h1", { text: "Rate card consolidation" }),
    el("p", {
      text: "Billed titles mapped to standard roles, vendors compared on total cost rather than hourly rate, and every dollar of saving traced to the invoice lines behind it.",
    }),
  ]);
}

function headline(bridge, panel) {
  const current = findNumber(bridge, "current cost USD");
  const next = findNumber(bridge, "new cost USD");
  const kept = findNumber(panel, "panel vendors");
  const saving = current && next ? current.value - next.value : null;
  return el("div", { class: "cards" }, [
    card([
      el("span", { class: "kpi-label", text: "Cost today, 12 months" }),
      countUp(el("span", { class: "kpi-value" }, [traced(current, current ? compactMoney(current.value) : undefined)]), current, compactMoney),
    ]),
    card(
      [
        el("span", { class: "kpi-label", text: "Cost on the proposed panel" }),
        el("span", { class: "kpi-value" }, [traced(next, next ? compactMoney(next.value) : undefined)]),
        saving !== null
          ? el("span", {
              class: "badge badge--ok",
              text: `${compactMoney(saving)} less, ${((saving / current.value) * 100).toFixed(1)}%`,
            })
          : null,
      ],
      { delay: 60 },
    ),
    card(
      [
        el("span", { class: "kpi-label", text: "Vendors kept" }),
        countUp(el("span", { class: "kpi-value" }, [traced(kept)]), kept),
        el("span", { class: "kpi-note", text: `of a ${panel.data?.panel_size_limit} vendor limit, status ${panel.data?.status}` }),
      ],
      { delay: 120 },
    ),
  ]);
}

function bridgeChart(bridge) {
  const theme = chartTheme();
  const { plot, node } = chartBox("Current cost to new cost, by step");
  const points = STEPS.map(([label, name, measure]) => ({ name, measure, number: findNumber(bridge, label) })).filter(
    (p) => p.number,
  );
  if (!points.length) return note("The savings bridge has not been computed.");

  onAttached(() =>
    drawChart(
      plot,
      [
        {
          type: "waterfall",
          x: points.map((p) => p.name),
          y: points.map((p) => (p.measure === "total" ? 0 : p.measure === "absolute" ? p.number.value : -p.number.value)),
          measure: points.map((p) => p.measure),
          text: points.map((p) => compactMoney(p.number.value)),
          textposition: "outside",
          connector: { line: { color: theme.ink, width: 1 } },
          increasing: { marker: { color: theme.warn } },
          decreasing: { marker: { color: theme.accent } },
          totals: { marker: { color: theme.ink } },
          hovertemplate: "%{x}: %{text}<extra></extra>",
        },
      ],
      { yaxis: { title: { text: "USD over 12 months" }, tickformat: "$,.2s" } },
    ),
  );

  const rows = points.map((p) =>
    el("tr", {}, [el("td", { text: p.name }), el("td", { class: "num" }, [traced(p.number)])]),
  );
  return el("div", {}, [
    node,
    card([
      el("div", { class: "table-wrap" }, [
        el("table", {}, [
          el("thead", {}, [el("tr", {}, [el("th", { text: "Step" }), el("th", { class: "num", text: "USD" })])]),
          el("tbody", {}, rows),
        ]),
      ]),
      el("p", { class: "note note--advice", text: `Baseline months: ${(bridge.data?.baseline_months || []).join(", ")}` }),
    ]),
  ]);
}

function panelCard(panel) {
  const kept = panel.data?.kept_vendors || [];
  const cost = findNumber(panel, "panel annual cost at contracted rates USD");
  return card([
    el("p", {}, [
      "Cheapest panel that still covers every role, at ",
      traced(cost, cost ? compactMoney(cost.value) : undefined),
      " a year.",
    ]),
    el("p", { class: "scope", text: `Kept: ${kept.join(", ") || "none"}` }),
    el("p", {
      class: "kpi-note",
      text: `Critical roles held by at least two vendors: ${(panel.data?.critical_roles || []).length}. ${
        panel.data?.relaxed_constraint
          ? `Relaxed to stay feasible: ${panel.data.relaxed_constraint}.`
          : "No constraint had to be relaxed."
      }`,
    }),
  ]);
}

function leakageCards(leakage) {
  const kinds = ["overbill", "duplicate"];
  return el("div", {}, [
    el(
      "div",
      { class: "cards" },
      kinds.map((kind, i) => {
        const usd = findNumber(leakage, `${kind} USD all vendors`);
        const lines = findNumber(leakage, `${kind} lines all vendors`);
        return card(
          [
            el("span", { class: "kpi-label", text: kind === "overbill" ? "Billed above the contracted card" : "Duplicate invoice lines" }),
            el("span", { class: "kpi-value" }, [traced(usd)]),
            el("span", { class: "kpi-note" }, [lines ? traced(lines, `${fmt(lines.value, "count")} lines`) : "no count"]),
          ],
          { delay: i * 60 },
        );
      }),
    ),
    el("p", { class: "note note--advice", text: leakage.data?.threshold_rule || "" }),
  ]);
}

function mappingCards(mapping) {
  const statuses = mapping.data?.statuses || [];
  if (!statuses.length) return note("No titles have been mapped.");
  return el(
    "div",
    { class: "cards" },
    statuses.map((status, i) => {
      const number = findNumber(mapping, `titles ${status}`);
      return card(
        [
          el("span", { class: "kpi-label", text: titleCase(status) }),
          el("span", { class: "kpi-value" }, [traced(number)]),
        ],
        { delay: i * 50 },
      );
    }),
  );
}
