/* Audit and cost: every run, every tool call and every dollar the language model has spent. */

import { card, el, fmt, getJson, note, section } from "../lib.js";

export const title = "Audit and cost";

export async function render(view) {
  view.replaceChildren(head());
  const audit = await getJson("/api/audit");
  view.append(
    section("Model spend", [costCards(audit)]),
    section("Recent runs", [runsTable(audit.runs)]),
    section("Recent tool calls", [callsTable(audit.tool_calls)]),
    el("p", {
      class: "note note--advice",
      text: "Costs come from each API response's usage, never an estimate. A run with no model calls cost nothing: the numbers were computed locally.",
    }),
  );
}

function head() {
  return el("div", { class: "page-head" }, [
    el("h1", { text: "Audit and cost" }),
    el("p", {
      text: "Every answer leaves a trail: the run that asked, the tools it called, the rows each number came from and what the model cost.",
    }),
  ]);
}

function costCards(audit) {
  const cost = audit.cost || {};
  const cards = [
    ["Spent so far", `$${Number(cost.total_usd || 0).toFixed(4)}`, `${fmt(cost.calls || 0, "count")} model calls`],
    ["Input tokens", fmt(cost.input_tokens || 0, "count"), `${fmt(cost.cache_read_tokens || 0, "count")} read from cache`],
    ["Output tokens", fmt(cost.output_tokens || 0, "count"), ""],
  ];
  const tiles = cards.map(([label, value, hint], i) =>
    card(
      [
        el("span", { class: "kpi-label", text: label }),
        el("span", { class: "kpi-value", text: value }),
        hint ? el("span", { class: "kpi-note", text: hint }) : null,
      ],
      { delay: i * 60 },
    ),
  );
  const byModel = audit.by_model || [];
  if (byModel.length) {
    tiles.push(
      card(
        [
          el("span", { class: "kpi-label", text: "By model" }),
          el(
            "ul",
            { class: "scope" },
            byModel.map((row) => el("li", { text: `${row.model} (${row.task}): $${Number(row.cost_usd).toFixed(4)}` })),
          ),
        ],
        { delay: 180 },
      ),
    );
  }
  return el("div", { class: "cards" }, tiles);
}

function runsTable(runs) {
  if (!runs?.length) return note("No runs recorded yet.");
  const rows = runs.map((run) =>
    el("tr", {}, [
      el("td", { text: run.agent_id }),
      el("td", { text: run.user }),
      el("td", { text: String(run.started_at).replace("T", " ").slice(0, 19) }),
      el("td", { class: "num", text: fmt(run.tool_calls, "count") }),
      el("td", {}, [
        el("span", { class: `badge badge--${run.status === "completed" ? "ok" : "flag"}`, text: run.status }),
      ]),
    ]),
  );
  return card([
    el("div", { class: "table-wrap" }, [
      el("table", {}, [
        el("thead", {}, [
          el("tr", {}, [
            el("th", { text: "Agent" }),
            el("th", { text: "Caller" }),
            el("th", { text: "Started" }),
            el("th", { class: "num", text: "Tool calls" }),
            el("th", { text: "Status" }),
          ]),
        ]),
        el("tbody", {}, rows),
      ]),
    ]),
  ]);
}

function callsTable(calls) {
  if (!calls?.length) return note("No tool calls recorded yet.");
  const rows = calls.map((call) =>
    el("tr", {}, [
      el("td", { text: call.tool_name }),
      el("td", { text: String(call.started_at).replace("T", " ").slice(0, 19) }),
      el("td", { class: "num", text: call.ms === null ? "" : `${Math.round(call.ms)} ms` }),
      el("td", { class: "num", text: fmt(call.numbers, "count") }),
      el("td", {}, [
        el("span", { class: `badge badge--${call.status === "ok" ? "ok" : "flag"}`, text: call.status }),
      ]),
    ]),
  );
  return card([
    el("div", { class: "table-wrap" }, [
      el("table", {}, [
        el("thead", {}, [
          el("tr", {}, [
            el("th", { text: "Tool" }),
            el("th", { text: "Started" }),
            el("th", { class: "num", text: "Took" }),
            el("th", { class: "num", text: "Numbers" }),
            el("th", { text: "Status" }),
          ]),
        ]),
        el("tbody", {}, rows),
      ]),
    ]),
  ]);
}
