/* Interview quality and fairness. Interviewers, panels, loops and stages: never a candidate.
   Cells with fewer than five people are suppressed before they reach this page. */

import {
  callTool, card, countUp, chartTheme, chartBox, drawChart, el, findNumber, fmt,
  note, onAttached, section, titleCase, traced,
} from "../lib.js";

const ROLE = "backend";
const STAGE = "technical";
const DIMENSION = "sex";

export const title = "Interviews";

export async function render(view, meta) {
  if (!meta.packs.interviews) {
    view.replaceChildren(head(null), note("The interviews pipeline has not run on this warehouse yet.", "error"));
    return;
  }
  view.replaceChildren(head(null));

  const [plan, flags, panels, validity, impact, evidence, tagger] = await Promise.all([
    callTool("iv.plan_status", {}),
    callTool("iv.interviewer_flags", {}),
    callTool("iv.panel_agreement", {}),
    callTool("iv.predictive_validity", {}),
    callTool("iv.impact_ratios", { role_id: ROLE, stage: STAGE, dimension: DIMENSION }),
    callTool("iv.process_quality", { metric: "evidence_quality" }),
    callTool("iv.tagger_quality", {}),
  ]);

  view.replaceChildren(
    head(plan),
    section("What this run found", [planCards(plan)]),
    section("Flagged interviewers", [flagList(flags)]),
    section("Panel agreement", [panelChart(panels)]),
    section("Does the interview predict performance?", [validityChart(validity)]),
    section(`Selection rates, ${titleCase(ROLE)} at the ${STAGE} stage, by ${DIMENSION}`, [impactTable(impact)]),
    section("Evidence cited in notes", [evidenceCards(evidence, tagger)]),
    el("p", {
      class: "note note--advice",
      text: "Not legal advice. Aggregates only: no candidate is scored or ranked here, and any group under five people is suppressed.",
    }),
  );
}

function head(plan) {
  return el("div", { class: "page-head" }, [
    el("h1", { text: "Interview quality and fairness" }),
    el("p", {
      text: "The process is measured, never the person. Thresholds and tests were registered before the run, and every comparison shares one Benjamini-Hochberg correction.",
    }),
    plan
      ? el("div", { class: "scope" }, [
          el("span", { text: `plan ${plan.data.plan} v${plan.data.plan_version}` }),
          el("span", { text: `hash ${String(plan.data.plan_sha256).slice(0, 12)}` }),
          el("span", { text: `tagger: ${plan.data.tagger}` }),
          el("span", { text: `run ${plan.data.run_status}` }),
        ])
      : null,
  ]);
}

function planCards(plan) {
  const wanted = [
    ["tests in BH family", "Tests in one corrected family"],
    ["flagged interviewers", "Interviewers flagged"],
    ["flagged panels", "Panels flagged"],
    ["flagged fairness cells", "Fairness cells flagged"],
    ["non-predictive roles", "Roles that do not predict"],
    ["process flags", "Process flags"],
  ];
  return el(
    "div",
    { class: "cards" },
    wanted.map(([label, text], i) => {
      const number = findNumber(plan, label);
      return card(
        [el("span", { class: "kpi-label", text }), countUp(el("span", { class: "kpi-value" }, [traced(number)]), number)],
        { delay: i * 50 },
      );
    }),
  );
}

function flagList(flags) {
  const rows = flags.data?.flagged || [];
  if (!rows.length) return note("No interviewer drifts from their peers beyond the registered threshold.");
  return el(
    "div",
    { class: "cards" },
    rows.map((row, i) =>
      card(
        [
          el("span", { class: "kpi-label", text: `${row.interviewer_id}, ${titleCase(row.role_id)}` }),
          el("span", { class: "kpi-value" }, [
            traced(findNumber(flags, `flagged cells ${row.interviewer_id} ${row.role_id}`)),
          ]),
          el("span", { class: "badge badge--flag", text: `${row.cells} competencies measured` }),
          el("span", { class: "kpi-note", text: "Rated above peers on the same role and competency." }),
        ],
        { delay: i * 60 },
      ),
    ),
  );
}

function panelChart(panels) {
  const rows = (panels.data?.panels || []).filter((p) => !p.suppressed);
  if (!rows.length) return note("No panel has enough loops to measure.");
  const theme = chartTheme();
  const { plot, node } = chartBox("Fleiss kappa per panel, against the 0.40 flag line");
  const points = rows
    .map((row) => ({ row, number: findNumber(panels, `Fleiss kappa ${row.panel_id}`) }))
    .filter((p) => p.number)
    .sort((a, b) => a.number.value - b.number.value);

  onAttached(() =>
    drawChart(
      plot,
      [
        {
          type: "bar",
          orientation: "h",
          x: points.map((p) => p.number.value),
          y: points.map((p) => p.row.panel_id),
          marker: { color: points.map((p) => (p.row.flagged ? theme.warn : theme.accent)) },
          hovertemplate: "%{y}: kappa %{x:.2f}<extra></extra>",
        },
      ],
      {
        height: Math.max(320, points.length * 22),
        margin: { l: 150, r: 20, t: 20, b: 48 },
        xaxis: { title: { text: "Fleiss kappa" }, range: [-0.1, 1] },
        yaxis: { title: { text: "" }, automargin: true },
        shapes: [
          {
            type: "line", x0: 0.4, x1: 0.4, y0: -0.5, y1: points.length - 0.5,
            line: { color: theme.warn, width: 1, dash: "dash" },
          },
        ],
      },
    ),
  );

  const flagged = points.filter((p) => p.row.flagged);
  return el("div", {}, [
    node,
    flagged.length
      ? el("p", { class: "scope" }, [
          `Flagged: ${flagged.map((p) => `${p.row.panel_id} (${p.row.members} members, ${p.row.n} loops)`).join(", ")}`,
        ])
      : note("Every panel agrees better than chance."),
  ]);
}

function validityChart(validity) {
  const rows = (validity.data?.roles || []).filter((r) => !r.suppressed);
  if (!rows.length) return note("No role has enough hires with later performance to measure.");
  const theme = chartTheme();
  const { plot, node } = chartBox("Spearman correlation of interview score with later performance, 95% interval");
  const points = rows
    .map((row) => ({ row, number: findNumber(validity, `Spearman rho ${row.role_id}`) }))
    .filter((p) => p.number);

  onAttached(() =>
    drawChart(
      plot,
      [
        {
          type: "scatter",
          mode: "markers",
          x: points.map((p) => p.number.value),
          y: points.map((p) => titleCase(p.row.role_id)),
          marker: { size: 11, color: points.map((p) => (p.row.flagged ? theme.warn : theme.accent)) },
          error_x: {
            type: "data",
            symmetric: false,
            array: points.map((p) => p.row.ci_high - p.number.value),
            arrayminus: points.map((p) => p.number.value - p.row.ci_low),
            color: theme.ink,
            thickness: 1,
            width: 5,
          },
          hovertemplate: "%{y}: rho %{x:.2f}<extra></extra>",
        },
      ],
      {
        height: Math.max(300, points.length * 42),
        margin: { l: 130, r: 20, t: 20, b: 48 },
        xaxis: { title: { text: "Spearman rho" }, range: [-0.6, 1], zeroline: true, zerolinewidth: 2 },
        yaxis: { title: { text: "" } },
      },
    ),
  );

  const failing = points.filter((p) => p.row.flagged).map((p) => titleCase(p.row.role_id));
  return el("div", {}, [
    node,
    failing.length
      ? note(`Interval spans zero, so the interview does not predict performance: ${failing.join(", ")}.`, "error")
      : note("Every role's interview score predicts later performance."),
  ]);
}

function impactTable(impact) {
  const groups = impact.data?.groups || [];
  if (!groups.length) return note("Nothing to show at this stage.");
  const rows = groups.map((group) => {
    if (group.suppressed) {
      return el("tr", {}, [
        el("td", { text: group.group }),
        el("td", { colspan: "4", text: "Suppressed: fewer than five people in this cell." }),
      ]);
    }
    const rate = findNumber(impact, `selection rate ${group.group} ${ROLE} ${STAGE}`);
    const ratio = findNumber(impact, `impact ratio ${group.group} ${ROLE} ${STAGE}`);
    return el("tr", {}, [
      el("td", { text: group.group }),
      el("td", { class: "num", text: fmt(group.n, "count") }),
      el("td", { class: "num" }, [traced(rate)]),
      el("td", { class: "num" }, [traced(ratio)]),
      el("td", {}, [
        group.flagged
          ? el("span", { class: "badge badge--flag", text: `below four fifths, q ${group.q_band}` })
          : el("span", { class: "badge badge--ok", text: "reference or above" }),
      ]),
    ]);
  });
  return card([
    el("div", { class: "table-wrap" }, [
      el("table", {}, [
        el("thead", {}, [
          el("tr", {}, [
            el("th", { text: "Group" }),
            el("th", { class: "num", text: "People" }),
            el("th", { class: "num", text: "Selection rate" }),
            el("th", { class: "num", text: "Impact ratio" }),
            el("th", { text: "Result" }),
          ]),
        ]),
        el("tbody", {}, rows),
      ]),
    ]),
    el("p", {
      class: "note note--advice",
      text: "Impact ratio is each group's selection rate over the highest group's. Below 0.80 is the four-fifths flag.",
    }),
  ]);
}

function evidenceCards(evidence, tagger) {
  const rows = (evidence.data?.rows || []).filter((r) => !r.suppressed);
  const below = rows.filter((r) => r.flagged);
  const kappa = findNumber(tagger, "Cohen's kappa");
  return el("div", { class: "cards" }, [
    card([
      el("span", { class: "kpi-label", text: "Interviewers measured" }),
      el("span", { class: "kpi-value", text: fmt(rows.length, "count") }),
      el("span", {
        class: `badge badge--${below.length ? "flag" : "ok"}`,
        text: below.length ? `${below.length} below the threshold` : "all above the threshold",
      }),
    ]),
    card(
      [
        el("span", { class: "kpi-label", text: "Tagger agreement with labels" }),
        el("span", { class: "kpi-value" }, [traced(kappa)]),
        el("span", { class: "kpi-note", text: (tagger.data?.caveat || "").slice(0, 180) }),
      ],
      { delay: 60 },
    ),
  ]);
}
