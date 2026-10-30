/* Shared helpers: the API client, DOM building, number formatting and the lineage drawer.
   Every number a page shows comes from a tool's numbers[] array, so it always has a lineage id. */

const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/* ---------- API ---------- */

export async function callTool(name, params = {}) {
  const response = await fetch(`/api/tools/${name}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(params),
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new ApiError(body.detail || `${name} failed`, response.status);
  return body;
}

export async function getJson(path) {
  const response = await fetch(path);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new ApiError(body.detail || `${path} failed`, response.status);
  return body;
}

export async function postJson(path, payload) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new ApiError(body.detail || `${path} failed`, response.status);
  return body;
}

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

/* ---------- DOM ---------- */

export function el(tag, props = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value === null || value === undefined) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value; // text only: nothing here sets innerHTML
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (key === "style") node.setAttribute("style", value);
    else node.setAttribute(key, value);
  }
  for (const child of [].concat(children)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child.nodeType ? child : document.createTextNode(String(child)));
  }
  return node;
}

/** A double-bezel card: outer shell, inner core. */
export function card(children, { reveal = true, delay = 0 } = {}) {
  const core = el("div", { class: "core" }, children);
  const shell = el("div", { class: reveal ? "shell reveal" : "shell" }, [core]);
  if (reveal && delay) shell.style.setProperty("--d", `${delay}ms`);
  return shell;
}

export function section(title, children) {
  return el("section", { class: "section" }, [title ? el("h2", { text: title }) : null, ...[].concat(children)]);
}

export function note(message, kind = "info") {
  return el("p", { class: `note note--${kind}`, text: message });
}

export function skeleton(height = "14rem") {
  return el("span", { class: "skeleton", style: `height:${height}`, "aria-hidden": "true" });
}

/* ---------- Formatting ---------- */

const MONEY = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });
const MONEY_CENTS = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" });
const PLAIN = new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 });

export function fmt(value, unit) {
  if (value === null || value === undefined) return "not available";
  if (unit === "USD") return Math.abs(value) >= 1000 ? MONEY.format(value) : MONEY_CENTS.format(value);
  if (unit === "USD/hour") return `${MONEY_CENTS.format(value)}/h`;
  if (unit === "ratio" || unit === "rate" || unit === "share") return `${(value * 100).toFixed(1)}%`;
  if (unit === "count") return PLAIN.format(value);
  return PLAIN.format(value);
}

export function compactMoney(value) {
  const millions = value / 1_000_000;
  return `${millions >= 0 ? "" : "-"}$${Math.abs(millions).toFixed(2)}M`;
}

export function titleCase(text) {
  return String(text).replaceAll("_", " ").replace(/^./, (c) => c.toUpperCase());
}

/* ---------- Numbers with lineage ---------- */

/** Index a tool result's numbers[] by label, so a page can ask for the one it wants. */
export function numbersByLabel(result) {
  const index = new Map();
  for (const number of result.numbers || []) index.set(number.label, number);
  return index;
}

/** The first number whose label starts with `prefix`. */
export function findNumber(result, prefix) {
  return (result.numbers || []).find((n) => n.label.startsWith(prefix)) || null;
}

/** Every number whose label starts with `prefix`, with that prefix stripped off the label. */
export function matchNumbers(result, prefix) {
  return (result.numbers || [])
    .filter((n) => n.label.startsWith(prefix))
    .map((n) => ({ ...n, rest: n.label.slice(prefix.length).trim() }));
}

/** A number rendered as a button that opens its lineage. */
export function traced(number, text) {
  if (!number) return el("span", { text: "not available" });
  return el("button", {
    type: "button",
    class: "traced",
    text: text ?? fmt(number.value, number.unit),
    title: "Show where this number came from",
    "data-lineage": number.lineage_id,
    onclick: () => openLineage(number.lineage_id),
  });
}

/* ---------- Lineage drawer ---------- */

const drawer = () => document.getElementById("drawer");
const scrim = () => document.getElementById("scrim");

export async function openLineage(lineageId) {
  const body = document.getElementById("drawer-body");
  body.replaceChildren(skeleton("8rem"));
  drawer().hidden = false;
  scrim().hidden = false;
  document.getElementById("drawer-close").focus();
  try {
    const row = await getJson(`/api/lineage/${encodeURIComponent(lineageId)}`);
    body.replaceChildren(
      el("dl", {}, [
        el("dt", { text: "Value" }),
        el("dd", { text: `${fmt(row.value, row.unit)} (${row.unit})` }),
        el("dt", { text: "Produced by" }),
        el("dd", { text: `${row.tool_name}, run ${row.agent_run_id}` }),
        el("dt", { text: "Source tables" }),
        el("dd", { text: (row.source_tables || []).join(", ") || "none recorded" }),
        el("dt", { text: "Query" }),
        el("dd", {}, [el("pre", { text: (row.sql_text || "").trim() })]),
        el("dt", { text: `Rows (${(row.row_ids || []).length})` }),
        el("dd", {}, [el("p", { class: "rowids", text: (row.row_ids || []).join(", ") || "none recorded" })]),
      ]),
    );
  } catch (error) {
    body.replaceChildren(note(error.message, "error"));
  }
}

export function closeLineage() {
  drawer().hidden = true;
  scrim().hidden = true;
}

/* ---------- Entrances ---------- */

export function revealAll(root) {
  const targets = root.querySelectorAll(".reveal");
  if (REDUCED) {
    targets.forEach((t) => t.classList.add("is-in"));
    return;
  }
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry, i) => {
        if (!entry.isIntersecting) return;
        entry.target.style.setProperty("--d", `${i * 70}ms`);
        entry.target.classList.add("is-in");
        observer.unobserve(entry.target);
      });
    },
    { rootMargin: "0px 0px -8% 0px" },
  );
  targets.forEach((t) => observer.observe(t));
}

/* ---------- Charts ----------
   A page is built detached and attached in one go, so anything that needs real layout (Plotly sizes
   itself from the element) registers here and the shell runs it once the page is in the document. */

const pending = [];

export function onAttached(job) {
  pending.push(job);
}

export function flushAttached() {
  const jobs = pending.splice(0, pending.length);
  for (const job of jobs) job();
}

/** Forget anything a render that is being replaced had queued. */
export function resetAttached() {
  pending.length = 0;
}

const CSS = () => getComputedStyle(document.documentElement);

export function chartTheme() {
  const styles = CSS();
  const ink = styles.getPropertyValue("--ink").trim();
  const line = styles.getPropertyValue("--line").trim();
  return {
    accent: styles.getPropertyValue("--accent").trim(),
    warn: styles.getPropertyValue("--warn").trim(),
    ink,
    layout: {
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { family: "IBM Plex Sans, system-ui, sans-serif", size: 12, color: ink },
      margin: { l: 64, r: 16, t: 28, b: 52 },
      xaxis: { gridcolor: line, zerolinecolor: line, automargin: true },
      yaxis: { gridcolor: line, zerolinecolor: line, automargin: true },
      showlegend: false,
      hovermode: "closest",
    },
  };
}

export function drawChart(node, traces, layout) {
  const theme = chartTheme();
  window.Plotly.newPlot(node, traces, { ...theme.layout, ...layout }, {
    displayModeBar: false,
    responsive: true,
    staticPlot: REDUCED,
  });
}

export function chartBox(titleText, { height = "22rem" } = {}) {
  const plot = el("div", { class: "chart", style: `min-height:${height}` });
  return { plot, node: card([el("h3", { class: "kpi-label", text: titleText }), plot]) };
}
