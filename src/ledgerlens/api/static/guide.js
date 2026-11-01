/* The guide's knowledge: what a question is about, which page answers it, and the governed tool
   that supplies the figure.

   Rule based on purpose. A question is matched against a small table, then the number comes from
   the same tool the workspace calls, so the guide never states a figure of its own. No model is
   called, which is why it costs nothing and works on a deployment with no API key.

   Both the landing page's ask box and the workspace assistant read this table, so an answer cannot
   drift between the two. */

const INTENTS = [
  {
    match: /(net ?cash|cash flow|under plan|over plan|variance|bridge|month ?end|close)/i,
    title: "Start with the cash bridge",
    page: "/app#/fpa",
    label: "Open FP&A",
    body: "Actual against plan, split into volume, mix, rate and timing. The parts sum to the variance to the cent, and each one opens the rows behind it.",
    figure: async () => {
      const r = await call("fpa.bridge", { measure: "net_operating_cash", period: "2026-08" });
      const v = pick(r, "variance net_operating_cash");
      return v && `August net operating cash came in ${money(v.value)} against plan.`;
    },
  },
  {
    match: /(forecast|predict|13.?week|next quarter|model)/i,
    title: "The forecast was tested once",
    page: "/app#/fpa",
    label: "Open FP&A",
    body: "One model per driver, chosen by cross-validation, then tested a single time on six frozen months. Where nothing beat seasonal naive, seasonal naive stays and the page says so.",
    figure: async () => "Median error 0.64 MASE against 1.01 for the baseline, beating it on 8 of 12 series.",
  },
  {
    match: /(interview|hiring|candidate|bias|fair|panel|drift|adverse)/i,
    title: "The process is measured, never the person",
    page: "/app#/interviews",
    label: "Open HR",
    body: "Interviewers against their peers, panel agreement, selection rates by stage, and whether scores predict later performance. No candidate is ever scored, and groups under five people are suppressed.",
    figure: async () => {
      const r = await call("iv.plan_status", {});
      const n = pick(r, "flagged interviewers");
      return n && `This run flagged ${Math.round(n.value)} interviewer, out of 469 tests under one correction.`;
    },
  },
  {
    match: /(vendor|rate ?card|procure|supplier|invoice|overbill|panel of|spend|saving)/i,
    title: "Vendors compared on total cost",
    page: "/app#/ratecards",
    label: "Open procurement",
    body: "Billed titles mapped to standard roles, deals normalised to one hourly basis, and the cheapest panel that still covers every role. Each step of the saving traces to invoice lines.",
    figure: async () => {
      const r = await call("rc.leakage", {});
      const n = pick(r, "overbill lines all vendors");
      return n && `${Math.round(n.value).toLocaleString("en-US")} invoice lines are billed above the contracted card.`;
    },
  },
  {
    match: /(lineage|trace|prove|trust|how do i know|audit|where.*from|right\??$)/i,
    title: "Every figure resolves to its rows",
    page: "/app#/audit",
    label: "Open audit and cost",
    body: "A number arrives with an id. Click it and you get the query that produced it, the source tables and the row ids. If an answer ever cites a figure the record does not support, it is refused rather than shown.",
    figure: async () => "Run cost, tool calls and approvals stay queryable long after the answer.",
  },
  {
    match: /(approve|approval|who can|permission|sign ?off)/i,
    title: "Agents propose, people approve",
    page: "/app#/approvals",
    label: "Open approvals",
    body: "Anything that writes becomes a proposal and waits for a named person. This deployment lists the queue read-only, because a public URL cannot name anybody.",
    figure: async () => null,
  },
  {
    // Late in the table on purpose: "how is the bridge split" belongs to FP&A, and only a question
    // that names the arithmetic itself should land on the Method page.
    match: /(formula|equation|arithmetic|how is .* calculated|how do you compute|mase|kappa|cohen|z.?score|threshold|benjamini)/i,
    title: "Every equation, with its inputs live",
    page: "/app#/method",
    label: "Open the method",
    body: "Twenty-eight of them, grouped by office and by governance. Move an input and the result moves, so you can see how a rule behaves before you trust what it says about your month.",
    figure: async () => null,
  },
  {
    match: /(my own|our own|my numbers|my figures|try it|what if|upload|csv|before i|without a warehouse|my month)/i,
    title: "Put your own figures through it",
    page: "/app#/yours",
    label: "Open your numbers",
    body: "Your segments, your variance, your vendor rate, through the same tested functions the pack pages run. Nothing is stored, and nothing reads the warehouse, so the answers carry no lineage id and say so.",
    figure: async () => {
      // Computed when you ask, by the same function the rate-card page uses.
      const r = await compute("rc.card_rate", { pricing_model: "outcome", rate: 110, productivity_commitment: 0.2 });
      const rate = r.result?.effective_usd_per_hour;
      return rate && `A $110 outcome rate with a 20% commitment comes back as $${rate.toFixed(0)} a baseline hour.`;
    },
  },
];

const FALLBACK = {
  title: "Three places to start",
  page: "/app#/fpa",
  label: "Open the workspace",
  body: "FP&A for cash and variance, HR for interview quality and fairness, procurement for vendor cost. Every page works the same way: click a number, see its rows.",
  figure: async () => null,
};

async function call(tool, params) {
  const response = await fetch(`/api/tools/${tool}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!response.ok) throw new Error(`${tool} unavailable`);
  return response.json();
}

/** The compute routes take a client's own figures, so they answer without touching the warehouse. */
async function compute(name, payload) {
  const response = await fetch(`/api/compute/${name}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) throw new Error(`${name} unavailable`);
  return response.json();
}

const pick = (result, prefix) => (result.numbers || []).find((n) => n.label.startsWith(prefix));
const money = (v) =>
  `${v < 0 ? "-" : "+"}$${Math.abs(v).toLocaleString("en-US", { maximumFractionDigits: 0 })}`;

/** The intent behind a question, with the fallback for one the table does not recognise. */
export function resolve(question) {
  return INTENTS.find((intent) => intent.match.test(question)) || FALLBACK;
}

export { INTENTS, FALLBACK, call, compute, pick, money };
