/* Shell: hash routing between the five pages, the footer's running cost, and the lineage drawer. */

import { closeLineage, el, flushAttached, getJson, note, resetAttached, revealAll, skeleton } from "./lib.js";
import * as approvals from "./pages/approvals.js";
import * as audit from "./pages/audit.js";
import * as fpa from "./pages/fpa.js";
import * as interviews from "./pages/interviews.js";
import * as ratecards from "./pages/ratecards.js";

const PAGES = { fpa, interviews, ratecards, approvals, audit };
const DEFAULT_PAGE = "fpa";

const view = document.getElementById("view");
let meta = null;
// A page fetches several panels before it appends them, so a second navigation can land mid-render.
// Each render takes a ticket; a render that is no longer the current one throws its output away.
let ticket = 0;

function currentPage() {
  const name = window.location.hash.replace(/^#\/?/, "");
  return name in PAGES ? name : DEFAULT_PAGE;
}

function markTab(name) {
  for (const tab of document.querySelectorAll(".tabs a")) {
    if (tab.dataset.tab === name) tab.setAttribute("aria-current", "page");
    else tab.removeAttribute("aria-current");
  }
  document.title = `${PAGES[name].title} · LedgerLens`;
}

async function show(name) {
  const mine = ++ticket;
  resetAttached();
  markTab(name);
  view.replaceChildren(el("div", { class: "skeleton-page" }, [skeleton("2rem"), skeleton("14rem")]));
  // Pages build into a detached node, so a slow render never half-paints the page it is replacing.
  const staging = document.createElement("div");
  try {
    if (!meta) meta = await getJson("/api/meta");
    await PAGES[name].render(staging, meta);
    if (mine !== ticket) return;
    view.replaceChildren(...staging.childNodes);
    flushAttached();
    revealAll(view);
  } catch (error) {
    if (mine !== ticket) return;
    view.replaceChildren(
      el("div", { class: "page-head" }, [el("h1", { text: PAGES[name].title })]),
      note(error.message || "Something went wrong loading this page.", "error"),
    );
  }
}

async function footer() {
  const scope = document.getElementById("footer-scope");
  const cost = document.getElementById("footer-cost");
  try {
    const [spend, info] = await Promise.all([getJson("/api/audit"), meta ?? getJson("/api/meta")]);
    const total = Number(spend.cost?.total_usd || 0);
    cost.textContent = `Model spend so far: $${total.toFixed(4)} over ${spend.cost?.calls || 0} calls`;
    scope.textContent = info.fpa ? `FP&A world: ${info.fpa.variant}, as of ${info.fpa.as_of}` : "";
  } catch {
    cost.textContent = "Model spend: unavailable";
  }
}

document.getElementById("drawer-close").addEventListener("click", closeLineage);
document.getElementById("scrim").addEventListener("click", closeLineage);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closeLineage();
});
window.addEventListener("hashchange", () => show(currentPage()));

await show(currentPage());
await footer();
