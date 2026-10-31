/* The workspace guide: a small unit docked in the corner that knows which page you are on.

   It answers from the same table the landing page uses (guide.js), so the two can never say
   different things, and every figure it quotes comes back from a governed tool rather than from
   the text. Nothing here calls a model: the deployment has no key, and a guide that invents a
   number would undo the point of the product. */

import { resolve } from "./guide.js";
import { el } from "./lib.js";

/* What the unit says when you open it on a page, and the three questions worth asking there. */
const CONTEXT = {
  fpa: {
    line: "This page rebuilds cash from the payment events underneath it. Ask what moved, or how a figure was derived.",
    asks: ["Why is net cash under plan?", "How good is the forecast?", "Where does this number come from?"],
  },
  interviews: {
    line: "Interview quality, measured on the process rather than on any candidate.",
    asks: ["How is interviewer drift found?", "Why are small groups hidden?", "Does the score predict performance?"],
  },
  ratecards: {
    line: "Vendor rates normalised to one basis, and the cheapest panel that still covers every role.",
    asks: ["Where is the overbilling?", "How is the panel chosen?", "What would switching save?"],
  },
  yours: {
    line: "Your own figures, through the same tested functions. Nothing is stored and nothing reads the warehouse.",
    asks: ["Why is there no lineage id?", "How is the variance bridge split?", "What makes a variance material?"],
  },
  method: {
    line: "Every formula the platform uses, with its inputs live. Move one and the result moves.",
    asks: ["How is the variance bridge split?", "What is MASE?", "How are rates normalised?"],
  },
  approvals: {
    line: "Agents propose; a named person decides. This deployment is read-only.",
    asks: ["Who can approve?", "What needs an approval?", "Why can nothing be decided here?"],
  },
  audit: {
    line: "Every run, every tool call and every number, still queryable after the fact.",
    asks: ["How do I trace a number?", "What did this cost?", "What is a lineage id?"],
  },
};

const FALLBACK_CONTEXT = {
  line: "Ask about cash, interview quality, vendor rates, or how any figure is proved.",
  asks: ["Why is net cash under plan?", "How is interviewer drift found?", "Where does this number come from?"],
};

let panel = null;
let answerBox = null;
let asksRow = null;
let openerLine = null;
let toggle = null;

/** The unit itself: a visor and two eyes, drawn in CSS so it can react without loading anything. */
function unit() {
  return el("span", { class: "guide-unit", "aria-hidden": "true" }, [
    el("span", { class: "guide-visor" }, [el("i", { class: "guide-eye" }), el("i", { class: "guide-eye" })]),
  ]);
}

function setOpen(open) {
  panel.hidden = !open;
  toggle.setAttribute("aria-expanded", String(open));
  document.body.classList.toggle("is-guided", open);
  if (open) panel.querySelector("input").focus();
}

function render(intent, figure) {
  const parts = [
    el("h3", { text: intent.title }),
    el("p", { text: intent.body }),
    figure ? el("p", { class: "guide-figure", text: figure }) : null,
    el("a", { class: "btn btn--quiet", href: intent.page.replace("/app", ""), text: intent.label }),
  ];
  answerBox.replaceChildren(...parts.filter(Boolean));
  answerBox.hidden = false;
}

async function answer(question) {
  const intent = resolve(question);
  render(intent, null);
  try {
    // The figure comes from the tool, so the guide quotes the warehouse rather than itself.
    const figure = await intent.figure();
    if (figure) render(intent, figure);
  } catch {
    render(intent, "That live figure is not available right now; the page itself still carries it.");
  }
}

/** Point the guide at the page now showing, so its opener and its suggestions match the work. */
export function focusOn(name) {
  if (!panel) return;
  const context = CONTEXT[name] || FALLBACK_CONTEXT;
  openerLine.textContent = context.line;
  answerBox.hidden = true;
  asksRow.replaceChildren(
    ...context.asks.map((ask) =>
      el("button", { type: "button", class: "guide-ask", text: ask, onclick: () => answer(ask) }),
    ),
  );
}

export function mount() {
  toggle = el("button", {
    type: "button",
    class: "guide-toggle",
    "aria-expanded": "false",
    "aria-controls": "guide-panel",
    onclick: () => setOpen(panel.hidden),
  }, [unit(), el("span", { class: "guide-toggle-text", text: "Guide" })]);

  openerLine = el("p", { class: "guide-line" });
  asksRow = el("div", { class: "guide-asks" });
  answerBox = el("div", { class: "guide-answer", hidden: "" });

  const input = el("input", { type: "text", name: "q", placeholder: "Ask about this page", autocomplete: "off" });
  const form = el("form", { class: "guide-form" }, [input, el("button", { type: "submit", class: "btn", text: "Ask" })]);
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const question = input.value.trim();
    if (question) answer(question);
  });

  panel = el("div", {
    class: "guide-panel", id: "guide-panel", hidden: "", role: "dialog", "aria-label": "Workspace guide",
  }, [
    el("div", { class: "guide-head" }, [
      el("h2", { text: "Guide" }),
      el("button", { type: "button", class: "drawer-close", text: "Close", onclick: () => setOpen(false) }),
    ]),
    openerLine,
    asksRow,
    answerBox,
    form,
    el("p", { class: "guide-foot", text: "Rule based, no model call. Figures come from the same tools the pages use." }),
  ]);

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !panel.hidden) setOpen(false);
  });

  document.body.append(el("div", { class: "guide" }, [panel, toggle]));
}
