/* The approval queue. Agents propose; a named person decides. A shared deployment shows it read-only. */

import { card, el, getJson, note, postJson, section } from "../lib.js";

export const title = "Approvals";

export async function render(view) {
  view.replaceChildren(head());
  const queue = await getJson("/api/approvals");
  view.append(
    queue.can_decide
      ? note("You are running LedgerLens yourself, so you can decide these.")
      : note("This deployment shows the queue read-only. Decide these where you run LedgerLens yourself.", "error"),
    section(`Pending (${queue.pending.length})`, [list(queue, view)]),
  );
}

function head() {
  return el("div", { class: "page-head" }, [
    el("h1", { text: "Approvals" }),
    el("p", {
      text: "Nothing an agent proposes changes state until someone named approves it. Each decision is logged with who made it and why, and an approved record is used exactly once.",
    }),
  ]);
}

function list(queue, view) {
  if (!queue.pending.length) return note("Nothing is waiting for a decision.");
  return el(
    "div",
    { class: "cards" },
    queue.pending.map((approval, i) => row(approval, queue.can_decide, view, i)),
  );
}

function row(approval, canDecide, view, index) {
  const who = el("input", { type: "text", id: `who-${approval.approval_id}`, placeholder: "Your name", maxlength: "120" });
  const why = el("input", { type: "text", id: `why-${approval.approval_id}`, placeholder: "Reason", maxlength: "2000" });
  const message = el("p", { class: "kpi-note" });

  const decide = async (approve) => {
    if (!who.value.trim() || !why.value.trim()) {
      message.replaceChildren(note("A decision needs your name and a reason.", "error"));
      return;
    }
    message.replaceChildren(el("span", { class: "kpi-note", text: "Recording the decision" }));
    try {
      await postJson(`/api/approvals/${encodeURIComponent(approval.approval_id)}/decide`, {
        approve,
        decided_by: who.value.trim(),
        reason: why.value.trim(),
      });
      render(view);
    } catch (error) {
      message.replaceChildren(note(error.message, "error"));
    }
  };

  return card(
    [
      el("span", { class: "kpi-label", text: approval.action }),
      el("p", { class: "rowids", text: JSON.stringify(approval.payload) }),
      el("p", { class: "kpi-note", text: `Proposed by run ${approval.agent_run_id}` }),
      canDecide
        ? el("div", { class: "controls" }, [
            el("div", { class: "field" }, [el("label", { for: who.id, text: "Decided by" }), who]),
            el("div", { class: "field" }, [el("label", { for: why.id, text: "Reason" }), why]),
            el("button", { type: "button", class: "btn", onclick: () => decide(true) }, [
              "Approve",
              el("span", { class: "btn-orb", "aria-hidden": "true", text: "↗" }),
            ]),
            el("button", { type: "button", class: "btn btn--quiet", text: "Reject", onclick: () => decide(false) }),
          ])
        : null,
      message,
    ],
    { delay: index * 60 },
  );
}
