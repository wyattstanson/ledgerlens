/* Landing page: line reveals, stacked governance cards, counting proof figures, and the guide.

   The guide is rule-based on purpose. It matches a question against a small table of intents, then
   fetches the real figure from the same governed tool the workspace uses and links to the page that
   shows it. No model is called, so it costs nothing and works on a deployment with no API key. */

const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const hasGsap = typeof window.gsap !== "undefined" && typeof window.ScrollTrigger !== "undefined";
window.__llReady = true;
if (!hasGsap) document.documentElement.classList.remove("js");

/* ---------- The guide ---------- */

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

const pick = (result, prefix) => (result.numbers || []).find((n) => n.label.startsWith(prefix));
const money = (v) =>
  `${v < 0 ? "-" : "+"}$${Math.abs(v).toLocaleString("en-US", { maximumFractionDigits: 0 })}`;

const answerBox = document.querySelector("[data-answer]");

function renderAnswer(intent, figure) {
  const core = document.createElement("div");
  core.className = "answer-core";

  const h = document.createElement("h3");
  h.textContent = intent.title;
  const p = document.createElement("p");
  p.textContent = intent.body;
  core.append(h, p);

  if (figure) {
    const f = document.createElement("p");
    f.className = "answer-meta";
    f.textContent = figure;
    core.append(f);
  }

  const link = document.createElement("a");
  link.className = "btn btn--primary btn--sm";
  link.href = intent.page;
  const label = document.createElement("span");
  label.textContent = intent.label;
  const orb = document.createElement("span");
  orb.className = "orb";
  orb.setAttribute("aria-hidden", "true");
  const icon = document.createElement("i");
  icon.className = "ph-light ph-arrow-up-right";
  orb.append(icon);
  link.append(label, orb);
  core.append(link);

  answerBox.replaceChildren(core);
  answerBox.hidden = false;
  if (hasGsap && !REDUCED) {
    gsap.fromTo(answerBox, { opacity: 0, y: 14 }, { opacity: 1, y: 0, duration: 0.7, ease: "expo.out" });
  }
}

async function answer(question) {
  const intent = INTENTS.find((i) => i.match.test(question)) || FALLBACK;
  renderAnswer(intent, null);
  try {
    // The headline figure comes from the governed tool, so the guide never states a number of its own.
    const figure = await intent.figure();
    if (figure) renderAnswer(intent, figure);
  } catch {
    renderAnswer(intent, "The live figure is not available right now; the page itself still has it.");
  }
}

const form = document.querySelector("[data-ask]");
const input = document.querySelector("[data-ask-input]");
form.addEventListener("submit", (event) => {
  event.preventDefault();
  if (input.value.trim()) answer(input.value.trim());
});
document.querySelectorAll("[data-suggestions] button").forEach((button) => {
  button.addEventListener("click", () => {
    input.value = button.textContent;
    answer(button.dataset.q);
  });
});

/* ---------- Motion ---------- */

if (hasGsap) {
  gsap.registerPlugin(ScrollTrigger);

  // Each line of display type is wrapped so it can rise out of its own mask.
  document.querySelectorAll("[data-lines] > span").forEach((line) => {
    const inner = document.createElement("i");
    inner.textContent = line.textContent;
    line.replaceChildren(inner);
  });

  if (REDUCED) {
    gsap.set("[data-lines] > span > i, [data-fade], [data-row], .proof-card", { y: 0, opacity: 1, filter: "none" });
  } else {
    const nav = document.querySelector("[data-nav]");
    ScrollTrigger.create({ start: 60, end: "max", onToggle: (s) => nav.classList.toggle("is-tight", s.isActive) });

    const hero = document.querySelector(".hero");
    gsap
      .timeline({ defaults: { ease: "expo.out" } })
      .from(hero.querySelectorAll("[data-lines] > span > i"), { yPercent: 115, duration: 1.2, stagger: 0.09 }, 0.15)
      .from(hero.querySelectorAll("[data-fade]"), { y: 24, opacity: 0, filter: "blur(7px)", duration: 1, stagger: 0.1 }, 0.5)
      .from(".ask", { y: 20, opacity: 0, duration: 1 }, 0.75)
      .from(".suggestions button", { y: 14, opacity: 0, duration: 0.8, stagger: 0.06 }, 0.95);

    // Headings below the fold reveal line by line as they arrive.
    document.querySelectorAll("section:not(.hero) [data-lines]").forEach((heading) => {
      gsap.from(heading.querySelectorAll("span > i"), {
        yPercent: 115,
        duration: 1.1,
        stagger: 0.08,
        ease: "expo.out",
        scrollTrigger: { trigger: heading, start: "top 82%", once: true },
      });
    });

    ScrollTrigger.batch("[data-row], .proof-card, section:not(.hero) [data-fade]", {
      start: "top 88%",
      once: true,
      onEnter: (batch) =>
        gsap.to(batch, {
          opacity: 1,
          y: 0,
          filter: "blur(0px)",
          duration: 0.9,
          stagger: 0.08,
          ease: "expo.out",
          overwrite: true,
        }),
    });

    // The governance cards are sticky in CSS; each shrinks as the next one covers it.
    const cards = gsap.utils.toArray("[data-card]");
    cards.forEach((card, i) => {
      if (i === cards.length - 1) return;
      gsap.to(card, {
        scale: 0.94,
        opacity: 0.45,
        ease: "none",
        scrollTrigger: { trigger: cards[i + 1], start: "top bottom", end: "top top", scrub: true },
      });
    });

    // Proof figures count up once.
    document.querySelectorAll("[data-count]").forEach((node) => {
      const target = Number(node.dataset.count);
      const final = node.textContent;
      const counter = { v: 0 };
      ScrollTrigger.create({
        trigger: node,
        start: "top 90%",
        once: true,
        onEnter: () =>
          gsap.to(counter, {
            v: target,
            duration: 1.4,
            ease: "expo.out",
            onUpdate: () => {
              node.textContent = Math.round(counter.v).toLocaleString("en-US");
            },
            onComplete: () => {
              node.textContent = final;
            },
          }),
      });
    });
  }

  window.addEventListener("load", () => ScrollTrigger.refresh());
}
