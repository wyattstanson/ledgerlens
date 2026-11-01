/* Landing page: line reveals, stacked governance cards, counting proof figures, and the guide.

   The guide is rule-based on purpose. It matches a question against a small table of intents, then
   fetches the real figure from the same governed tool the workspace uses and links to the page that
   shows it. No model is called, so it costs nothing and works on a deployment with no API key. */

import { resolve } from "./guide.js";

const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const hasGsap = typeof window.gsap !== "undefined" && typeof window.ScrollTrigger !== "undefined";
window.__llReady = true;
if (!hasGsap) document.documentElement.classList.remove("js");

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
  const intent = resolve(question);
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

/* ---------- Mobile menu ---------- */

const burger = document.querySelector("[data-burger]");
const menu = document.querySelector("[data-menu]");
const setMenu = (open) => {
  burger.setAttribute("aria-expanded", String(open));
  burger.setAttribute("aria-label", open ? "Close menu" : "Open menu");
  menu.classList.toggle("is-open", open);
  menu.setAttribute("aria-hidden", String(!open));
  document.body.style.overflow = open ? "hidden" : "";
};
burger.addEventListener("click", () => setMenu(burger.getAttribute("aria-expanded") !== "true"));
menu.querySelectorAll("a").forEach((link) => link.addEventListener("click", () => setMenu(false)));
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && menu.classList.contains("is-open")) setMenu(false);
});

document.querySelector("[data-year]").textContent = String(new Date().getFullYear());

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
      // fromTo, not from: the stylesheet pre-hides [data-fade] so nothing flashes before GSAP loads,
      // and `from` would read that 0 as the tween's destination and animate the hero's eyebrow and
      // lede from invisible to invisible.
      .fromTo(
        hero.querySelectorAll("[data-fade]"),
        { y: 24, opacity: 0, filter: "blur(7px)" },
        { y: 0, opacity: 1, filter: "blur(0px)", duration: 1, stagger: 0.1 },
        0.5,
      )
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
