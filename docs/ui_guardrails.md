# UI guardrails (anti-vibecoded)

Applies to every Streamlit page, chart and export from Phase 5 on. Check a page against this list
before calling it done.

## Avoid the generic tells
Being careful does not mean being flat. Keep depth, motion and real interaction, just not the
template versions of them.

- No harsh or rainbow gradients, neon, purple-and-black, basic pastels, radial orbs, dot grids.
- No pure white page background; use a warm or neutral off-white. Dark mode is neutral black, not blue-tinted.
- No Inter, Geist or Space Grotesk. No emoji, lucide or sparkle icons as decoration.
- No drop shadow on everything, liquid glass, coloured left stripes on cards, or soft over-rounded corners.
- No "three feature cards in a row", bento grids, terminal-window mockups, checkmark-bullet lists or pricing tiers.
- Copy: no em dashes, no "it's not X, it's Y", no fake testimonials, no placeholder text.
- Hover and arrow animations only where they carry meaning (for example a row that opens a lineage drawer).

## Motion and effects (taste-skill `soft-skill`)
Use soft-skill for effects only: depth, motion and interaction physics. Fonts, colours and layouts
still come from this file, which wins wherever the two conflict.

- **Depth:** major cards use the double-bezel build, an outer shell with a hairline ring and small
  padding around an inner core with its own inset highlight and a concentric, smaller radius. Shadows
  are wide, soft and neutral (low-opacity black at a large blur), never harsh or coloured.
- **Easing:** every transition uses a custom spring-like curve, for example
  `cubic-bezier(0.32, 0.72, 0, 1)` at 500 to 800 ms. No `linear`, no `ease-in-out`, no instant state jumps.
- **Entrance:** sections, KPI tiles and chart cards fade up from a slight offset and blur, staggered
  in 60 to 100 ms steps. KPI numbers count up from zero once.
- **Buttons:** pill primary buttons press to `scale(0.98)` when active; a trailing arrow sits in its
  own small circle and nudges diagonally on hover.
- **Grain:** an optional film-grain overlay goes on a fixed, `pointer-events: none` layer at about 3% opacity.
- **Performance:** animate only `transform` and `opacity`. Use `backdrop-filter` only on fixed
  elements (top bar, dialogs), never on scrolling content. No scroll listeners; reveal with
  IntersectionObserver or CSS-only load animations. Respect `prefers-reduced-motion`.
- **Streamlit:** inject one shared stylesheet from a helper used by every page. Confirm the injection
  API and script sandboxing against the installed Streamlit version before relying on JavaScript.

Ignored from soft-skill because they conflict with the list above: Geist, purple or emerald glow
orbs, liquid glass, bento grids, 10px uppercase eyebrow tags (text floor is 12px, sentence case),
and OLED-black glass themes (dark mode stays neutral black without glow).

## Do the basics properly
- Show real product data: every page renders actual synthetic outputs with lineage, never mockups.
- Loading states: skeletons or spinners on every slow query, success and error messages on every action.
- No horizontal scroll, no mobile overflow; wide tables scroll inside their own container.
- Page titles and a favicon set via `st.set_page_config`. No dead nav items, no broken buttons or links.
- Footer: running LLM cost (section 7), "synthetic data" label, and a "not legal advice" note on compliance pages.
- Charts follow the dataviz skill: one accent hue, labelled axes, colour-blind-safe palettes.
