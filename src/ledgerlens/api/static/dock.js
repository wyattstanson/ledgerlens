/* The rail, docked: collapse it to its icons and leave it that way.

   A per-viewer convenience, so the choice lives in localStorage rather than anywhere the server
   knows about. Storage can be absent, full, or throw outright in a private window, so every read
   and write is guarded and the rail simply opens when it cannot remember. */

const KEY = "ledgerlens.rail.docked";

function remembered() {
  try {
    return window.localStorage.getItem(KEY) === "1";
  } catch {
    return false; // blocked or unavailable: the rail opens, which is the safe default
  }
}

function remember(docked) {
  try {
    window.localStorage.setItem(KEY, docked ? "1" : "0");
  } catch {
    // Nothing to do: the rail still works, it just forgets between visits.
  }
}

export function mountDock() {
  const toggle = document.getElementById("rail-dock");
  if (!toggle) return;

  const apply = (docked) => {
    document.body.classList.toggle("is-docked", docked);
    toggle.setAttribute("aria-expanded", String(!docked));
    const label = docked ? "Expand the sidebar" : "Collapse the sidebar";
    toggle.title = label;
    toggle.querySelector(".sr-only").textContent = label;
  };

  apply(remembered());
  toggle.addEventListener("click", () => {
    const docked = !document.body.classList.contains("is-docked");
    apply(docked);
    remember(docked);
  });
}
