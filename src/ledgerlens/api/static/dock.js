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

const FULL = "ledgerlens.window.full";

function rememberedFull() {
  try {
    return window.localStorage.getItem(FULL) === "1";
  } catch {
    return false;
  }
}

function rememberFull(full) {
  try {
    window.localStorage.setItem(FULL, full ? "1" : "0");
  } catch {
    // The window still fills or floats; it just forgets which between visits.
  }
}

export function mountDock() {
  const toggle = document.getElementById("rail-dock");
  const minimise = document.getElementById("light-min");
  const zoom = document.getElementById("light-zoom");
  const close = document.getElementById("light-close");
  if (!toggle) return;

  /* Both the sidebar button and the amber light drive one state, so they can never disagree. */
  const applyDock = (docked) => {
    document.body.classList.toggle("is-docked", docked);
    toggle.setAttribute("aria-expanded", String(!docked));
    const label = docked ? "Expand the sidebar" : "Collapse the sidebar";
    toggle.title = label;
    toggle.querySelector(".sr-only").textContent = label;
    if (minimise) minimise.title = label;
  };

  const applyFull = (full) => {
    document.body.classList.toggle("is-full", full);
    if (zoom) zoom.title = full ? "Float the window" : "Fill the screen";
  };

  const flipDock = () => {
    const docked = !document.body.classList.contains("is-docked");
    applyDock(docked);
    remember(docked);
  };

  applyDock(remembered());
  applyFull(rememberedFull());

  toggle.addEventListener("click", flipDock);
  minimise?.addEventListener("click", flipDock);
  zoom?.addEventListener("click", () => {
    const full = !document.body.classList.contains("is-full");
    applyFull(full);
    rememberFull(full);
  });
  // Closing a window in a browser means leaving it, so the red light goes back to the public page.
  close?.addEventListener("click", () => {
    window.location.href = "/";
  });
}
