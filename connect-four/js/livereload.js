// Polls the dev server for source-file changes (see server.py's /__version)
// and reloads the page when one is detected, so edits made outside the
// browser show up without a manual refresh. Purely a development aid --
// delete the <script> tag in index.html to turn it off. Harmless if the dev
// server isn't running (fetch just keeps failing silently).
(() => {
  const POLL_MS = 1000;
  let baseline = null;

  async function checkVersion() {
    let res;
    try {
      res = await fetch("/__version", { cache: "no-store" });
    } catch {
      return; // dev server not reachable -- try again next tick
    }
    if (!res.ok) return;
    const { version } = await res.json();
    if (baseline === null) {
      baseline = version;
      return;
    }
    if (version !== baseline) {
      location.reload();
    }
  }

  setInterval(checkVersion, POLL_MS);
  checkVersion();
})();
