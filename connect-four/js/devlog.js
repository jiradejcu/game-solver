// Mirrors this page's console to the dev server, which appends it to
// dev-console.log. Purely a development aid -- delete the <script> tag in
// index.html to turn it off.
//
// Loaded before everything else so it captures failures in later scripts,
// including a failed OpenCV.js load.
(() => {
  const ENDPOINT = "/__log";
  const FLUSH_MS = 300;
  const MAX_QUEUE = 300; // the vision loop runs per-frame; don't let it flood

  // Keep unpatched references so logging never recurses into itself.
  const original = {};
  const LEVELS = ["log", "info", "warn", "error", "debug"];
  for (const level of LEVELS) original[level] = console[level].bind(console);

  let queue = [];
  let dropped = 0;
  let timer = null;

  function describe(value, depth = 0) {
    if (value instanceof Error) return `${value.name}: ${value.message}\n${value.stack || ""}`.trim();
    if (typeof value === "string") return value;
    if (typeof value === "function") return `[function ${value.name || "anonymous"}]`;
    if (value === null || value === undefined || typeof value !== "object") return String(value);
    if (value instanceof HTMLElement) return `<${value.tagName.toLowerCase()}#${value.id || ""}>`;
    if (depth > 2) return "…";
    try {
      const seen = new WeakSet();
      return JSON.stringify(value, (_k, v) => {
        if (typeof v === "object" && v !== null) {
          if (seen.has(v)) return "[circular]";
          seen.add(v);
        }
        // Typed arrays and big pixel buffers are noise in a log file.
        if (ArrayBuffer.isView(v) && v.length > 16) return `[${v.constructor.name} len=${v.length}]`;
        return v;
      });
    } catch {
      return Object.prototype.toString.call(value);
    }
  }

  function push(level, args, where) {
    if (queue.length >= MAX_QUEUE) {
      dropped++;
      return;
    }
    queue.push({
      level,
      where,
      text: Array.from(args).map((a) => describe(a)).join(" "),
    });
    if (!timer) timer = setTimeout(flush, FLUSH_MS);
  }

  function flush() {
    timer = null;
    if (!queue.length) return;
    const batch = queue;
    queue = [];
    if (dropped) {
      batch.push({ level: "warn", text: `[devlog] dropped ${dropped} messages (rate limit)` });
      dropped = 0;
    }
    // keepalive so the last batch still lands during a page unload.
    fetch(ENDPOINT, {
      method: "POST",
      body: JSON.stringify(batch),
      keepalive: true,
      headers: { "Content-Type": "application/json" },
    }).catch(() => {
      /* server gone; nothing useful to do, and logging here would recurse */
    });
  }

  for (const level of LEVELS) {
    console[level] = (...args) => {
      original[level](...args);
      push(level, args);
    };
  }

  window.addEventListener("error", (e) => {
    const where = e.filename ? `${e.filename}:${e.lineno}:${e.colno}` : "";
    push("error", [e.error || e.message || "unknown script error"], where);
  });

  window.addEventListener("unhandledrejection", (e) => {
    push("error", ["Unhandled promise rejection:", e.reason]);
  });

  // Don't strand the tail of the queue when the tab closes.
  window.addEventListener("pagehide", flush);

  push("info", [`[devlog] attached — ${location.href}`]);
})();
