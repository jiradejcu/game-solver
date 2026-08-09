// Glues camera -> vision.js -> game/board state -> solverWorker -> UI.

// ---- Persisted color calibration (so blue/red/green don't need re-calibrating every reload) ----
const COLOR_SETTINGS_KEY = "mumhy-vision-colors";
const COLOR_SETTING_KEYS = [
  "blueHueMin", "blueHueMax", "blueSatMin", "blueValMin",
  "redHueMax1", "redHueMin2", "redSatMin", "redValMin",
  "greenHueMin", "greenHueMax", "greenSatMin", "greenValMin",
];

function loadSavedColorSettings() {
  let saved;
  try {
    saved = JSON.parse(localStorage.getItem(COLOR_SETTINGS_KEY));
  } catch {
    return; // corrupt/old data -- keep defaults
  }
  if (!saved) return;
  for (const key of COLOR_SETTING_KEYS) {
    if (typeof saved[key] === "number") Vision.settings[key] = saved[key];
  }
  console.log("[calibrate] loaded saved blue/red/green calibration from localStorage");
}

function saveColorSettings() {
  const toSave = {};
  for (const key of COLOR_SETTING_KEYS) toSave[key] = Vision.settings[key];
  localStorage.setItem(COLOR_SETTINGS_KEY, JSON.stringify(toSave));
}

loadSavedColorSettings();

const BOARD_ROWS = Connect4Solver.ROWS;
const BOARD_COLS = Connect4Solver.COLS;
const STABILITY_FRAMES = 3; // consecutive matching frames required before a slot is committed

const el = (id) => document.getElementById(id);

const dom = {
  cvStatus: el("cvStatus"),
  video: el("video"),
  overlay: el("overlay"),
  videoWrap: document.querySelector(".video-wrap"),
  startBtn: el("startBtn"),
  cameraSelect: el("cameraSelect"),
  pauseChk: el("pauseChk"),
  snapshotBtn: el("snapshotBtn"),
  snapshotStatus: el("snapshotStatus"),
  calibrateEmptyBtn: el("calibrateEmptyBtn"),
  calibrateStatus: el("calibrateStatus"),
  calibrateBlueBtn: el("calibrateBlueBtn"),
  calibrateRedBtn: el("calibrateRedBtn"),
  calibrateGreenBtn: el("calibrateGreenBtn"),
  calibrateColorStatus: el("calibrateColorStatus"),
  detectStatus: el("detectStatus"),
  turnStatus: el("turnStatus"),
  boardGrid: el("boardGrid"),
  arrowRow: el("arrowRow"),
  resetBtn: el("resetBtn"),
  recomputeBtn: el("recomputeBtn"),
  swapTurnBtn: el("swapTurnBtn"),
  suggestionText: el("suggestionText"),
  evalBar: el("evalBar"),
  captureCanvas: el("captureCanvas"),
  maskCanvas: el("maskCanvas"),
  redMaskCanvas: el("redMaskCanvas"),
  greenMaskCanvas: el("greenMaskCanvas"),
  warpedCanvas: el("warpedCanvas"),
  hsvReadout: el("hsvReadout"),
  showDebugChk: el("showDebugChk"),
  debugPreviews: el("debugPreviews"),
  flipHChk: el("flipHChk"),
  flipVChk: el("flipVChk"),
  showGridOverlayChk: el("showGridOverlayChk"),
  showHoughDebugChk: el("showHoughDebugChk"),
};

// ---- Game state -----------------------------------------------------
// committedBoard[row][col], row 0 = bottom (gravity), 0/1/2.
let committedBoard = makeEmptyBoard();
let stabilityBuffer = makeEmptyBuffer(); // per-cell {value, count} in image space (row 0 = top)
let turnOverride = null; // 1, 2, or null (auto from parity)
let lastSolvedSignature = null;
let requestCounter = 0;

function makeEmptyBoard() {
  return Array.from({ length: BOARD_ROWS }, () => Array(BOARD_COLS).fill(0));
}
function makeEmptyBuffer() {
  return Array.from({ length: BOARD_ROWS }, () => Array.from({ length: BOARD_COLS }, () => ({ value: "empty", count: 0 })));
}

// ---- OpenCV lifecycle -------------------------------------------------
// Camera capture doesn't depend on OpenCV at all, so the Start button is
// always clickable; only the per-frame detection loop waits on cvReady.
// `cv` is a global injected by the opencv.js <script> tag — if that script
// failed to load (network/firewall issue) `cv` won't exist yet, so we guard
// and retry instead of letting a bare reference crash this whole file (which
// would silently break every button on the page, including Start Camera).
let cvReady = false;

function wireOpenCv() {
  if (typeof cv === "undefined") return false;
  cv["onRuntimeInitialized"] = () => {
    cvReady = true;
    dom.cvStatus.textContent = "Vision engine ready.";
  };
  return true;
}

if (!wireOpenCv()) {
  let cvWireAttempts = 0;
  const cvWireRetry = setInterval(() => {
    cvWireAttempts += 1;
    if (wireOpenCv() || cvWireAttempts > 100) clearInterval(cvWireRetry); // give up after ~20s
  }, 200);
}

// ---- Solver worker ------------------------------------------------------
const worker = new Worker("js/solverWorker.js");
worker.onmessage = (e) => {
  const { requestId, result, error } = e.data;
  if (requestId !== requestCounter) return; // stale response
  if (error) {
    dom.suggestionText.textContent = "Solver error: " + error;
    return;
  }
  renderSuggestion(result);
};

function requestBestMove() {
  const turn = getCurrentTurn();
  requestCounter += 1;
  worker.postMessage({
    requestId: requestCounter,
    board: Connect4Solver.cloneBoard(committedBoard),
    aiPlayer: turn,
    timeBudgetMs: 1200,
  });
}

// ---- Camera setup ---------------------------------------------------
async function populateCameraList() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const cams = devices.filter((d) => d.kind === "videoinput").reverse();
    dom.cameraSelect.innerHTML = "";
    cams.forEach((c, i) => {
      const opt = document.createElement("option");
      opt.value = c.deviceId;
      opt.textContent = c.label || `Camera ${i + 1}`;
      dom.cameraSelect.appendChild(opt);
    });
  } catch (e) {
    // enumerateDevices may fail before permission is granted; ignore.
  }
}

let currentStream = null;
const CAMERA_DEVICE_KEY = "mumhy-camera-device-id";

// `preferredDeviceId`, if given, overrides the dropdown -- used by the
// auto-resume-after-reload path (see attemptStartCamera at the bottom of
// this file), since the dropdown has no options yet that early (it's only
// populated after a stream is granted once) and would otherwise silently
// fall back to a default/environment-facing camera instead of whichever one
// was actually in use.
async function startCamera(preferredDeviceId) {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(
      "Camera API unavailable — this page must be served over https:// or http://localhost (not file://), and needs a modern browser."
    );
  }
  if (currentStream) {
    currentStream.getTracks().forEach((t) => t.stop());
  }
  const deviceId = preferredDeviceId || dom.cameraSelect.value;
  const constraints = {
    video: deviceId
      ? { deviceId: { exact: deviceId } }
      : { facingMode: { ideal: "environment" } },
    audio: false,
  };
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia(constraints);
  } catch (e) {
    // The remembered/selected device may no longer exist (unplugged,
    // re-enumerated after a reboot, etc.) -- fall back to the default
    // instead of failing outright.
    if (deviceId && e.name === "OverconstrainedError") {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: "environment" } }, audio: false });
    } else {
      throw e;
    }
  }
  currentStream = stream;
  dom.video.srcObject = stream;
  await dom.video.play();
  await populateCameraList();

  // Reflect + remember whichever device was actually granted (populating the
  // dropdown above resets its selection to the list's first entry, which
  // isn't necessarily this one) so a later auto-resume goes straight to the
  // right camera instead of needing a second manual click to fix it.
  const activeDeviceId = stream.getVideoTracks()[0]?.getSettings().deviceId;
  if (activeDeviceId) {
    dom.cameraSelect.value = activeDeviceId;
    localStorage.setItem(CAMERA_DEVICE_KEY, activeDeviceId);
  }

  setupCanvasSizes();
  startDetectionLoop();
}

function setupCanvasSizes() {
  const vw = dom.video.videoWidth;
  const vh = dom.video.videoHeight;
  if (!vw || !vh) return;
  dom.videoWrap.style.aspectRatio = `${vw} / ${vh}`;

  const maxW = 480;
  const capW = Math.min(maxW, vw);
  const capH = Math.round((capW * vh) / vw);
  dom.captureCanvas.width = capW;
  dom.captureCanvas.height = capH;
  dom.overlay.width = capW;
  dom.overlay.height = capH;
}

// ---- Snapshot capture (dev aid: see server.py's /__snapshot) ----------
// Composites the current camera frame + overlay (board outline, grid dots,
// raw Hough debug) plus the mask/warped debug previews into one PNG and
// posts it to the dev server, which writes it to snapshot.png. Lets a look
// at the file substitute for eyes on the live page.
async function captureSnapshot() {
  if (!dom.overlay.width || !dom.overlay.height) {
    dom.snapshotStatus.textContent = "Start the camera first.";
    return;
  }

  // Debug/Hough previews only get drawn into when their checkboxes are on --
  // force them on and wait one detection tick so a snapshot always has the
  // full picture, instead of silently omitting panels a caller forgot to enable.
  if (!dom.showDebugChk.checked || !dom.showHoughDebugChk.checked) {
    dom.showDebugChk.checked = true;
    dom.debugPreviews.classList.remove("hidden");
    dom.showHoughDebugChk.checked = true;
    await new Promise((resolve) => setTimeout(resolve, 400));
  }

  const mainW = dom.overlay.width;
  const mainH = dom.overlay.height;
  const previews = [
    { canvas: dom.maskCanvas, label: "blue mask" },
    { canvas: dom.redMaskCanvas, label: "red mask" },
    { canvas: dom.greenMaskCanvas, label: "green mask" },
    { canvas: dom.warpedCanvas, label: "warped board" },
  ].filter((p) => p.canvas.width > 0 && p.canvas.height > 0);

  // Stacked full-width (not a cramped side-by-side strip) so real detail --
  // a small gap in the base a few pixels wide -- doesn't get downscaled into
  // invisibility. captureSnapshot is the only way to actually see a frame of
  // what the vision pipeline is doing, so it needs to preserve enough
  // resolution for that to mean something.
  const labelH = 20;
  const panelLayout = previews.map((p) => {
    const h = Math.round(mainW * (p.canvas.height / p.canvas.width));
    return { ...p, h };
  });
  const previewH = panelLayout.reduce((sum, p) => sum + labelH + p.h, 0);

  const canvas = document.createElement("canvas");
  canvas.width = mainW;
  canvas.height = mainH + previewH;
  const ctx = canvas.getContext("2d");

  ctx.drawImage(dom.video, 0, 0, mainW, mainH);
  ctx.drawImage(dom.overlay, 0, 0);

  let y = mainH;
  panelLayout.forEach((p) => {
    ctx.fillStyle = "#111";
    ctx.fillRect(0, y, mainW, labelH + p.h);
    ctx.fillStyle = "#fff";
    ctx.font = "13px sans-serif";
    ctx.fillText(p.label, 4, y + 14);
    ctx.imageSmoothingEnabled = false; // nearest-neighbor -- don't blur away small real gaps
    ctx.drawImage(p.canvas, 0, y + labelH, mainW, p.h);
    ctx.imageSmoothingEnabled = true;
    y += labelH + p.h;
  });

  fetch("/__snapshot", {
    method: "POST",
    body: canvas.toDataURL("image/png"),
    headers: { "Content-Type": "text/plain" },
  })
    .then((r) => {
      if (!r.ok) {
        dom.snapshotStatus.textContent = "Snapshot failed to save.";
      } else if (previews.length) {
        dom.snapshotStatus.textContent = "Saved to snapshot.png (with mask previews).";
      } else {
        dom.snapshotStatus.textContent = 'Saved to snapshot.png -- check "Show mask / warped preview" to include the mask panels too.';
      }
    })
    .catch(() => {
      dom.snapshotStatus.textContent = "Snapshot failed (dev server not reachable).";
    });
}

dom.snapshotBtn.addEventListener("click", captureSnapshot);

// Lets an external process (e.g. an agent iterating on vision.js) request a
// snapshot without a human clicking the button: touch snapshot-request.flag
// in the project root (see server.py), and this picks up the mtime change
// within a second and captures automatically.
(function watchSnapshotRequests() {
  const POLL_MS = 1000;
  let baseline = null;

  async function check() {
    let res;
    try {
      res = await fetch("/__snapshot-request-version", { cache: "no-store" });
    } catch {
      return;
    }
    if (!res.ok) return;
    const { version } = await res.json();
    if (baseline === null) {
      baseline = version;
      return;
    }
    if (version !== baseline) {
      baseline = version;
      captureSnapshot();
    }
  }

  setInterval(check, POLL_MS);
  check();
})();

// ---- Detection loop ---------------------------------------------------
let detectionTimer = null;

function startDetectionLoop() {
  if (detectionTimer) clearInterval(detectionTimer);
  detectionTimer = setInterval(detectTick, 300);
}

function detectTick() {
  if (!cvReady) {
    dom.detectStatus.textContent = "Vision engine still loading…";
    return;
  }
  if (dom.pauseChk.checked) return;
  if (!dom.video.videoWidth) return;

  if (dom.captureCanvas.width !== Math.min(480, dom.video.videoWidth)) {
    setupCanvasSizes();
  }

  const ctx = dom.captureCanvas.getContext("2d");
  ctx.drawImage(dom.video, 0, 0, dom.captureCanvas.width, dom.captureCanvas.height);

  const showDebug = dom.showDebugChk.checked;
  const houghDebug = { collectRaw: dom.showHoughDebugChk.checked };
  const src = cv.imread(dom.captureCanvas);
  let result;
  try {
    result = Vision.processFrame(cv, src, {
      warpedCanvas: showDebug ? dom.warpedCanvas : null,
      debugMaskCanvas: showDebug ? dom.maskCanvas : null,
      debugRedCanvas: showDebug ? dom.redMaskCanvas : null,
      debugGreenCanvas: showDebug ? dom.greenMaskCanvas : null,
      debugHough: houghDebug,
    });
  } finally {
    src.delete();
  }

  drawOverlay(result, houghDebug);
  updateDetectStatus(result.found);
  logDetection(result, houghDebug);

  if (result.found && result.grid) {
    ingestGrid(result.grid);
  }

  renderBoard();
}

let lastFoundLogged = null;

// Mirrors per-tick detection detail to console -> devlog -> dev-console.log
// so misclassifications can be diagnosed from the log instead of only via
// the live HSV-inspector click UI.
function logDetection(result, houghDebug) {
  if (result.found !== lastFoundLogged) {
    lastFoundLogged = result.found;
    console.log(`[vision] board ${result.found ? "found" : "lost"}`);
  }
  if (result.found && houghDebug) {
    console.log(
      `[corners] method=${houghDebug.cornerMethod} period=${houghDebug.period} ` +
      `bboxTop=${houghDebug.bboxTop} bboxBottom=${houghDebug.bboxBottom} ` +
      `rowCounts=${JSON.stringify(houghDebug.rowCounts)} allRowCenters=${JSON.stringify(houghDebug.allRowCenters)}`
    );
  }
  if (result.found && result.grid && result.colors) {
    console.debug("[vision] frame\n" + Vision.formatGridDebug(result.grid, result.colors));
  }
}

const CLASSIFICATION_COLOR = {
  red: "#ff4d4d",
  green: "#39d353",
  empty: "rgba(255,255,255,0.4)",
};

function drawOverlay(result, houghDebug) {
  const ctx = dom.overlay.getContext("2d");
  ctx.clearRect(0, 0, dom.overlay.width, dom.overlay.height);

  // Drawn first (underneath) regardless of whether a board was ultimately
  // found, since seeing every raw candidate is exactly what's needed to
  // diagnose *why* nothing was found or why the pick keeps changing.
  drawHoughDebug(ctx, houghDebug);

  if (!result || !result.found) return;

  const [tl, tr, br, bl] = result.corners;

  // Just the plain detected quad now (minAreaRect of the blue blob, inset by
  // a fixed border trim) -- no more per-edge reconstruction/refinement to
  // distinguish visually, see findBoardCorners.
  ctx.lineWidth = 3;
  ctx.strokeStyle = "#4da3ff";
  ctx.beginPath();
  ctx.moveTo(tl.x, tl.y);
  ctx.lineTo(tr.x, tr.y);
  ctx.lineTo(br.x, br.y);
  ctx.lineTo(bl.x, bl.y);
  ctx.closePath();
  ctx.stroke();

  if (!dom.showGridOverlayChk.checked || !result.grid || !result.orderedCorners) return;
  drawGridClassification(ctx, result);
}

// Draws every raw Hough line/circle candidate this frame produced -- purely
// for visual calibration checking (do detected shapes land where the real
// frame/holes actually are?). No picking or fitting happens on this data
// anymore; see collectHoughDebug in vision.js.
function drawHoughDebug(ctx, hd) {
  if (!hd) return;

  ctx.lineWidth = 1;
  ctx.strokeStyle = "rgba(255, 235, 59, 0.7)"; // all raw horizontal line candidates
  (hd.horizontals || []).forEach((s) => {
    ctx.beginPath();
    ctx.moveTo(s.x1, s.y1);
    ctx.lineTo(s.x2, s.y2);
    ctx.stroke();
  });

  ctx.strokeStyle = "rgba(0, 229, 255, 0.7)"; // all raw vertical line candidates
  (hd.verticals || []).forEach((s) => {
    ctx.beginPath();
    ctx.moveTo(s.x1, s.y1);
    ctx.lineTo(s.x2, s.y2);
    ctx.stroke();
  });

  ctx.lineWidth = 1;
  ctx.strokeStyle = "rgba(255, 0, 229, 0.8)"; // every raw Hough circle (hole candidate)
  (hd.circles || []).forEach((c) => {
    ctx.beginPath();
    ctx.arc(c.x, c.y, c.r, 0, Math.PI * 2);
    ctx.stroke();
  });
}

// Paints each detected slot's live classification (red/green/empty) as a
// dot directly on the camera image, at the same warped-grid position
// sampleGrid read it from — so misreads are visible at a glance instead of
// only in the [vision] frame console/devlog dump.
function drawGridClassification(ctx, result) {
  const centers = Vision.mapPointsToSource(cv, result.orderedCorners, Vision.cellCenterPoints());
  const radius = 8;
  let i = 0;
  for (let row = 0; row < BOARD_ROWS; row++) {
    for (let col = 0; col < BOARD_COLS; col++) {
      const { x, y } = centers[i++];
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fillStyle = CLASSIFICATION_COLOR[result.grid[row][col]] || CLASSIFICATION_COLOR.empty;
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = "rgba(0,0,0,0.6)";
      ctx.stroke();
    }
  }
}

function updateDetectStatus(found) {
  dom.detectStatus.textContent = found ? "Board detected" : "No board detected";
  dom.detectStatus.classList.toggle("badge-on", found);
  dom.detectStatus.classList.toggle("badge-off", !found);
}

// Merges one frame's raw grid (image space: row 0 = top) into committedBoard.
function ingestGrid(grid) {
  // 1. Gravity cleanup per column: anything above the first empty slot
  //    (scanning from the physical bottom upward) is treated as noise.
  const cleaned = grid.map((row) => row.slice());
  for (let c = 0; c < BOARD_COLS; c++) {
    let seenEmpty = false;
    for (let rImg = BOARD_ROWS - 1; rImg >= 0; rImg--) {
      if (seenEmpty) {
        cleaned[rImg][c] = "empty";
      } else if (cleaned[rImg][c] === "empty") {
        seenEmpty = true;
      }
    }
  }

  // 2. Temporal stability: only accept a reading once it repeats.
  let changed = false;
  for (let rImg = 0; rImg < BOARD_ROWS; rImg++) {
    for (let c = 0; c < BOARD_COLS; c++) {
      const reading = cleaned[rImg][c];
      const buf = stabilityBuffer[rImg][c];
      if (buf.value === reading) {
        buf.count = Math.min(buf.count + 1, STABILITY_FRAMES);
      } else {
        buf.value = reading;
        buf.count = 1;
      }

      if (buf.count < STABILITY_FRAMES) continue;

      const boardRow = BOARD_ROWS - 1 - rImg; // convert image row (0=top) to board row (0=bottom)
      const current = committedBoard[boardRow][c];

      if (reading === "empty") {
        // Never un-commit a piece from a transient empty reading (occlusion, glare).
        continue;
      }
      const value = reading === "red" ? 1 : 2;
      if (current === 0) {
        committedBoard[boardRow][c] = value;
        changed = true;
      } else if (current !== value) {
        // Persistent disagreement on an already-committed slot: trust the
        // newer stable reading (handles an earlier misclassification).
        committedBoard[boardRow][c] = value;
        changed = true;
      }
    }
  }

  if (changed) {
    console.log("[board] committed change:\n" + committedBoard.map((row) => row.join(" ")).join("\n"));
    turnOverride = null;
    maybeSolve();
  }
}

// ---- Turn tracking ----------------------------------------------------
function getFirstPlayer() {
  const checked = document.querySelector('input[name="firstPlayer"]:checked');
  return checked ? parseInt(checked.value, 10) : 1;
}

function getCurrentTurn() {
  if (turnOverride) return turnOverride;
  const totalMoves = committedBoard.flat().filter((v) => v !== 0).length;
  const first = getFirstPlayer();
  const other = first === 1 ? 2 : 1;
  return totalMoves % 2 === 0 ? first : other;
}

dom.swapTurnBtn.addEventListener("click", () => {
  turnOverride = getCurrentTurn() === 1 ? 2 : 1;
  updateTurnStatus();
  maybeSolve();
});

document.querySelectorAll('input[name="firstPlayer"]').forEach((r) =>
  r.addEventListener("change", () => {
    turnOverride = null;
    updateTurnStatus();
    maybeSolve();
  })
);

function updateTurnStatus() {
  const turn = getCurrentTurn();
  dom.turnStatus.textContent = "Turn: " + (turn === 1 ? "Red" : "Green");
}

// ---- Solve trigger (debounced by board signature) ----------------------
function boardSignature() {
  return committedBoard.flat().join("") + "|" + getCurrentTurn();
}

function maybeSolve() {
  const sig = boardSignature();
  if (sig === lastSolvedSignature) return;
  lastSolvedSignature = sig;
  updateTurnStatus();

  const totalMoves = committedBoard.flat().filter((v) => v !== 0).length;
  if (totalMoves === 42) {
    dom.suggestionText.textContent = "Board is full.";
    return;
  }
  dom.suggestionText.textContent = "Thinking…";
  requestBestMove();
}

dom.recomputeBtn.addEventListener("click", () => {
  lastSolvedSignature = null;
  maybeSolve();
});

function resetBoardState() {
  committedBoard = makeEmptyBoard();
  stabilityBuffer = makeEmptyBuffer();
  turnOverride = null;
  lastSolvedSignature = null;
  dom.suggestionText.textContent = "Point the camera at the board to begin.";
  dom.evalBar.style.left = "50%";
  clearHighlight();
  renderBoard();
  updateTurnStatus();
}

dom.resetBtn.addEventListener("click", resetBoardState);

// ---- Board diagram rendering & manual correction -----------------------
function renderBoard() {
  dom.boardGrid.innerHTML = "";
  // Display top row first (visual row 0 = physical top = board row BOARD_ROWS-1).
  for (let displayRow = 0; displayRow < BOARD_ROWS; displayRow++) {
    const boardRow = BOARD_ROWS - 1 - displayRow;
    for (let c = 0; c < BOARD_COLS; c++) {
      const val = committedBoard[boardRow][c];
      const cellDiv = document.createElement("div");
      cellDiv.className = "cell" + (val === 1 ? " red" : val === 2 ? " green" : "");
      cellDiv.dataset.col = c;
      cellDiv.dataset.row = boardRow;
      cellDiv.addEventListener("click", () => onCellClick(c, boardRow));
      dom.boardGrid.appendChild(cellDiv);
    }
  }
}

function onCellClick(col, row) {
  const openRow = Connect4Solver.getOpenRow(committedBoard, col);
  const topFilledRow = openRow === -1 ? BOARD_ROWS - 1 : openRow - 1;

  if (row === topFilledRow && topFilledRow >= 0) {
    const cur = committedBoard[row][col];
    committedBoard[row][col] = cur === 1 ? 2 : cur === 2 ? 0 : 1;
  } else if (row === openRow) {
    committedBoard[row][col] = getCurrentTurn();
  } else {
    return; // ignore clicks that would create a floating piece
  }
  turnOverride = null;
  renderBoard();
  maybeSolve();
}

function renderArrows(bestCol) {
  dom.arrowRow.innerHTML = "";
  for (let c = 0; c < BOARD_COLS; c++) {
    const div = document.createElement("div");
    div.className = "arrow" + (c === bestCol ? " active" : "");
    div.textContent = "▼";
    dom.arrowRow.appendChild(div);
  }
}

function clearHighlight() {
  dom.arrowRow.innerHTML = "";
  document.querySelectorAll(".cell.highlight-col").forEach((c) => c.classList.remove("highlight-col"));
}

function highlightColumn(col) {
  document.querySelectorAll(".cell").forEach((cellDiv) => {
    cellDiv.classList.toggle("highlight-col", parseInt(cellDiv.dataset.col, 10) === col);
  });
}

function renderSuggestion(result) {
  if (!result) {
    dom.suggestionText.textContent = "No legal moves.";
    clearHighlight();
    return;
  }
  const turn = getCurrentTurn();
  const colorName = turn === 1 ? "Red" : "Green";
  const humanCol = result.column + 1;

  let verdict = "";
  if (result.forced === "win" || result.evaluation > 900000) {
    verdict = ` — ${colorName} has a forced win!`;
  } else if (result.evaluation < -900000) {
    verdict = " — this looks lost, but here's the best try.";
  }

  dom.suggestionText.textContent = `${colorName}: drop in column ${humanCol}${verdict}`;
  renderArrows(result.column);
  highlightColumn(result.column);

  // Eval bar: clamp heuristic score into a -1..1 range for display.
  const clamped = Math.max(-1, Math.min(1, result.evaluation / 5000));
  const pct = 50 + clamped * 50 * (turn === 1 ? 1 : -1);
  dom.evalBar.style.left = `${pct}%`;
}

// ---- Manual HSV slider calibration (alongside the eyedropper above) ------
function bindColorSlider(key) {
  const input = el("s_" + key);
  input.value = Vision.settings[key];
  input.addEventListener("input", () => {
    Vision.settings[key] = parseInt(input.value, 10);
    saveColorSettings();
  });
}
COLOR_SETTING_KEYS.forEach(bindColorSlider);

// Reflects Vision.settings back into the sliders after a programmatic
// change (e.g. the eyedropper), since bindColorSlider only wires input -> settings.
function syncColorSliders() {
  COLOR_SETTING_KEYS.forEach((key) => {
    el("s_" + key).value = Vision.settings[key];
  });
}

function invalidateEmptyCalibration(reason) {
  if (!Vision.hasEmptyCalibration()) return;
  Vision.clearEmptyCalibration();
  dom.calibrateStatus.textContent = `Calibration cleared (${reason}) — recalibrate with an empty board.`;
}

dom.flipHChk.addEventListener("change", () => {
  Vision.settings.flipH = dom.flipHChk.checked;
  invalidateEmptyCalibration("orientation changed");
});
dom.flipVChk.addEventListener("change", () => {
  Vision.settings.flipV = dom.flipVChk.checked;
  invalidateEmptyCalibration("orientation changed");
});
dom.showDebugChk.addEventListener("change", () => {
  dom.debugPreviews.classList.toggle("hidden", !dom.showDebugChk.checked);
});

dom.warpedCanvas.addEventListener("click", (e) => {
  const rect = dom.warpedCanvas.getBoundingClientRect();
  const scaleX = dom.warpedCanvas.width / rect.width;
  const scaleY = dom.warpedCanvas.height / rect.height;
  const px = (e.clientX - rect.left) * scaleX;
  const py = (e.clientY - rect.top) * scaleY;
  const c = Math.min(BOARD_COLS - 1, Math.max(0, Math.floor(px / Vision.CELL_PX)));
  const rImg = Math.min(BOARD_ROWS - 1, Math.max(0, Math.floor(py / Vision.CELL_PX)));
  const info = Vision.debugCell(rImg, c);
  const text = info
    ? `Row ${rImg + 1} (from top), Col ${c + 1} — H:${info.h.toFixed(0)} S:${info.s.toFixed(0)} V:${info.v.toFixed(0)} → classified as ${info.classification}`
    : "No sample yet at that spot — wait for the board to be detected.";
  dom.hsvReadout.textContent = text;
  console.log("[hsv-inspect] " + text);
});

// ---- Color calibration (click-to-sample instead of hand-tuning sliders) ----
function rgbToHsvCv(r, g, b) {
  r /= 255; g /= 255; b /= 255;
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const delta = max - min;
  let h = 0;
  if (delta !== 0) {
    if (max === r) h = 60 * (((g - b) / delta) % 6);
    else if (max === g) h = 60 * ((b - r) / delta + 2);
    else h = 60 * ((r - g) / delta + 4);
    if (h < 0) h += 360;
  }
  const s = max === 0 ? 0 : delta / max;
  const v = max;
  return { h: h / 2, s: s * 255, v: v * 255 }; // OpenCV scale: H 0-179, S/V 0-255
}

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, Math.round(v)));

const COLOR_CALIBRATION_LABELS = { blue: "blue frame", red: "red piece", green: "green piece" };
let pendingColorCalibration = null; // "blue" | "red" | "green" | null

function armColorCalibration(kind) {
  pendingColorCalibration = kind;
  dom.calibrateColorStatus.textContent = `Click the ${COLOR_CALIBRATION_LABELS[kind]} in the camera preview above…`;
}

dom.calibrateBlueBtn.addEventListener("click", () => armColorCalibration("blue"));
dom.calibrateRedBtn.addEventListener("click", () => armColorCalibration("red"));
dom.calibrateGreenBtn.addEventListener("click", () => armColorCalibration("green"));

dom.videoWrap.addEventListener("click", (e) => {
  if (!pendingColorCalibration) return;
  const kind = pendingColorCalibration;
  pendingColorCalibration = null;

  const rect = dom.overlay.getBoundingClientRect();
  const scaleX = dom.overlay.width / rect.width;
  const scaleY = dom.overlay.height / rect.height;
  const x = Math.round((e.clientX - rect.left) * scaleX);
  const y = Math.round((e.clientY - rect.top) * scaleY);

  const half = 3;
  const x0 = Math.max(0, x - half);
  const y0 = Math.max(0, y - half);
  const w = Math.min(dom.captureCanvas.width - x0, half * 2 + 1);
  const h = Math.min(dom.captureCanvas.height - y0, half * 2 + 1);
  if (w <= 0 || h <= 0) {
    dom.calibrateColorStatus.textContent = "Click was outside the camera frame — try again.";
    return;
  }

  const { data } = dom.captureCanvas.getContext("2d").getImageData(x0, y0, w, h);
  let rSum = 0, gSum = 0, bSum = 0, n = 0;
  for (let i = 0; i < data.length; i += 4) {
    rSum += data[i];
    gSum += data[i + 1];
    bSum += data[i + 2];
    n++;
  }
  const { h: hh, s: ss, v: vv } = rgbToHsvCv(rSum / n, gSum / n, bSum / n);

  // A single click samples one point, but a real piece's surface isn't one
  // color -- specular highlights from glossy plastic push V way up in spots,
  // shadows/angle push both S and V down elsewhere. These margins only
  // lower the *minimum* (there's no upper bound on sat/val, see
  // classifyHSV), so they need to be generous enough to cover that natural
  // per-piece spread, not just camera noise around a single sample point.
  const HUE_TOLERANCE = 15;
  const SAT_MARGIN = 70;
  const VAL_MARGIN = 70;
  const satMin = clamp(ss - SAT_MARGIN, 0, 255);
  const valMin = clamp(vv - VAL_MARGIN, 0, 255);
  let summary;

  if (kind === "blue") {
    Vision.settings.blueHueMin = clamp(hh - HUE_TOLERANCE, 0, 179);
    Vision.settings.blueHueMax = clamp(hh + HUE_TOLERANCE, 0, 179);
    Vision.settings.blueSatMin = satMin;
    Vision.settings.blueValMin = valMin;
    summary = `hue ${Vision.settings.blueHueMin}-${Vision.settings.blueHueMax}, sat >= ${satMin}, val >= ${valMin}`;
  } else if (kind === "green") {
    Vision.settings.greenHueMin = clamp(hh - HUE_TOLERANCE, 0, 179);
    Vision.settings.greenHueMax = clamp(hh + HUE_TOLERANCE, 0, 179);
    Vision.settings.greenSatMin = satMin;
    Vision.settings.greenValMin = valMin;
    summary = `hue ${Vision.settings.greenHueMin}-${Vision.settings.greenHueMax}, sat >= ${satMin}, val >= ${valMin}`;
  } else {
    // Red wraps around 0/180 on OpenCV's hue wheel: classifyHSV treats it as
    // (h <= redHueMax1 || h >= redHueMin2). The sampled hue only ever sits on
    // one side of that wrap, so pin the far bound to a no-op value (0 or 179)
    // instead of leaving it at its old value, which could falsely cover the
    // whole opposite arc.
    if (hh <= 90) {
      Vision.settings.redHueMax1 = clamp(hh + HUE_TOLERANCE, 0, 179);
      Vision.settings.redHueMin2 = 179;
    } else {
      Vision.settings.redHueMin2 = clamp(hh - HUE_TOLERANCE, 0, 179);
      Vision.settings.redHueMax1 = 0;
    }
    Vision.settings.redSatMin = satMin;
    Vision.settings.redValMin = valMin;
    summary = `hue <= ${Vision.settings.redHueMax1} or >= ${Vision.settings.redHueMin2}, sat >= ${satMin}, val >= ${valMin}`;
  }

  saveColorSettings();
  syncColorSliders();
  dom.calibrateColorStatus.textContent =
    `Calibrated from sampled H:${hh.toFixed(0)} S:${ss.toFixed(0)} V:${vv.toFixed(0)} — ${summary}. Saved for next time.`;
  console.log(`[calibrate-${kind}] ` + dom.calibrateColorStatus.textContent);
});

dom.calibrateEmptyBtn.addEventListener("click", () => {
  const ok = Vision.calibrateEmpty();
  if (ok) {
    // Calibrating implies the physical board is empty right now, so drop any
    // previously committed pieces (which the sticky-commit logic in
    // ingestGrid would otherwise never clear from a plain "empty" reading).
    resetBoardState();
  }
  dom.calibrateStatus.textContent = ok
    ? "Calibrated — empty holes in this lighting are now the baseline for detecting pieces, and the board display was reset."
    : "No frame captured yet — start the camera and point it at the board first.";
  console.log(`[calibrate] empty-board calibration ${ok ? "captured" : "failed (no frame yet)"}`);
});

// ---- Wiring -------------------------------------------------------------
const CAMERA_ERROR_HINTS = {
  NotAllowedError: "Permission denied — allow camera access for this page (check the icon in the address bar), or Windows Settings > Privacy > Camera if it's blocked system-wide.",
  NotFoundError: "No camera found — check that a webcam is connected and enabled in Device Manager.",
  NotReadableError: "Camera is already in use by another app (Zoom, Teams, another browser tab, etc.) — close it and try again.",
  OverconstrainedError: "No camera matches the requested settings — try a different camera from the dropdown.",
  SecurityError: "Blocked by the browser's security policy — make sure you're on https:// or http://localhost.",
  AbortError: "Camera start was interrupted — try again.",
};

const CAMERA_WAS_ON_KEY = "mumhy-camera-was-on";

function attemptStartCamera(preferredDeviceId) {
  dom.cvStatus.textContent = "Requesting camera…";
  return startCamera(preferredDeviceId)
    .then(() => {
      dom.cvStatus.textContent = cvReady ? "Vision engine ready." : "Camera running — vision engine still loading…";
      localStorage.setItem(CAMERA_WAS_ON_KEY, "1");
    })
    .catch((e) => {
      const hint = CAMERA_ERROR_HINTS[e.name];
      dom.cvStatus.textContent = `Camera error (${e.name || "Error"}): ${e.message}` + (hint ? " — " + hint : "");
      console.error(e);
    });
}

dom.startBtn.addEventListener("click", () => attemptStartCamera());

navigator.mediaDevices?.addEventListener?.("devicechange", populateCameraList);
window.addEventListener("resize", setupCanvasSizes);

renderBoard();
updateTurnStatus();

// Camera permission, once granted, persists for the origin -- so once the
// camera has been started manually one time, it's safe to resume it
// automatically on every later load without a fresh user gesture. Mainly
// for js/livereload.js: without this, every auto-reload while iterating on
// vision.js would silently drop the camera and require re-clicking Start.
// Pass the remembered device id directly (see startCamera) instead of
// leaving it to fall back to a default -- the dropdown has no options yet
// this early, so an unpreferred fallback would need a second manual click
// to correct, same as the bug this is fixing.
if (localStorage.getItem(CAMERA_WAS_ON_KEY) === "1") {
  attemptStartCamera(localStorage.getItem(CAMERA_DEVICE_KEY));
}
