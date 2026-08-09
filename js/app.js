// Glues camera -> vision.js -> game/board state -> solverWorker -> UI.

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
  calibrateEmptyBtn: el("calibrateEmptyBtn"),
  calibrateStatus: el("calibrateStatus"),
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
  warpedCanvas: el("warpedCanvas"),
  hsvReadout: el("hsvReadout"),
  showDebugChk: el("showDebugChk"),
  debugPreviews: el("debugPreviews"),
  flipHChk: el("flipHChk"),
  flipVChk: el("flipVChk"),
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

async function startCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(
      "Camera API unavailable — this page must be served over https:// or http://localhost (not file://), and needs a modern browser."
    );
  }
  if (currentStream) {
    currentStream.getTracks().forEach((t) => t.stop());
  }
  const deviceId = dom.cameraSelect.value;
  const constraints = {
    video: deviceId
      ? { deviceId: { exact: deviceId } }
      : { facingMode: { ideal: "environment" } },
    audio: false,
  };
  const stream = await navigator.mediaDevices.getUserMedia(constraints);
  currentStream = stream;
  dom.video.srcObject = stream;
  await dom.video.play();
  await populateCameraList();
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
  const src = cv.imread(dom.captureCanvas);
  let result;
  try {
    result = Vision.processFrame(cv, src, {
      warpedCanvas: showDebug ? dom.warpedCanvas : null,
      debugMaskCanvas: showDebug ? dom.maskCanvas : null,
    });
  } finally {
    src.delete();
  }

  drawOverlay(result.found ? result.corners : null);
  updateDetectStatus(result.found);
  logDetection(result);

  if (result.found && result.grid) {
    ingestGrid(result.grid);
  }

  renderBoard();
}

let lastFoundLogged = null;

// Mirrors per-tick detection detail to console -> devlog -> dev-console.log
// so misclassifications can be diagnosed from the log instead of only via
// the live HSV-inspector click UI.
function logDetection(result) {
  if (result.found !== lastFoundLogged) {
    lastFoundLogged = result.found;
    console.log(`[vision] board ${result.found ? "found" : "lost"}`);
  }
  if (result.found && result.grid && result.colors) {
    console.debug("[vision] frame\n" + Vision.formatGridDebug(result.grid, result.colors));
  }
}

function drawOverlay(corners) {
  const ctx = dom.overlay.getContext("2d");
  ctx.clearRect(0, 0, dom.overlay.width, dom.overlay.height);
  if (!corners) return;
  ctx.strokeStyle = "#4da3ff";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(corners[0].x, corners[0].y);
  for (let i = 1; i < corners.length; i++) ctx.lineTo(corners[i].x, corners[i].y);
  ctx.closePath();
  ctx.stroke();
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

dom.resetBtn.addEventListener("click", () => {
  committedBoard = makeEmptyBoard();
  stabilityBuffer = makeEmptyBuffer();
  turnOverride = null;
  lastSolvedSignature = null;
  dom.suggestionText.textContent = "Point the camera at the board to begin.";
  dom.evalBar.style.left = "50%";
  clearHighlight();
  renderBoard();
  updateTurnStatus();
});

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

// ---- Vision settings wiring ---------------------------------------------
function bindSetting(inputId, key, isFloat = false) {
  const input = el(inputId);
  input.value = Vision.settings[key];
  input.addEventListener("input", () => {
    Vision.settings[key] = isFloat ? parseFloat(input.value) : parseInt(input.value, 10);
  });
}

[
  ["s_blueHueMin", "blueHueMin"],
  ["s_blueHueMax", "blueHueMax"],
  ["s_blueSatMin", "blueSatMin"],
  ["s_blueValMin", "blueValMin"],
  ["s_redHueMax1", "redHueMax1"],
  ["s_redHueMin2", "redHueMin2"],
  ["s_redSatMin", "redSatMin"],
  ["s_redValMin", "redValMin"],
  ["s_greenHueMin", "greenHueMin"],
  ["s_greenHueMax", "greenHueMax"],
  ["s_greenSatMin", "greenSatMin"],
  ["s_greenValMin", "greenValMin"],
].forEach(([id, key]) => bindSetting(id, key));

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

dom.calibrateEmptyBtn.addEventListener("click", () => {
  const ok = Vision.calibrateEmpty();
  dom.calibrateStatus.textContent = ok
    ? "Calibrated — empty holes in this lighting are now the baseline for detecting pieces."
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

dom.startBtn.addEventListener("click", () => {
  dom.cvStatus.textContent = "Requesting camera…";
  startCamera()
    .then(() => {
      dom.cvStatus.textContent = cvReady ? "Vision engine ready." : "Camera running — vision engine still loading…";
    })
    .catch((e) => {
      const hint = CAMERA_ERROR_HINTS[e.name];
      dom.cvStatus.textContent = `Camera error (${e.name || "Error"}): ${e.message}` + (hint ? " — " + hint : "");
      console.error(e);
    });
});

navigator.mediaDevices?.addEventListener?.("devicechange", populateCameraList);
window.addEventListener("resize", setupCanvasSizes);

renderBoard();
updateTurnStatus();
