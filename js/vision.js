// Computer-vision board reader built on OpenCV.js.
//
// Pipeline per frame:
//   1. Threshold the frame for the board's blue frame color -> mask.
//   2. Find the largest contour in that mask and take its minAreaRect
//      as the grid's four corners (robust to a slightly rotated/tilted view).
//   3. Perspective-warp the grid to a flat top-down 7x6 image.
//   4. Sample the average HSV color at each of the 42 slot centers and
//      classify it as empty / red / green.
//
// All thresholds are tunable at runtime (see Vision.settings) because the
// exact color response depends on the camera and lighting.

const Vision = (() => {
  const ROWS = 6;
  const COLS = 7;
  const CELL_PX = 60; // size of one warped cell, in pixels
  const WARP_W = COLS * CELL_PX;
  const WARP_H = ROWS * CELL_PX;

  const settings = {
    blueHueMin: 90,
    blueHueMax: 130,
    blueSatMin: 60,
    blueValMin: 40,

    redHueMax1: 10, // red wraps around hue 0/180 in OpenCV's 0-179 hue space
    redHueMin2: 170,
    redSatMin: 90,
    redValMin: 55,

    greenHueMin: 35,
    greenHueMax: 90,
    greenSatMin: 55,
    greenValMin: 45,

    minAreaFraction: 0.03, // discard candidate board contour smaller than this fraction of the frame
    flipH: false,
    flipV: false,

    // Max color distance (see colorDelta) from the calibrated empty-slot
    // baseline for a cell to still count as empty. Lets an empty-board
    // calibration override hue-range misreads caused by lighting/white-balance
    // color casts (e.g. white/cream holes reading as green).
    emptyMaxDist: 45,
  };

  // Per-cell {h,s,v} sampled the last time an empty board was calibrated
  // (see calibrateEmpty), or null if never calibrated. ROWS x COLS.
  let emptyReference = null;
  // Per-cell {h,s,v} from the most recent sampleGrid call, used as the
  // source frame for calibrateEmpty.
  let lastColors = null;

  function colorDelta(a, b) {
    const dh = Math.min(Math.abs(a.h - b.h), 180 - Math.abs(a.h - b.h));
    const ds = Math.abs(a.s - b.s);
    const dv = Math.abs(a.v - b.v);
    return dh * 1.5 + ds * 0.6 + dv * 0.3;
  }

  // Captures the most recently sampled frame's colors as the empty-slot
  // baseline. Call this while the physical board is empty. Returns false if
  // no frame has been sampled yet.
  function calibrateEmpty() {
    if (!lastColors) return false;
    emptyReference = lastColors.map((row) => row.map((c) => ({ ...c })));
    return true;
  }

  function clearEmptyCalibration() {
    emptyReference = null;
  }

  function hasEmptyCalibration() {
    return !!emptyReference;
  }

  // Debug helper: returns the last-sampled {h,s,v,classification} for one
  // grid cell (image space: row 0 = top), or null if no frame sampled yet.
  function debugCell(r, c) {
    if (!lastColors || !lastColors[r] || !lastColors[r][c]) return null;
    const { h, s, v } = lastColors[r][c];
    return { h, s, v, classification: classifyCell(h, s, v, r, c) };
  }

  function orderCorners(pts) {
    // pts: [{x,y} x4] -> returns [TL, TR, BR, BL]
    const bySum = [...pts].sort((a, b) => a.x + a.y - (b.x + b.y));
    const tl = bySum[0];
    const br = bySum[3];
    const byDiff = [...pts].sort((a, b) => a.x - a.y - (b.x - b.y));
    const bl = byDiff[0];
    const tr = byDiff[3];
    return [tl, tr, br, bl];
  }

  function rotatedRectCorners(rect) {
    const { center, size, angle } = rect;
    const rad = (angle * Math.PI) / 180;
    const cos = Math.cos(rad);
    const sin = Math.sin(rad);
    const hw = size.width / 2;
    const hh = size.height / 2;
    const local = [
      { x: -hw, y: -hh },
      { x: hw, y: -hh },
      { x: hw, y: hh },
      { x: -hw, y: hh },
    ];
    return local.map((p) => ({
      x: center.x + p.x * cos - p.y * sin,
      y: center.y + p.x * sin + p.y * cos,
    }));
  }

  // Finds the board's blue frame in `srcMat` (RGBA cv.Mat).
  // Returns { found, corners (TL,TR,BR,BL in source-image coords), mask (cv.Mat, caller must delete) }.
  function findBoardCorners(cv, srcMat) {
    const hsv = new cv.Mat();
    cv.cvtColor(srcMat, hsv, cv.COLOR_RGBA2RGB);
    cv.cvtColor(hsv, hsv, cv.COLOR_RGB2HSV);

    const low = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [
      settings.blueHueMin,
      settings.blueSatMin,
      settings.blueValMin,
      0,
    ]);
    const high = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [
      settings.blueHueMax,
      255,
      255,
      255,
    ]);
    const mask = new cv.Mat();
    cv.inRange(hsv, low, high, mask);
    low.delete();
    high.delete();

    const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(5, 5));
    cv.morphologyEx(mask, mask, cv.MORPH_CLOSE, kernel);
    cv.morphologyEx(mask, mask, cv.MORPH_OPEN, kernel);
    kernel.delete();

    const contours = new cv.MatVector();
    const hierarchy = new cv.Mat();
    cv.findContours(mask, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

    let bestIdx = -1;
    let bestArea = 0;
    const frameArea = srcMat.rows * srcMat.cols;
    for (let i = 0; i < contours.size(); i++) {
      const c = contours.get(i);
      const area = cv.contourArea(c);
      if (area > bestArea) {
        bestArea = area;
        bestIdx = i;
      }
      c.delete();
    }

    let result = { found: false, corners: null, mask };
    if (bestIdx >= 0 && bestArea / frameArea >= settings.minAreaFraction) {
      const c = contours.get(bestIdx);
      const rect = cv.minAreaRect(c);
      c.delete();
      const corners = orderCorners(rotatedRectCorners(rect));
      result = { found: true, corners, mask };
    }

    contours.delete();
    hierarchy.delete();
    hsv.delete();
    return result;
  }

  // Warps the quad defined by `corners` (TL,TR,BR,BL) to a flat WARP_W x WARP_H image.
  function warpBoard(cv, srcMat, corners) {
    const srcTri = cv.matFromArray(4, 1, cv.CV_32FC2, [
      corners[0].x,
      corners[0].y,
      corners[1].x,
      corners[1].y,
      corners[2].x,
      corners[2].y,
      corners[3].x,
      corners[3].y,
    ]);
    const dstTri = cv.matFromArray(4, 1, cv.CV_32FC2, [
      0, 0,
      WARP_W, 0,
      WARP_W, WARP_H,
      0, WARP_H,
    ]);
    const M = cv.getPerspectiveTransform(srcTri, dstTri);
    const warped = new cv.Mat();
    cv.warpPerspective(srcMat, warped, M, new cv.Size(WARP_W, WARP_H));
    srcTri.delete();
    dstTri.delete();
    M.delete();
    return warped;
  }

  function classifyHSV(h, s, v) {
    const isRed =
      ((h <= settings.redHueMax1 || h >= settings.redHueMin2) &&
        s >= settings.redSatMin &&
        v >= settings.redValMin);
    if (isRed) return "red";

    const isGreen =
      h >= settings.greenHueMin &&
      h <= settings.greenHueMax &&
      s >= settings.greenSatMin &&
      v >= settings.greenValMin;
    if (isGreen) return "green";

    return "empty";
  }

  // Classifies one sampled cell, preferring the calibrated empty baseline
  // (if any) over hue-range matching so a lighting color cast that pushes
  // white/cream holes into the red/green hue range doesn't misread them.
  function classifyCell(h, s, v, r, c) {
    if (emptyReference) {
      const ref = emptyReference[r][c];
      if (colorDelta({ h, s, v }, ref) <= settings.emptyMaxDist) return "empty";
    }
    return classifyHSV(h, s, v);
  }

  // Samples the 42 slot centers of a warped board image.
  // Returns { grid, colors } where grid[row][col] (row 0 = top of image) is
  // 'empty' | 'red' | 'green', and colors[row][col] is the averaged {h,s,v}.
  function sampleGrid(cv, warpedMat) {
    const hsv = new cv.Mat();
    cv.cvtColor(warpedMat, hsv, cv.COLOR_RGBA2RGB);
    cv.cvtColor(hsv, hsv, cv.COLOR_RGB2HSV);

    const grid = [];
    const colors = [];
    const sampleHalf = Math.floor(CELL_PX * 0.28);

    for (let r = 0; r < ROWS; r++) {
      const gridRow = [];
      const colorRow = [];
      for (let c = 0; c < COLS; c++) {
        const cx = Math.round(c * CELL_PX + CELL_PX / 2);
        const cy = Math.round(r * CELL_PX + CELL_PX / 2);
        const x0 = Math.max(0, cx - sampleHalf);
        const y0 = Math.max(0, cy - sampleHalf);
        const w = Math.min(WARP_W - x0, sampleHalf * 2);
        const h = Math.min(WARP_H - y0, sampleHalf * 2);
        const rect = new cv.Rect(x0, y0, w, h);
        const roi = hsv.roi(rect);
        const mean = cv.mean(roi);
        roi.delete();

        const [hh, ss, vv] = mean;
        gridRow.push(classifyCell(hh, ss, vv, r, c));
        colorRow.push({ h: hh, s: ss, v: vv });
      }
      grid.push(gridRow);
      colors.push(colorRow);
    }

    hsv.delete();
    lastColors = colors;
    return { grid, colors };
  }

  // Full pipeline for one frame. `srcMat` is an RGBA cv.Mat of the camera frame.
  // Returns { found, corners, grid, warpedDataUrl } — grid is image-row-major
  // (row 0 = top of the physical board) and already flip-adjusted per settings.
  function processFrame(cv, srcMat, opts = {}) {
    const { corners, found, mask } = findBoardCorners(cv, srcMat);
    if (opts.debugMaskCanvas && mask) {
      cv.imshow(opts.debugMaskCanvas, mask);
    }
    mask.delete();

    if (!found) {
      return { found: false, corners: null, grid: null };
    }

    let orderedCorners = corners;
    if (settings.flipH) {
      orderedCorners = [orderedCorners[1], orderedCorners[0], orderedCorners[3], orderedCorners[2]];
    }
    if (settings.flipV) {
      orderedCorners = [orderedCorners[3], orderedCorners[2], orderedCorners[1], orderedCorners[0]];
    }

    const warped = warpBoard(cv, srcMat, orderedCorners);
    const { grid, colors } = sampleGrid(cv, warped);

    if (opts.warpedCanvas) {
      cv.imshow(opts.warpedCanvas, warped);
    }
    warped.delete();

    return { found: true, corners, grid, colors };
  }

  // Renders a grid + its raw {h,s,v} samples as a compact multi-line string,
  // one row per physical board row, e.g. "r0: e(12,20,180) r(8,64,48) …" —
  // meant for console/devlog output when debugging misclassifications.
  function formatGridDebug(grid, colors) {
    const lines = [];
    for (let r = 0; r < grid.length; r++) {
      const parts = [];
      for (let c = 0; c < grid[r].length; c++) {
        const { h, s, v } = colors[r][c];
        parts.push(`${grid[r][c][0]}(${h.toFixed(0)},${s.toFixed(0)},${v.toFixed(0)})`);
      }
      lines.push(`r${r}: ${parts.join(" ")}`);
    }
    return lines.join("\n");
  }

  return {
    ROWS,
    COLS,
    CELL_PX,
    settings,
    findBoardCorners,
    warpBoard,
    sampleGrid,
    classifyHSV,
    processFrame,
    calibrateEmpty,
    clearEmptyCalibration,
    hasEmptyCalibration,
    debugCell,
    formatGridDebug,
  };
})();
