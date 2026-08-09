// Computer-vision board reader built on OpenCV.js.
//
// Pipeline per frame:
//   1. Threshold the frame for the board's blue frame color -> mask.
//   2. Find the largest contour in that mask and take its minAreaRect for
//      the top edge, left/right width, and rotation. The bottom edge is NOT
//      trusted from the blob directly -- the board's base is the same blue
//      plastic and merges into the same contour, with no holes in it, so
//      the blob's actual bottom edge includes the base. Instead it's
//      measured directly: scan the mask row by row and find where the
//      periodic "hole, gap, hole, gap..." pattern stops (see
//      findGridBottomByHoles / cornersFromMaskHoles). Falls back to the
//      blob's own bottom edge if no clear hole pattern is found.
//      collectHoughDebug separately exposes the *raw* Hough line/circle
//      output (see Vision.processFrame's debugHough option) purely for
//      visual calibration checking, with no attempt to pick or fit anything
//      from it.
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
    // Fraction of the detected blue rect's width/height to pull each edge
    // inward before warping, so the frame's own plastic border isn't
    // stretched into the grid and sampled as if it were a slot.
    cornerInset: 0.04,
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

  // Shrinks a [TL,TR,BR,BL] quad toward its own centroid by `frac` (e.g.
  // 0.04 = each corner moves 4% of the way toward the center), to trim the
  // frame's own plastic border off the warp instead of stretching it into
  // the grid.
  function insetQuad(corners, frac) {
    const cx = corners.reduce((s, p) => s + p.x, 0) / corners.length;
    const cy = corners.reduce((s, p) => s + p.y, 0) / corners.length;
    return corners.map((p) => ({
      x: cx + (p.x - cx) * (1 - frac),
      y: cy + (p.y - cy) * (1 - frac),
    }));
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

  // Detects hole-like circles via the Hough circle transform. Unlike the
  // line detection below, this isn't just for visualization -- a snapshot
  // (see js/app.js's captureSnapshot) confirmed these land precisely on
  // real holes even where a raw per-pixel mask-row scan gets fooled by the
  // base's own texture/notches, so findGridBottomByCircles uses this
  // directly to measure the grid's actual extent. `bbox` (the detected blue
  // contour's bounding box) only scales the search parameters; pass null to
  // search the whole frame with generous defaults.
  function detectHoleCircles(cv, srcMat, bbox) {
    const gray = new cv.Mat();
    cv.cvtColor(srcMat, gray, cv.COLOR_RGBA2GRAY);
    cv.GaussianBlur(gray, gray, new cv.Size(5, 5), 0);

    const cellPx = bbox ? bbox.width / COLS : Math.min(srcMat.cols, srcMat.rows) / 10;
    const circles = new cv.Mat();
    cv.HoughCircles(
      gray, circles, cv.HOUGH_GRADIENT, 1,
      Math.max(8, cellPx * 0.6), // min distance between circle centers
      100, 30, // Canny high threshold, accumulator threshold
      Math.max(4, Math.round(cellPx * 0.22)), Math.max(8, Math.round(cellPx * 0.55)) // min/max radius
    );
    gray.delete();

    const result = [];
    for (let i = 0; i < circles.cols; i++) {
      result.push({ x: circles.data32F[i * 3], y: circles.data32F[i * 3 + 1], r: circles.data32F[i * 3 + 2] });
    }
    circles.delete();
    return result;
  }

  // Runs Hough line detection and records every raw candidate into `debug`,
  // with no picking or fitting -- purely for visual inspection (see
  // Vision.processFrame's debugHough option), e.g. to check that the
  // calibrated blue/red/green ranges actually land on the right shapes.
  // `bbox` only scales the search parameters; pass null to search the whole
  // frame with generous defaults.
  function collectHoughLineDebug(cv, mask, bbox, debug) {
    const margin = bbox ? Math.round(Math.max(bbox.width, bbox.height) * 0.1) : 0;
    const rx = bbox ? Math.max(0, bbox.x - margin) : 0;
    const ry = bbox ? Math.max(0, bbox.y - margin) : 0;
    const rw = bbox ? Math.min(mask.cols - rx, bbox.width + margin * 2) : mask.cols;
    const rh = bbox ? Math.min(mask.rows - ry, bbox.height + margin * 2) : mask.rows;
    const roi = mask.roi(new cv.Rect(rx, ry, rw, rh));

    const edges = new cv.Mat();
    cv.Canny(roi, edges, 50, 150);
    // Loose on purpose: this is a raw visualization, not a fit, so showing
    // extra false candidates is far cheaper than showing none. A tighter
    // minLen (scaled off bbox) was finding zero lines on tilted/occluded
    // frames even though the mask clearly had a real edge, because no single
    // unbroken segment reached the threshold.
    const minLen = bbox ? Math.max(15, Math.min(bbox.width, bbox.height) * 0.15) : 15;
    const lines = new cv.Mat();
    cv.HoughLinesP(edges, lines, 1, Math.PI / 180, 25, minLen, 25);

    const horizontals = [];
    const verticals = [];
    for (let i = 0; i < lines.rows; i++) {
      const x1 = lines.data32S[i * 4] + rx;
      const y1 = lines.data32S[i * 4 + 1] + ry;
      const x2 = lines.data32S[i * 4 + 2] + rx;
      const y2 = lines.data32S[i * 4 + 3] + ry;
      let angle = (Math.atan2(y2 - y1, x2 - x1) * 180) / Math.PI;
      angle = ((angle % 180) + 180) % 180; // normalize to [0,180)
      const seg = { x1, y1, x2, y2 };
      if (angle <= 20 || angle >= 160) horizontals.push(seg);
      else if (angle >= 70 && angle <= 110) verticals.push(seg);
    }
    edges.delete();
    lines.delete();
    roi.delete();

    debug.horizontals = horizontals;
    debug.verticals = verticals;
  }

  // Binary mask (CV_8U, same size as `hsv`) of pixels whose hue falls in
  // [hueMin, hueMax] and whose sat/val clear the given minimums. This is
  // exactly what calibrating a color sets (see app.js's color-calibration
  // click handler) -- rendering it full-frame (see Vision.processFrame's
  // debugRedCanvas/debugGreenCanvas/debugMaskCanvas options) is the most
  // direct way to check a calibration actually captured the right range.
  function computeRangeMask(cv, hsv, hueMin, hueMax, satMin, valMin) {
    const low = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [hueMin, satMin, valMin, 0]);
    const high = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [hueMax, 255, 255, 255]);
    const mask = new cv.Mat();
    cv.inRange(hsv, low, high, mask);
    low.delete();
    high.delete();
    return mask;
  }

  // Red wraps around hue 0/180 in OpenCV's 0-179 hue space (see
  // settings.redHueMax1/redHueMin2 and classifyHSV), so its mask is the OR
  // of the two ranges on either side of the wrap.
  function computeRedMask(cv, hsv) {
    const mask1 = computeRangeMask(cv, hsv, 0, settings.redHueMax1, settings.redSatMin, settings.redValMin);
    const mask2 = computeRangeMask(cv, hsv, settings.redHueMin2, 179, settings.redSatMin, settings.redValMin);
    const mask = new cv.Mat();
    cv.bitwise_or(mask1, mask2, mask);
    mask1.delete();
    mask2.delete();
    return mask;
  }

  // Least-squares fit of values[k] = intercept + slope*indices[k]. Returns
  // null if degenerate (indices don't vary).
  function linearFit(values, indices) {
    const n = values.length;
    let sumI = 0, sumV = 0, sumII = 0, sumIV = 0;
    for (let k = 0; k < n; k++) {
      const i = indices[k];
      sumI += i;
      sumV += values[k];
      sumII += i * i;
      sumIV += i * values[k];
    }
    const denom = n * sumII - sumI * sumI;
    if (denom === 0) return null;
    const slope = (n * sumIV - sumI * sumV) / denom;
    const intercept = (sumV - slope * sumI) / n;
    return { slope, intercept, at: (i) => intercept + slope * i };
  }

  // A real row can be entirely missing from `realRows` for just one frame
  // (e.g. occlusion drops its circle count below the "confident row"
  // threshold) -- naively treating the remaining rows as consecutive
  // indices 0,1,2... would then misattribute a 2-row gap as a 1-row one and
  // badly bias linearFit's slope. Estimates a period from the median
  // consecutive gap in `yCenters` (robust to at most one skipped row) and
  // uses it to infer each row's true index. Returns null if the inferred
  // indices don't make sense (e.g. overflow past the known row count).
  function inferRowIndices(yCenters) {
    const gaps = [];
    for (let i = 1; i < yCenters.length; i++) gaps.push(yCenters[i] - yCenters[i - 1]);
    const sortedGaps = [...gaps].sort((a, b) => a - b);
    const medianGap = sortedGaps[Math.floor(sortedGaps.length / 2)];
    if (!(medianGap > 2)) return null;

    const indices = [0];
    for (let i = 1; i < yCenters.length; i++) {
      const steps = Math.max(1, Math.round((yCenters[i] - yCenters[i - 1]) / medianGap));
      indices.push(indices[i - 1] + steps);
    }
    if (indices[indices.length - 1] > ROWS - 1) return null;
    return indices;
  }

  // Computes [TL,TR,BR,BL] directly from detected hole-circles (see
  // detectHoleCircles), instead of from `rect` (minAreaRect of the whole
  // blue blob, grid+base) -- earlier versions of this only re-measured the
  // BOTTOM edge from circles while still projecting BL/BR's x-position from
  // rect's rotation, which the base (wider than the grid, and not
  // necessarily symmetric -- e.g. a notch on only one side) can skew,
  // producing believable-looking but wrong left/right edges. This measures
  // all four edges from the same source instead of mixing two.
  //
  // Clusters circles into rows across the WHOLE bbox, but only ever trusts
  // the first ROWS clusters from the top -- that's a hard, known fact about
  // this board, so anything clustered below that (the base, whatever
  // texture it has) is provably not a real row and is simply never looked
  // at, rather than needing to guess a "safe" fraction of the bbox to
  // search within (tried that both narrow -- not enough rows measured,
  // amplifying error through extrapolation -- and wide -- picked up real
  // circles the base's texture produces and got unstable).
  //
  // For each trusted row, fits how its y-center, leftmost-circle-x, and
  // rightmost-circle-x each vary with row index (a line per quantity) --
  // capturing whatever rotation/perspective tilt is really there instead of
  // assuming the sides are vertical -- then evaluates those lines at
  // row -0.5 (top edge) and row ROWS-0.5 (bottom edge), stepping out by
  // half a cell past the leftmost/rightmost/top/bottom-most hole *centers*
  // to reach the frame's actual inner edge.
  //
  // Returns null if too few real rows were measured to be confident. If
  // `debug` is passed, records diagnostics for why this succeeded/failed
  // (see Vision.processFrame's debugHough option).
  function cornersFromCircleGrid(circles, bbox, debug) {
    if (debug) {
      debug.bboxTop = bbox ? bbox.y : null;
      debug.bboxBottom = bbox ? bbox.y + bbox.height : null;
    }
    if (!bbox) return null;
    const cellPx = bbox.width / COLS;

    const inBbox = circles.filter((c) => c.y >= bbox.y && c.y < bbox.y + bbox.height);
    if (inBbox.length < 2) return null;

    // Gap-based row clustering: sort by y, split wherever the gap to the
    // next circle is more than half a cell (i.e. clearly a different row,
    // not just two circles in the same row at different x). Rows with only
    // a single circle are dropped as likely noise rather than a real,
    // mostly-occluded row.
    const sorted = [...inBbox].sort((a, b) => a.y - b.y);
    const rowGroups = [[sorted[0]]];
    for (let i = 1; i < sorted.length; i++) {
      if (sorted[i].y - sorted[i - 1].y > cellPx * 0.5) rowGroups.push([]);
      rowGroups[rowGroups.length - 1].push(sorted[i]);
    }
    const realRows = rowGroups.filter((r) => r.length >= 2).slice(0, ROWS); // never trust more than the known row count
    if (debug) debug.rowCounts = rowGroups.map((r) => r.length);
    if (realRows.length < 2) return null;

    const yCenters = realRows.map((r) => r.reduce((s, c) => s + c.y, 0) / r.length);
    const rowIndices = inferRowIndices(yCenters);
    if (debug) debug.allRowCenters = yCenters;
    if (!rowIndices) return null;

    const leftVals = realRows.map((r) => Math.min(...r.map((c) => c.x)));
    const rightVals = realRows.map((r) => Math.max(...r.map((c) => c.x)));
    const yFit = linearFit(yCenters, rowIndices);
    const leftFit = linearFit(leftVals, rowIndices);
    const rightFit = linearFit(rightVals, rowIndices);
    if (!yFit || !leftFit || !rightFit || !(yFit.slope > 2)) return null;
    if (debug) debug.period = yFit.slope;

    const cellHalf = cellPx * 0.5;
    const topY = yFit.at(-0.5);
    const bottomY = yFit.at(ROWS - 0.5);
    const tl = { x: leftFit.at(-0.5) - cellHalf, y: topY };
    const tr = { x: rightFit.at(-0.5) + cellHalf, y: topY };
    const bl = { x: leftFit.at(ROWS - 0.5) - cellHalf, y: bottomY };
    const br = { x: rightFit.at(ROWS - 0.5) + cellHalf, y: bottomY };
    return [tl, tr, br, bl];
  }

  // Finds the board's blue frame in `srcMat` (RGBA cv.Mat).
  // Returns { found, corners (TL,TR,BR,BL in source-image coords), mask (cv.Mat, caller must delete) }.
  // `debugHough`, if passed, is filled with the raw Hough line/circle
  // candidates for visualization (see Vision.processFrame).
  function findBoardCorners(cv, srcMat, debugHough) {
    const hsv = new cv.Mat();
    cv.cvtColor(srcMat, hsv, cv.COLOR_RGBA2RGB);
    cv.cvtColor(hsv, hsv, cv.COLOR_RGB2HSV);

    const mask = computeRangeMask(cv, hsv, settings.blueHueMin, settings.blueHueMax, settings.blueSatMin, settings.blueValMin);

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
    let bbox = null;
    if (bestIdx >= 0 && bestArea / frameArea >= settings.minAreaFraction) {
      const c = contours.get(bestIdx);
      const rect = cv.minAreaRect(c);
      bbox = cv.boundingRect(c);
      c.delete();
      const circles = detectHoleCircles(cv, srcMat, bbox);
      const holeCorners = cornersFromCircleGrid(circles, bbox, debugHough);
      const corners = insetQuad(holeCorners || orderCorners(rotatedRectCorners(rect)), settings.cornerInset);
      if (debugHough) {
        debugHough.cornerMethod = holeCorners ? "circles" : "rect-fallback";
        debugHough.circles = circles;
      }
      result = { found: true, corners, mask };
    }

    // collectRaw (line detection) is opt-in (app.js only sets it when "Show
    // raw Hough candidates" is checked) since it's purely for visualization
    // and runs its own Canny/HoughLinesP pass -- real cost every frame.
    // Circle detection above always runs regardless, since corner-fitting
    // depends on it now, not just the debug view.
    if (debugHough && debugHough.collectRaw) collectHoughLineDebug(cv, mask, bbox, debugHough);

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
    const { corners, found, mask } = findBoardCorners(cv, srcMat, opts.debugHough);
    if (opts.debugMaskCanvas && mask) {
      cv.imshow(opts.debugMaskCanvas, mask);
    }
    mask.delete();

    if (opts.debugRedCanvas || opts.debugGreenCanvas) {
      const hsv = new cv.Mat();
      cv.cvtColor(srcMat, hsv, cv.COLOR_RGBA2RGB);
      cv.cvtColor(hsv, hsv, cv.COLOR_RGB2HSV);
      if (opts.debugRedCanvas) {
        const redMask = computeRedMask(cv, hsv);
        cv.imshow(opts.debugRedCanvas, redMask);
        redMask.delete();
      }
      if (opts.debugGreenCanvas) {
        const greenMask = computeRangeMask(
          cv, hsv, settings.greenHueMin, settings.greenHueMax, settings.greenSatMin, settings.greenValMin
        );
        cv.imshow(opts.debugGreenCanvas, greenMask);
        greenMask.delete();
      }
      hsv.delete();
    }

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

    return { found: true, corners, orderedCorners, grid, colors };
  }

  // Returns the WARP_W x WARP_H-space center point of each of the 42 grid
  // cells, in the same image-row-major order (row 0 = top) as sampleGrid's
  // `grid` output.
  function cellCenterPoints() {
    const pts = [];
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        pts.push({ x: c * CELL_PX + CELL_PX / 2, y: r * CELL_PX + CELL_PX / 2 });
      }
    }
    return pts;
  }

  // Maps warped-space points (e.g. from cellCenterPoints) back into the
  // source frame, using the inverse of the homography warpBoard applies.
  // Lets callers draw directly on the live camera image instead of only
  // the flattened board preview. `corners` must be the same
  // (flip-adjusted) corners passed to warpBoard — i.e. processFrame's
  // `orderedCorners`.
  function mapPointsToSource(cv, corners, points) {
    const dstTri = cv.matFromArray(4, 1, cv.CV_32FC2, [
      corners[0].x, corners[0].y,
      corners[1].x, corners[1].y,
      corners[2].x, corners[2].y,
      corners[3].x, corners[3].y,
    ]);
    const srcTri = cv.matFromArray(4, 1, cv.CV_32FC2, [0, 0, WARP_W, 0, WARP_W, WARP_H, 0, WARP_H]);
    const M = cv.getPerspectiveTransform(srcTri, dstTri);
    const srcPts = cv.matFromArray(points.length, 1, cv.CV_32FC2, points.flatMap((p) => [p.x, p.y]));
    const dstPts = new cv.Mat();
    cv.perspectiveTransform(srcPts, dstPts, M);
    const result = [];
    for (let i = 0; i < points.length; i++) {
      result.push({ x: dstPts.data32F[i * 2], y: dstPts.data32F[i * 2 + 1] });
    }
    srcTri.delete();
    dstTri.delete();
    M.delete();
    srcPts.delete();
    dstPts.delete();
    return result;
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
    cellCenterPoints,
    mapPointsToSource,
    calibrateEmpty,
    clearEmptyCalibration,
    hasEmptyCalibration,
    debugCell,
    formatGridDebug,
  };
})();
