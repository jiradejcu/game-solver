// Computer-vision board reader built on OpenCV.js.
//
// Pipeline per frame:
//   1. Threshold the frame for the board's blue frame color -> mask, and
//      find the largest contour in it (the frame + its base, which merge
//      into one shape since they're the same color and touch).
//   2. Find the grid's four corners:
//      - Top/left/right: the single sharpest black<->white transition in
//        the mask near each edge (findTopEdgeY/findLeftEdgeX/findRightEdgeX)
//        -- a direct measurement of where the mask actually changes, not a
//        fit through ambiguous candidates (an earlier version tried Hough
//        line detection here and kept latching onto false lines -- a
//        diagonal line formed by repeated holes' edges, background clutter
//        picked up by an over-generous search margin, etc.).
//      - Bottom: there's no sharp transition to find here -- the base is
//        the same blue plastic as the frame with nothing marking where the
//        grid ends and the base begins. Instead, detectHoleCircles finds
//        the 42 holes directly via the Hough circle transform, and
//        estimateBottomYFromCircles measures the row-to-row spacing from
//        however many rows it can confidently see and extrapolates one
//        more row past the last one.
//   3. Perspective-warp the grid to a flat top-down 7x6 image.
//   4. Sample the average HSV color at each of the 42 slot centers and
//      classify it as empty / red / green.
//
// All thresholds are tunable at runtime (see Vision.settings) because the
// exact color response depends on the camera and lighting.

const Vision = (() => {
  const ROWS = 6;
  const COLS = 7;
  // Size of one warped cell, in pixels. NOT equal on both axes -- the
  // physical board's holes aren't spaced in perfect squares (a snapshot of
  // the warped-board debug preview showed circular holes rendering as
  // visibly tall ovals when this was a single square CELL_PX, i.e. the row
  // spacing is tighter than the column spacing).
  const CELL_W = 60;
  const CELL_H = 48;
  const WARP_W = COLS * CELL_W;
  const WARP_H = ROWS * CELL_H;

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
  // Per-cell {redFrac,greenFrac} from the most recent sampleGrid call (see
  // classifyCell) -- used by debugCell so the HSV-inspector click UI shows
  // the same classification sampleGrid actually used, not an approximation.
  let lastFractions = null;

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
    const { redFrac, greenFrac } = lastFractions[r][c];
    return { h, s, v, redFrac, greenFrac, classification: classifyCell(h, s, v, r, c, redFrac, greenFrac) };
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

  // Detects hole-like circles via the Hough circle transform -- used by
  // estimateBottomYFromCircles to measure the grid's bottom edge, since a
  // snapshot (see js/app.js's captureSnapshot) confirmed these land
  // precisely on real holes even where the base's own texture/notches would
  // fool a simpler per-pixel approach. `bbox` (the detected blue contour's
  // bounding box) only scales the search parameters; pass null to search
  // the whole frame with generous defaults.
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

  // Fraction of mask pixels that are white (blue) in each row y in [y0,y1),
  // sampled across columns [x0,x1). One number per row -- a 1D brightness
  // profile you can scan for where it transitions from mostly-black to
  // mostly-white (or back).
  function maskRowProfile(mask, x0, x1, y0, y1) {
    const data = mask.data;
    const cols = mask.cols;
    const w = x1 - x0;
    const profile = [];
    for (let y = y0; y < y1; y++) {
      let count = 0;
      const rowOffset = y * cols;
      for (let x = x0; x < x1; x++) {
        if (data[rowOffset + x] > 127) count++;
      }
      profile.push(count / w);
    }
    return profile;
  }

  // Same as maskRowProfile but per-column instead of per-row.
  function maskColProfile(mask, y0, y1, x0, x1) {
    const data = mask.data;
    const cols = mask.cols;
    const h = y1 - y0;
    const profile = [];
    for (let x = x0; x < x1; x++) {
      let count = 0;
      for (let y = y0; y < y1; y++) {
        if (data[y * cols + x] > 127) count++;
      }
      profile.push(count / h);
    }
    return profile;
  }

  const MIN_EDGE_CONTRAST = 0.3;

  // Index of the steepest black->white rise in `profile` (comparing samples
  // `r` apart, to smooth single-pixel noise). Returns null if nothing rises
  // by at least MIN_EDGE_CONTRAST -- i.e. no real edge in this profile.
  function steepestRise(profile, r) {
    let bestIdx = -1;
    let bestRise = MIN_EDGE_CONTRAST;
    for (let i = r; i < profile.length - r; i++) {
      const rise = profile[i + r] - profile[i - r];
      if (rise > bestRise) {
        bestRise = rise;
        bestIdx = i;
      }
    }
    return bestIdx === -1 ? null : bestIdx;
  }

  // Same as steepestRise but for the steepest white->black fall.
  function steepestFall(profile, r) {
    let bestIdx = -1;
    let bestFall = MIN_EDGE_CONTRAST;
    for (let i = r; i < profile.length - r; i++) {
      const fall = profile[i - r] - profile[i + r];
      if (fall > bestFall) {
        bestFall = fall;
        bestIdx = i;
      }
    }
    return bestIdx === -1 ? null : bestIdx;
  }

  // Finds the board's top/left/right edges as the single sharpest
  // black<->white transition in the mask near each one -- not a Hough line
  // fit through ambiguous candidates, just: where does the mask actually,
  // directly change from background to frame? (Bottom isn't found this way
  // -- the base is the same blue plastic with no sharp transition where it
  // meets the grid, so there's nothing for this to find there; see
  // estimateBottomYFromCircles.) Each function samples a window near the
  // expected edge (using `bbox`, the blue contour's bounding box, as a
  // rough starting reference) and returns the transition's coordinate in
  // full-mask space, or null if no clear transition was found there.
  function findTopEdgeY(mask, bbox) {
    const x0 = Math.round(bbox.x + bbox.width * 0.2);
    const x1 = Math.round(bbox.x + bbox.width * 0.8);
    const y0 = Math.max(0, Math.round(bbox.y - bbox.height * 0.15));
    const y1 = Math.min(mask.rows, Math.round(bbox.y + bbox.height * 0.4));
    if (x1 <= x0 || y1 <= y0) return null;
    const idx = steepestRise(maskRowProfile(mask, x0, x1, y0, y1), 2);
    return idx === null ? null : y0 + idx;
  }

  function findLeftEdgeX(mask, bbox) {
    const y0 = Math.round(bbox.y + bbox.height * 0.2);
    const y1 = Math.round(bbox.y + bbox.height * 0.8);
    const x0 = Math.max(0, Math.round(bbox.x - bbox.width * 0.15));
    const x1 = Math.min(mask.cols, Math.round(bbox.x + bbox.width * 0.4));
    if (x1 <= x0 || y1 <= y0) return null;
    const idx = steepestRise(maskColProfile(mask, y0, y1, x0, x1), 2);
    return idx === null ? null : x0 + idx;
  }

  function findRightEdgeX(mask, bbox) {
    const y0 = Math.round(bbox.y + bbox.height * 0.2);
    const y1 = Math.round(bbox.y + bbox.height * 0.8);
    const x0 = Math.max(0, Math.round(bbox.x + bbox.width * 0.6));
    const x1 = Math.min(mask.cols, Math.round(bbox.x + bbox.width * 1.15));
    if (x1 <= x0 || y1 <= y0) return null;
    const idx = steepestFall(maskColProfile(mask, y0, y1, x0, x1), 2);
    return idx === null ? null : x0 + idx;
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

  // Estimates the y-coordinate of the grid's bottom edge from detected
  // hole-circles (see detectHoleCircles). Top/left/right are found directly
  // as a sharp mask transition (see findTopEdgeY etc.), but the bottom has
  // no such transition to find -- the base is the same blue plastic as the
  // frame with nothing sharp where it meets the grid -- so this instead
  // measures the row-to-row spacing from however many hole-rows it can
  // confidently see and extrapolates one more row's worth past the last one.
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
  // Returns null if too few real rows were measured to be confident. If
  // `debug` is passed, records diagnostics for why this succeeded/failed
  // (see Vision.processFrame's debugHough option).
  function estimateBottomYFromCircles(circles, bbox, debug) {
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

    const yFit = linearFit(yCenters, rowIndices);
    if (!yFit || !(yFit.slope > 2)) return null;
    if (debug) debug.period = yFit.slope;

    return yFit.at(ROWS - 0.5);
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
      bbox = cv.boundingRect(c);
      c.delete();

      // Top/left/right: the single sharpest black<->white transition in the
      // mask near each edge (see findTopEdgeY etc.) -- a direct measurement,
      // not a fit through ambiguous candidates. Bottom: extrapolated from
      // detected hole-circles' row spacing (estimateBottomYFromCircles),
      // since the base is the same blue plastic as the frame with no sharp
      // transition where it meets the grid for a direct measurement to find.
      const topY = findTopEdgeY(mask, bbox);
      const leftX = findLeftEdgeX(mask, bbox);
      const rightX = findRightEdgeX(mask, bbox);
      const circles = detectHoleCircles(cv, srcMat, bbox);
      const bottomY = estimateBottomYFromCircles(circles, bbox, debugHough);

      if (debugHough) {
        debugHough.topY = topY;
        debugHough.leftX = leftX;
        debugHough.rightX = rightX;
        debugHough.bottomY = bottomY;
        debugHough.circles = circles;
        debugHough.cornerMethod =
          topY != null && leftX != null && rightX != null && bottomY != null ? "mask-edges+circles-bottom" : "none";
      }

      if (topY != null && leftX != null && rightX != null && bottomY != null) {
        const corners = [
          { x: leftX, y: topY },
          { x: rightX, y: topY },
          { x: rightX, y: bottomY },
          { x: leftX, y: bottomY },
        ];
        result = { found: true, corners: insetQuad(corners, settings.cornerInset), mask };
      }
      // No fallback to a cruder guess when a measurement is missing --
      // "found: false" here is honest uncertainty, not a bug to paper over.
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

  // Minimum fraction of a cell's sample region that must match a color's
  // per-pixel mask (computeRangeMask/computeRedMask) to classify it as that
  // color. Cells are classified by per-pixel mask coverage, not by
  // classifying the region's *mean* HSV (the old approach) -- averaging is
  // unreliable because hue is meaningless noise for low-saturation pixels
  // (shadows, glare off an empty hole's rim), and a region that's mostly
  // gray with a few random-hued noisy pixels can average to a hue that
  // passes a color's range even though literally no individual pixel in it
  // does. Confirmed via a snapshot: a cell reading g(41,61,76) had a
  // completely black (zero-coverage) green mask at that exact location.
  const COLOR_COVERAGE_THRESHOLD = 0.35;

  // Classifies one sampled cell. Prefers the calibrated empty baseline (if
  // any) over color coverage, so a lighting color cast that pushes
  // white/cream holes into the red/green hue range doesn't misread them.
  // Otherwise picks whichever of red/green covers enough of the cell's
  // pixels (see COLOR_COVERAGE_THRESHOLD), or empty if neither does.
  function classifyCell(h, s, v, r, c, redFrac, greenFrac) {
    if (emptyReference) {
      const ref = emptyReference[r][c];
      if (colorDelta({ h, s, v }, ref) <= settings.emptyMaxDist) return "empty";
    }
    if (redFrac >= COLOR_COVERAGE_THRESHOLD && redFrac >= greenFrac) return "red";
    if (greenFrac >= COLOR_COVERAGE_THRESHOLD) return "green";
    return "empty";
  }

  // Samples the 42 slot centers of a warped board image.
  // Returns { grid, colors } where grid[row][col] (row 0 = top of image) is
  // 'empty' | 'red' | 'green', and colors[row][col] is the averaged {h,s,v}
  // (still reported for the HSV-inspector UI and empty-baseline calibration,
  // but no longer what classification itself is based on -- see
  // classifyCell).
  function sampleGrid(cv, warpedMat) {
    const hsv = new cv.Mat();
    cv.cvtColor(warpedMat, hsv, cv.COLOR_RGBA2RGB);
    cv.cvtColor(hsv, hsv, cv.COLOR_RGB2HSV);

    // Per-pixel color masks on the warped image, so each cell can be
    // classified by how much of its sample region actually matches a
    // color -- the same reliable per-pixel test the mask debug previews
    // use -- instead of by the region's mean HSV.
    const redMask = computeRedMask(cv, hsv);
    const greenMask = computeRangeMask(
      cv, hsv, settings.greenHueMin, settings.greenHueMax, settings.greenSatMin, settings.greenValMin
    );

    const grid = [];
    const colors = [];
    const fractions = [];
    const sampleHalfW = Math.floor(CELL_W * 0.28);
    const sampleHalfH = Math.floor(CELL_H * 0.28);

    for (let r = 0; r < ROWS; r++) {
      const gridRow = [];
      const colorRow = [];
      const fractionRow = [];
      for (let c = 0; c < COLS; c++) {
        const cx = Math.round(c * CELL_W + CELL_W / 2);
        const cy = Math.round(r * CELL_H + CELL_H / 2);
        const x0 = Math.max(0, cx - sampleHalfW);
        const y0 = Math.max(0, cy - sampleHalfH);
        const w = Math.min(WARP_W - x0, sampleHalfW * 2);
        const h = Math.min(WARP_H - y0, sampleHalfH * 2);
        const rect = new cv.Rect(x0, y0, w, h);

        const roi = hsv.roi(rect);
        const mean = cv.mean(roi);
        roi.delete();

        const area = w * h;
        const redRoi = redMask.roi(rect);
        const redFrac = area > 0 ? cv.countNonZero(redRoi) / area : 0;
        redRoi.delete();
        const greenRoi = greenMask.roi(rect);
        const greenFrac = area > 0 ? cv.countNonZero(greenRoi) / area : 0;
        greenRoi.delete();

        const [hh, ss, vv] = mean;
        gridRow.push(classifyCell(hh, ss, vv, r, c, redFrac, greenFrac));
        colorRow.push({ h: hh, s: ss, v: vv });
        fractionRow.push({ redFrac, greenFrac });
      }
      grid.push(gridRow);
      colors.push(colorRow);
      fractions.push(fractionRow);
    }

    redMask.delete();
    greenMask.delete();
    hsv.delete();
    lastColors = colors;
    lastFractions = fractions;
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
        pts.push({ x: c * CELL_W + CELL_W / 2, y: r * CELL_H + CELL_H / 2 });
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
    CELL_W,
    CELL_H,
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
