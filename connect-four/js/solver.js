// Connect-4 move solver: array-based minimax with alpha-beta pruning,
// iterative deepening, move ordering and a transposition table.
// Board convention: board[row][col], row 0 = bottom row, row ROWS-1 = top row.
// Cell values: 0 = empty, 1 = player 1, 2 = player 2.

const ROWS = 6;
const COLS = 7;
const CENTER_COL = 3;
const COLUMN_ORDER = [3, 2, 4, 1, 5, 0, 6];

const WIN_SCORE = 1000000;

function cloneBoard(board) {
  return board.map((row) => row.slice());
}

function getValidColumns(board) {
  const cols = [];
  for (const c of COLUMN_ORDER) {
    if (board[ROWS - 1][c] === 0) cols.push(c);
  }
  return cols;
}

function isValidColumn(board, col) {
  return board[ROWS - 1][col] === 0;
}

// Returns the row the piece would land on, or -1 if the column is full.
function getOpenRow(board, col) {
  for (let r = 0; r < ROWS; r++) {
    if (board[r][col] === 0) return r;
  }
  return -1;
}

function dropPiece(board, col, player) {
  const row = getOpenRow(board, col);
  board[row][col] = player;
  return row;
}

function undoDrop(board, col, row) {
  board[row][col] = 0;
}

function isBoardFull(board) {
  for (let c = 0; c < COLS; c++) {
    if (board[ROWS - 1][c] === 0) return false;
  }
  return true;
}

// Checks whether `player` has a 4-in-a-row that passes through (row, col).
// Cheaper than scanning the whole board after every move.
function checkWinAt(board, row, col, player) {
  const dirs = [
    [0, 1],
    [1, 0],
    [1, 1],
    [1, -1],
  ];
  for (const [dr, dc] of dirs) {
    let count = 1;
    let r = row + dr;
    let c = col + dc;
    while (r >= 0 && r < ROWS && c >= 0 && c < COLS && board[r][c] === player) {
      count++;
      r += dr;
      c += dc;
    }
    r = row - dr;
    c = col - dc;
    while (r >= 0 && r < ROWS && c >= 0 && c < COLS && board[r][c] === player) {
      count++;
      r -= dr;
      c -= dc;
    }
    if (count >= 4) return true;
  }
  return false;
}

function checkWinAnywhere(board, player) {
  // Horizontal
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c <= COLS - 4; c++) {
      if (
        board[r][c] === player &&
        board[r][c + 1] === player &&
        board[r][c + 2] === player &&
        board[r][c + 3] === player
      )
        return true;
    }
  }
  // Vertical
  for (let c = 0; c < COLS; c++) {
    for (let r = 0; r <= ROWS - 4; r++) {
      if (
        board[r][c] === player &&
        board[r + 1][c] === player &&
        board[r + 2][c] === player &&
        board[r + 3][c] === player
      )
        return true;
    }
  }
  // Diagonal /
  for (let r = 0; r <= ROWS - 4; r++) {
    for (let c = 0; c <= COLS - 4; c++) {
      if (
        board[r][c] === player &&
        board[r + 1][c + 1] === player &&
        board[r + 2][c + 2] === player &&
        board[r + 3][c + 3] === player
      )
        return true;
    }
  }
  // Diagonal \
  for (let r = 3; r < ROWS; r++) {
    for (let c = 0; c <= COLS - 4; c++) {
      if (
        board[r][c] === player &&
        board[r - 1][c + 1] === player &&
        board[r - 2][c + 2] === player &&
        board[r - 3][c + 3] === player
      )
        return true;
    }
  }
  return false;
}

function evaluateWindow(cells, player) {
  const opponent = player === 1 ? 2 : 1;
  const playerCount = cells.filter((v) => v === player).length;
  const emptyCount = cells.filter((v) => v === 0).length;
  const oppCount = cells.filter((v) => v === opponent).length;

  if (playerCount === 4) return 100000;
  if (playerCount === 3 && emptyCount === 1) return 100;
  if (playerCount === 2 && emptyCount === 2) return 10;
  if (oppCount === 3 && emptyCount === 1) return -120;
  if (oppCount === 2 && emptyCount === 2) return -8;
  return 0;
}

function scorePosition(board, player) {
  let score = 0;

  // Center column preference: pieces near the center take part in more
  // potential winning lines.
  for (let r = 0; r < ROWS; r++) {
    if (board[r][CENTER_COL] === player) score += 6;
  }

  // Horizontal windows
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c <= COLS - 4; c++) {
      const cells = [board[r][c], board[r][c + 1], board[r][c + 2], board[r][c + 3]];
      score += evaluateWindow(cells, player);
    }
  }
  // Vertical windows
  for (let c = 0; c < COLS; c++) {
    for (let r = 0; r <= ROWS - 4; r++) {
      const cells = [board[r][c], board[r + 1][c], board[r + 2][c], board[r + 3][c]];
      score += evaluateWindow(cells, player);
    }
  }
  // Diagonal / windows
  for (let r = 0; r <= ROWS - 4; r++) {
    for (let c = 0; c <= COLS - 4; c++) {
      const cells = [
        board[r][c],
        board[r + 1][c + 1],
        board[r + 2][c + 2],
        board[r + 3][c + 3],
      ];
      score += evaluateWindow(cells, player);
    }
  }
  // Diagonal \ windows
  for (let r = 3; r < ROWS; r++) {
    for (let c = 0; c <= COLS - 4; c++) {
      const cells = [
        board[r][c],
        board[r - 1][c + 1],
        board[r - 2][c + 2],
        board[r - 3][c + 3],
      ];
      score += evaluateWindow(cells, player);
    }
  }

  return score;
}

function boardKey(board, player, depth) {
  let s = "";
  for (let r = 0; r < ROWS; r++) for (let c = 0; c < COLS; c++) s += board[r][c];
  return s + "|" + player + "|" + depth;
}

class TimeUp extends Error {}

// depthRemaining plies of search left; player = whose turn it is at this node.
// aiPlayer = the player we are choosing the best move for (maximizer).
function negamax(board, depthRemaining, alpha, beta, player, movesPlayed, deadline, tt) {
  if (performance.now() > deadline) throw new TimeUp();

  const key = boardKey(board, player, depthRemaining);
  const cached = tt.get(key);
  if (cached) {
    if (cached.flag === "EXACT") return cached.score;
    if (cached.flag === "LOWER") alpha = Math.max(alpha, cached.score);
    else if (cached.flag === "UPPER") beta = Math.min(beta, cached.score);
    if (alpha >= beta) return cached.score;
  }

  const opponent = player === 1 ? 2 : 1;
  const validCols = getValidColumns(board);

  if (validCols.length === 0) return 0; // draw

  // Immediate win check for the player to move (cheap, and lets us "see"
  // one ply deeper than depthRemaining would otherwise allow).
  for (const c of validCols) {
    const row = getOpenRow(board, c);
    board[row][c] = player;
    const wins = checkWinAt(board, row, c, player);
    board[row][c] = 0;
    if (wins) {
      const score = WIN_SCORE - movesPlayed;
      tt.set(key, { score, flag: "EXACT" });
      return score;
    }
  }

  if (depthRemaining === 0) {
    const score = scorePosition(board, player) - scorePosition(board, opponent);
    return score;
  }

  // Order moves by a quick heuristic: prefer moves that create more threats.
  const scored = validCols.map((c) => {
    const row = getOpenRow(board, c);
    board[row][c] = player;
    const s = scorePosition(board, player);
    board[row][c] = 0;
    return { c, s };
  });
  scored.sort((a, b) => b.s - a.s);

  let best = -Infinity;
  let bestCol = scored[0].c;
  const origAlpha = alpha;

  for (const { c } of scored) {
    const row = dropPiece(board, c, player);
    let score;
    try {
      score = -negamax(board, depthRemaining - 1, -beta, -alpha, opponent, movesPlayed + 1, deadline, tt);
    } finally {
      undoDrop(board, c, row);
    }
    if (score > best) {
      best = score;
      bestCol = c;
    }
    if (best > alpha) alpha = best;
    if (alpha >= beta) break;
  }

  let flag = "EXACT";
  if (best <= origAlpha) flag = "UPPER";
  else if (best >= beta) flag = "LOWER";
  tt.set(key, { score: best, flag, bestCol });

  return best;
}

// Public entry point. Returns { column, score, depthReached, evaluation }.
// `evaluation` is from aiPlayer's perspective: positive = aiPlayer favored.
function findBestMove(board, aiPlayer, timeBudgetMs = 1200) {
  const validCols = getValidColumns(board);
  if (validCols.length === 0) return null;
  if (validCols.length === 1) {
    return { column: validCols[0], score: 0, depthReached: 0, evaluation: 0 };
  }

  const movesPlayed = board.flat().filter((v) => v !== 0).length;
  const opponent = aiPlayer === 1 ? 2 : 1;

  // Immediate win.
  for (const c of validCols) {
    const row = getOpenRow(board, c);
    board[row][c] = aiPlayer;
    const wins = checkWinAt(board, row, c, aiPlayer);
    board[row][c] = 0;
    if (wins) {
      return { column: c, score: WIN_SCORE, depthReached: 1, evaluation: WIN_SCORE, forced: "win" };
    }
  }

  // Must-block: opponent has exactly one immediate winning move.
  const opponentWinCols = [];
  for (const c of validCols) {
    const row = getOpenRow(board, c);
    board[row][c] = opponent;
    const wins = checkWinAt(board, row, c, opponent);
    board[row][c] = 0;
    if (wins) opponentWinCols.push(c);
  }

  const deadline = performance.now() + timeBudgetMs;
  const tt = new Map();

  let bestCol = opponentWinCols.length >= 1 ? opponentWinCols[0] : validCols[0];
  let bestScore = 0;
  let depthReached = 0;

  const maxDepth = Math.min(42 - movesPlayed, 22);

  try {
    for (let depth = 2; depth <= maxDepth; depth += 1) {
      let alpha = -Infinity;
      let beta = Infinity;
      let iterBestCol = null;
      let iterBestScore = -Infinity;

      // Order root moves: try previous best first.
      const ordered = [...validCols].sort((a, b) => (a === bestCol ? -1 : b === bestCol ? 1 : 0));

      for (const c of ordered) {
        const row = dropPiece(board, c, aiPlayer);
        let score;
        try {
          score = -negamax(board, depth - 1, -beta, -alpha, opponent, movesPlayed + 1, deadline, tt);
        } finally {
          undoDrop(board, c, row);
        }
        if (score > iterBestScore) {
          iterBestScore = score;
          iterBestCol = c;
        }
        if (score > alpha) alpha = score;
      }

      bestCol = iterBestCol;
      bestScore = iterBestScore;
      depthReached = depth;

      // Found a forced win/loss line — no need to search deeper.
      if (Math.abs(bestScore) > WIN_SCORE - 100) break;
    }
  } catch (e) {
    if (!(e instanceof TimeUp)) throw e;
    // Use whatever the last fully-completed iteration found.
  }

  return { column: bestCol, score: bestScore, depthReached, evaluation: bestScore };
}

const Connect4Solver = {
  ROWS,
  COLS,
  cloneBoard,
  getValidColumns,
  isValidColumn,
  getOpenRow,
  dropPiece,
  checkWinAt,
  checkWinAnywhere,
  isBoardFull,
  findBestMove,
};

if (typeof module !== "undefined" && module.exports) {
  module.exports = Connect4Solver;
}
