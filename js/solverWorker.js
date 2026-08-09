// Runs the Connect-4 search off the main thread so the camera/vision loop
// never stutters while the engine is thinking.
importScripts("solver.js");

self.onmessage = (e) => {
  const { requestId, board, aiPlayer, timeBudgetMs } = e.data;
  try {
    const result = Connect4Solver.findBestMove(board, aiPlayer, timeBudgetMs);
    self.postMessage({ requestId, result });
  } catch (err) {
    self.postMessage({ requestId, error: String(err && err.message ? err.message : err) });
  }
};
