"""Greedy + Local‑Swap table subset selection
------------------------------------------------
A faster search method, acts as a scalable replacement for ilp.py that follows the
Greedy‑then‑Local‑Swap heuristic.

Signals used
~~~~~~~~~~~~
* **Coarse relevance** (`R_q(t)`) – 1‑per‑table similarity from the first‑stage
  retriever (Contriever/TAPAS). 
* **Fine‑grained coverage** (`F`) – max column‑level similarity per
  decomposed sub‑question (Ni × M matrix).
* **Join compatibility** (`J(i,j)`) – pre‑computed table‑table compatibility
  (max value of the CR tensor for the pair).

No unionability or attribute‑coverage yet – those can be added later by
extending `delta_score()`.

The script keeps the CLI and helper functions consistent with the rest of the
repository so that *compare.py*, *metrics.py*, etc. continue to work.

Usage example
-------------
```bash
python greedy.py --dataset spider --model contriever \
                 --topk 20 --K 2 \
                 --lambda_cov 2.0 --lambda_join 1.0 --lambda_coarse 3.0 \
                 --swap_passes 2
```
This writes `./data/{dataset}/greedy_swap_k{K}.json` and prints evaluation
metrics via `metrics.eval_preds`.
"""

from __future__ import annotations

import argparse
import pickle
from typing import List, Sequence, Dict, Tuple

import numpy as np
from tqdm import tqdm

from utils import read_json, write_json, get_corpus, get_skip_idxs
from compatibility import get_cr
from metrics import eval_preds

# ---------------------------------------------------------------------------
# -------- Helper loading functions (unchanged from the old greedy.py) ------
# ---------------------------------------------------------------------------

def load_fine_scores(dataset: str):
    """Return a list of (#subq × |corpus|) fine‑grained score matrices."""
    with open(f"./data/{dataset}/contriever/score_decomp.pkl", "rb") as fh:
        flat = pickle.load(fh)

    decomp = read_json(f"./data/{dataset}/decomp.json")
    cum = [0]
    for subs in decomp:
        cum.append(cum[-1] + len(subs))

    corpus = get_corpus(dataset)
    mats: List[np.ndarray] = []

    for qi in range(len(decomp)):
        st, ed = cum[qi], cum[qi + 1]
        block = flat[st:ed]
        n_subq = ed - st
        mat = np.zeros((n_subq, len(corpus)), dtype=np.float32)
        
        for sub_idx, per_table_arrays in enumerate(block):
            for table_idx, arr in enumerate(per_table_arrays):
                try:
                    if isinstance(arr, np.ndarray):
                        # If it's already a numpy array, just take its max
                        value = float(arr.max())
                    elif isinstance(arr, list):
                        # If it's a list of numpy arrays, take max of their maxes
                        value = max(x.max() for x in arr if isinstance(x, np.ndarray))
                    else:
                        value = 0.0
                    mat[sub_idx, table_idx] = value
                except Exception as e:
                    print(f"Error at sub_idx={sub_idx}, table_idx={table_idx}")
                    print(f"arr content: {arr}")
                    raise e
                
        mats.append(mat)
    return mats
# ---------------------------------------------------------------------------
# --------------------------- Scoring utilities -----------------------------
# ---------------------------------------------------------------------------

def _coverage_gain(F: np.ndarray, best_cov: np.ndarray, idx: int) -> float:
    """Δ coverage if we add *table idx* on top of current best_cov."""
    return float(np.clip(F[:, idx] - best_cov, 0, None).sum())


def _join_gain(cr: Dict[Tuple[int, int], np.ndarray], selected: Sequence[int], idx: int) -> float:
    if not selected:
        return 0.0
    gain = 0.0
    for j in selected:
        mat = cr[(idx, j)] if (idx, j) in cr else cr[(j, idx)]
        # compatibility score for the pair = max entry in matrix
        gain += float(mat.max())
    return gain


def _total_score(selected: Sequence[int], F: np.ndarray, coarse: np.ndarray, cr: Dict[Tuple[int, int], np.ndarray],
                 lambda_cov: float, lambda_join: float, lambda_coarse: float) -> float:
    if not selected:
        return -np.inf

    # coverage term – sum over sub‑questions of the best table score
    cov_vec = F[:, selected].max(axis=1)
    cover = float(cov_vec.sum())

    # join term – sum over unordered pairs in S of max compat
    join = 0.0
    for i, ti in enumerate(selected):
        for tj in selected[i + 1:]:
            mat = cr[(ti, tj)] if (ti, tj) in cr else cr[(tj, ti)]
            join += float(mat.max())

    coarse_sum = float(coarse[selected].sum())
    return lambda_cov * cover + lambda_join * join + lambda_coarse * coarse_sum

# ---------------------------------------------------------------------------
# ----------------------- Greedy‑then‑Local‑Swap core -----------------------
# ---------------------------------------------------------------------------

def greedy_local_swap(
    F: np.ndarray,
    coarse: np.ndarray,
    cr: Dict[Tuple[int, int], np.ndarray],
    K: int,
    lambda_cov: float,
    lambda_join: float,
    lambda_coarse: float,
    swap_passes: int = 2,
) -> List[int]:
    """Return indices of the selected tables (relative to local candidate set).

    Args
    ----
    F : (#subq × M) fine‑grained relevance matrix.
    coarse : length‑M vector of coarse scores.
    cr : compatibility dict for the M tables (``(i,j) -> matrix``).
    K : maximum number of tables to select.
    lambda_* : weights on the three signals.
    swap_passes : how many full *t_out / t_in* swap scans to run.
    """

    M = F.shape[1]
    avail = set(range(M))
    selected: List[int] = []

    # Pre‑compute Δ_coverage for speed (we'll update incremental state).
    best_cov = np.zeros(F.shape[0], dtype=np.float32)

    # ---- 1️⃣  Initial seed = argmax gain ---------------------------------
    best_idx, best_gain = None, -np.inf
    for idx in avail:
        gain = (
            lambda_cov * _coverage_gain(F, best_cov, idx)
            + lambda_coarse * coarse[idx]
        )  # no join term because S is empty
        if gain > best_gain:
            best_gain, best_idx = gain, idx
    selected.append(best_idx)
    avail.remove(best_idx)
    best_cov = np.maximum(best_cov, F[:, best_idx])

    # ---- 2️⃣  Greedy add ---------------------------------------------------
    while len(selected) < K and avail:
        gains = []
        for idx in avail:
            cov_g = _coverage_gain(F, best_cov, idx)
            join_g = _join_gain(cr, selected, idx)
            total_gain = lambda_cov * cov_g + lambda_join * join_g + lambda_coarse * coarse[idx]
            gains.append((total_gain, idx, cov_g))

        gains.sort(reverse=True)
        best_gain, best_idx, cov_g = gains[0]

        # Stop early if no positive marginal improvement in coverage.
        if cov_g <= 0 and len(selected) >= K:
            break

        selected.append(best_idx)
        avail.remove(best_idx)
        best_cov = np.maximum(best_cov, F[:, best_idx])

    # ---- 3️⃣  Local swap passes -------------------------------------------
    def score(cur_sel: Sequence[int]) -> float:
        return _total_score(cur_sel, F, coarse, cr, lambda_cov, lambda_join, lambda_coarse)

    if swap_passes > 0:
        for _ in range(swap_passes):
            improved = False
            base_score = score(selected)
            for t_out in list(selected):
                for t_in in list(avail):
                    trial = selected.copy()
                    trial.remove(t_out)
                    trial.append(t_in)
                    if score(trial) > base_score + 1e-6:
                        # Accept swap
                        selected.remove(t_out)
                        selected.append(t_in)
                        avail.add(t_out)
                        avail.remove(t_in)
                        improved = True
                        base_score = score(selected)
                        break  # restart inner loop after change
                if improved:
                    break
            if not improved:
                break  # converged early

    return selected

# ---------------------------------------------------------------------------
# ------------------------------- Main CLI ----------------------------------
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Greedy+Local‑Swap table retrieval")
    parser.add_argument("--dataset", required=True, choices=["bird", "spider"])
    parser.add_argument("--model", default="contriever", choices=["contriever", "tapas"],
                        help="Name of first‑stage retriever used to load coarse scores")
    parser.add_argument("--topk", type=int, default=20, help="# candidate tables per query from stage‑1")
    parser.add_argument("--K", type=int, default=2, help="Max # tables to return per query")
    parser.add_argument("--lambda_cov", type=float, default=2.0)
    parser.add_argument("--lambda_join", type=float, default=1.0)
    parser.add_argument("--lambda_coarse", type=float, default=3.0)
    parser.add_argument("--swap_passes", type=int, default=0,
                        help="How many full local‑swap passes to perform (0 = pure greedy)")
    args = parser.parse_args()

    skip = set(get_skip_idxs(args.dataset))

    # Stage‑1 predictions (top‑k tables) ------------------------------------
    preds_topk = read_json(f"./data/{args.dataset}/{args.model}/preds_{args.topk}.json")

    # Fine‑grained per‑query matrices and coarse scores ---------------------
    fine_mats = load_fine_scores(args.dataset)
    coarse_all = np.load(f"./data/{args.dataset}/{args.model}/score.npy")

    corpus = get_corpus(args.dataset)

    all_preds: List[List[str]] = []

    for qi, top_tables in enumerate(tqdm(preds_topk, desc="Queries")):
        if qi in skip:
            all_preds.append([])
            continue

        idxs = [corpus.index(t) for t in top_tables]
        Fsub = fine_mats[qi][:, idxs]
        coarse_sub = coarse_all[qi][idxs]
        cr_sub = get_cr(args.dataset, top_tables)

        sel_local_idxs = greedy_local_swap(
            Fsub,
            coarse_sub,
            cr_sub,
            K=args.K,
            lambda_cov=args.lambda_cov,
            lambda_join=args.lambda_join,
            lambda_coarse=args.lambda_coarse,
            swap_passes=args.swap_passes,
        )
        all_preds.append([top_tables[i] for i in sel_local_idxs])

    out_fn = f"./data/{args.dataset}/greedy_swap_k{args.K}.json"
    write_json(all_preds, out_fn)
    print(f"Saved predictions → {out_fn}\n")

    # Optional immediate evaluation
    print("Evaluation on dev split (multi‑table queries only):")
    eval_preds(args.dataset, all_preds)


if __name__ == "__main__":
    main()
