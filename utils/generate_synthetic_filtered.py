"""
generate_synthetic_filtered.py — Synthetic filter labels + filtered GT for
any vsbt dataset.

Memory model
------------
The full train set is never loaded into RAM. Instead:
  - Neutral scores: rng.random(N), no train access.
  - Anticorrelated scores: chunked sequential dot product.
  - GT: ONE sequential scan of the HDF5 file per correlation type,
    processing all selectivities simultaneously. No random HDF5 seeks.

For LAION-100M at 30% selectivity peak RAM is ~4 GB.

Sequential scan structure (per corr_type)
------------------------------------------
  for each SCAN_CHUNK rows (sequential HDF5 read, ~1.5 GB/chunk):
    for each selectivity:
      find live vectors in this range (binary search on sorted live_idx)
      for each GT_CHUNK of those live vectors:
        dists = q @ live_vecs.T              (Q x GT_CHUNK matrix)
        merge into running per-query top-K

Total I/O: 2 full sequential reads of the dataset (one per corr type).
"""

import argparse
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import datasets as ds_mod

try:
    import h5py
except ImportError:
    h5py = None

SELECTIVITIES   = [0.1, 1.0, 5.0, 10.0, 30.0]
CORRELATION_TYPES = ["neutral", "anticorrelated"]
TOP_K       = 100
SEED        = 42
SCAN_CHUNK  = 500_000   # rows per sequential HDF5 read  (~1.5 GB for 768D)
GT_CHUNK    = 50_000    # live vecs per distance batch    (~2 GB dists matrix)
SCORE_CHUNK = 500_000   # rows per anticorr score chunk


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------

def _scores_neutral(n: int) -> np.ndarray:
    return np.random.default_rng(SEED).random(n).astype(np.float32)


def _scores_anticorrelated(train_raw, n: int, queries: np.ndarray) -> np.ndarray:
    centroid = queries.mean(axis=0).astype(np.float32)
    centroid /= np.linalg.norm(centroid) + 1e-8
    scores = np.empty(n, dtype=np.float32)
    for s in tqdm(range(0, n, SCORE_CHUNK), desc="  anticorr scores", ncols=80):
        e = min(s + SCORE_CHUNK, n)
        chunk = train_raw[s:e].astype(np.float32)
        scores[s:e] = -(chunk @ centroid)
    scores -= scores.min()
    scores /= scores.max() + 1e-8
    return scores


def make_filter_labels(scores: np.ndarray, sel_pct: float) -> np.ndarray:
    threshold = np.percentile(scores, 100.0 - sel_pct)
    return (scores >= threshold).astype(np.int8)


# ---------------------------------------------------------------------------
# Running top-K helpers
# ---------------------------------------------------------------------------

def _new_topk(Q: int, k: int):
    return (
        np.full((Q, k), np.inf,  dtype=np.float32),
        np.full((Q, k), -1,      dtype=np.int32),
    )


def _merge_topk(top_dists, top_ids, new_dists, new_ids):
    """Vectorised merge of (Q, k) running state with (Q, C) new candidates."""
    Q, k = top_dists.shape
    all_d = np.concatenate([top_dists, new_dists], axis=1)   # (Q, k+C)
    all_i = np.concatenate([top_ids,   new_ids],   axis=1)

    part = np.argpartition(all_d, k - 1, axis=1)[:, :k]
    rows = np.arange(Q)[:, None]
    top_dists = all_d[rows, part]
    top_ids   = all_i[rows, part]
    return top_dists, top_ids


# ---------------------------------------------------------------------------
# GT via one sequential scan per corr_type
# ---------------------------------------------------------------------------

def compute_all_gt(train_raw, queries: np.ndarray,
                   all_labels: dict,           # {sel: int8 array (N,)}
                   all_live_idx: dict,         # {sel: sorted int32 array}
                   metric: str,
                   top_k: int = TOP_K) -> dict:
    """
    One sequential scan of train_raw; computes GT for every selectivity.
    Returns {sel: (gt int32 (Q, top_k), actual_sel_pct float)}.
    """
    q = queries.astype(np.float32)
    if metric == "cos":
        q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-8
    elif metric in ("ip", "dot"):
        pass
    Q, D = q.shape

    sels      = list(all_labels.keys())
    top_dists = {s: _new_topk(Q, top_k)[0] for s in sels}
    top_ids   = {s: _new_topk(Q, top_k)[1] for s in sels}
    pointers  = {s: 0 for s in sels}           # pointer into each live_idx
    N         = len(next(iter(all_labels.values())))

    bar = tqdm(total=N, desc="  sequential scan", unit=" rows", ncols=80)

    for scan_s in range(0, N, SCAN_CHUNK):
        scan_e    = min(scan_s + SCAN_CHUNK, N)
        base_chunk = train_raw[scan_s:scan_e].astype(np.float32)  # sequential

        for sel in sels:
            live = all_live_idx[sel]
            p    = pointers[sel]

            # Binary search: find range of live_idx in [scan_s, scan_e)
            p_start = np.searchsorted(live, scan_s,  side="left")
            p_end   = np.searchsorted(live, scan_e,  side="left")
            pointers[sel] = p_end

            if p_start == p_end:
                continue

            live_in_range = live[p_start:p_end]   # sorted global IDs

            # Sub-batch distance computation to cap dists matrix size
            for gs in range(0, len(live_in_range), GT_CHUNK):
                ge       = min(gs + GT_CHUNK, len(live_in_range))
                g_ids    = live_in_range[gs:ge]                   # global IDs
                l_ids    = g_ids - scan_s                          # local offsets
                vecs     = base_chunk[l_ids]                       # (C, D)

                if metric == "cos":
                    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8

                C = len(vecs)

                if metric in ("cos", "ip", "dot"):
                    dists = -(q @ vecs.T)                          # (Q, C)
                else:
                    q_sq  = np.einsum("ij,ij->i", q, q)[:, None]
                    c_sq  = np.einsum("ij,ij->i", vecs, vecs)[None, :]
                    dists = q_sq + c_sq - 2.0 * (q @ vecs.T)

                new_ids = np.broadcast_to(g_ids, (Q, C)).copy()
                top_dists[sel], top_ids[sel] = _merge_topk(
                    top_dists[sel], top_ids[sel], dists, new_ids
                )

        bar.update(scan_e - scan_s)

    bar.close()

    results = {}
    for sel in sels:
        live    = all_live_idx[sel]
        actual  = len(live) / N * 100
        order   = np.argsort(top_dists[sel], axis=1)
        gt      = np.take_along_axis(top_ids[sel], order, axis=1)
        results[sel] = (gt, actual)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",      required=True)
    parser.add_argument("--out-dir",      default=".")
    parser.add_argument("--top-k",        type=int, default=TOP_K)
    parser.add_argument("--selectivities", nargs="+", type=float,
                        default=SELECTIVITIES)
    parser.add_argument("--correlations", nargs="+",
                        default=CORRELATION_TYPES,
                        choices=CORRELATION_TYPES)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset '{args.dataset}'...")
    ds        = ds_mod.get_dataset(args.dataset)
    metric    = ds["metric"]
    train_raw = ds["train"]
    queries   = ds["test"].astype(np.float32)
    N, D      = ds["num"], ds["dim"]
    Q         = queries.shape[0]
    print(f"  N={N:,}  D={D}  Q={Q}  metric={metric}")

    # --- compute scores ---
    print("\nComputing filter scores...")
    scores = {"neutral": _scores_neutral(N)}
    if "anticorrelated" in args.correlations:
        scores["anticorrelated"] = _scores_anticorrelated(
            train_raw, N, queries
        )

    # --- generate GT per corr_type ---
    for corr_type in args.correlations:
        out_path = out_dir / f"{args.dataset}-filtered-{corr_type}.hdf5"
        print(f"\n=== {corr_type} → {out_path.name} ===")

        sc         = scores[corr_type]
        all_labels   = {s: make_filter_labels(sc, s) for s in args.selectivities}
        all_live_idx = {
            s: np.where(all_labels[s] == 1)[0].astype(np.int32)
            for s in args.selectivities
        }

        for s, live in all_live_idx.items():
            print(f"  sel={s}%  live={len(live):,}  ({len(live)/N*100:.2f}%)")

        t0 = time.perf_counter()
        gt_results = compute_all_gt(
            train_raw, queries, all_labels, all_live_idx,
            metric, top_k=args.top_k,
        )
        elapsed = time.perf_counter() - t0
        print(f"  scan done in {elapsed:.0f}s")

        import h5py as _h5py
        with _h5py.File(out_path, "w") as hf:
            hf.create_dataset("test", data=queries, compression="lzf")
            hf.attrs["metric"]      = metric
            hf.attrs["source"]      = args.dataset
            hf.attrs["correlation"] = corr_type

            for sel in args.selectivities:
                gt, actual_sel = gt_results[sel]
                key = f"{sel}pct".replace(".", "_")
                hf.create_dataset(f"filter_{key}",
                                  data=all_labels[sel], compression="lzf")
                hf.create_dataset(f"neighbors_{key}",
                                  data=gt, compression="lzf")
                hf[f"neighbors_{key}"].attrs["target_selectivity_pct"] = sel
                hf[f"neighbors_{key}"].attrs["actual_selectivity_pct"] = actual_sel
                print(f"  neighbors_{key}: actual={actual_sel:.2f}%")

        size_mb = out_path.stat().st_size / 1024 / 1024
        print(f"  wrote {out_path.name} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
