"""
generate_synthetic_filtered.py — Add synthetic filter labels and pre-compute
filtered ground truth for any vsbt dataset.

Memory-efficient: never loads the full train set into RAM.
  - Neutral scores: random numbers, no train access needed.
  - Anticorrelated scores: chunked dot product, one slice at a time.
  - GT computation: chunked matrix multiply maintaining a running top-K;
    no FAISS index, no full-live-set copy in RAM.

Peak RAM for LAION-100M at 30% selectivity: ~8 GB.

Outputs one HDF5 file per (dataset, correlation_type):
  test                float32  (Q, D)
  filter_<sel>pct     int8     (N,)   — 1 = passes filter, 0 = filtered out
  neighbors_<sel>pct  int32    (Q, K) — exact GT within passing set

Usage (EC2):
  python utils/generate_synthetic_filtered.py --dataset laion-100m-test-ip
  python utils/generate_synthetic_filtered.py --dataset laion-5m-test-ip
  python utils/generate_synthetic_filtered.py --dataset sift-128-euclidean
  python utils/generate_synthetic_filtered.py --dataset dbpedia-openai-1000k-angular
"""

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import datasets as ds_mod

SELECTIVITIES = [0.1, 1.0, 5.0, 10.0, 30.0]
CORRELATION_TYPES = ["neutral", "anticorrelated"]
TOP_K = 100
SEED = 42
SCORE_CHUNK  = 500_000   # rows per chunk when computing anticorrelated scores
GT_CHUNK     = 200_000   # live vectors per chunk when computing GT


# ---------------------------------------------------------------------------
# Score computation (no full-train load)
# ---------------------------------------------------------------------------

def _filter_scores_neutral(n: int) -> np.ndarray:
    """Uniform random scores — no train vectors needed."""
    return np.random.default_rng(SEED).random(n).astype(np.float32)


def _filter_scores_anticorrelated(train_raw, n: int, d: int,
                                   queries: np.ndarray) -> np.ndarray:
    """
    Chunked dot product of each train vector against the query centroid.
    High score = far from query neighbourhood.
    Memory: one chunk of train at a time.
    """
    centroid = queries.mean(axis=0).astype(np.float32)
    centroid /= np.linalg.norm(centroid) + 1e-8

    scores = np.empty(n, dtype=np.float32)
    for start in tqdm(range(0, n, SCORE_CHUNK), desc="  anti-corr scores", ncols=80):
        end = min(start + SCORE_CHUNK, n)
        chunk = train_raw[start:end].astype(np.float32)
        scores[start:end] = -(chunk @ centroid)

    scores -= scores.min()
    scores /= scores.max() + 1e-8
    return scores


def make_filter_labels(scores: np.ndarray, selectivity_pct: float) -> np.ndarray:
    threshold = np.percentile(scores, 100.0 - selectivity_pct)
    return (scores >= threshold).astype(np.int8)


# ---------------------------------------------------------------------------
# GT computation (chunked matrix multiply, no FAISS, no full-live-set in RAM)
# ---------------------------------------------------------------------------

def compute_filtered_gt(train_raw, queries: np.ndarray, labels: np.ndarray,
                        metric: str, top_k: int = TOP_K) -> tuple:
    """
    Exact nearest-neighbour GT restricted to label=1 vectors.

    Processes live vectors in chunks of GT_CHUNK, maintaining a running
    top-K per query using vectorised argpartition.  Peak RAM per iteration:
    GT_CHUNK × D × 4 bytes for the chunk + Q × GT_CHUNK × 4 for distances.

    Returns (gt int32 (Q, top_k), actual_selectivity_pct float).
    """
    live_idx = np.where(labels == 1)[0].astype(np.int32)
    actual_sel = len(live_idx) / len(labels) * 100

    q = queries.astype(np.float32)
    if metric == "cos":
        q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-8

    Q = len(q)
    M = len(live_idx)

    # Running top-K state
    top_dists = np.full((Q, top_k), np.inf, dtype=np.float32)
    top_ids   = np.full((Q, top_k), -1, dtype=np.int32)

    bar = tqdm(total=M, desc=f"    GT", unit=" vecs", ncols=80)

    for start in range(0, M, GT_CHUNK):
        end = min(start + GT_CHUNK, M)
        chunk_live_ids = live_idx[start:end]                    # actual DB ids
        chunk = train_raw[chunk_live_ids].astype(np.float32)    # (C, D)

        if metric == "cos":
            chunk /= np.linalg.norm(chunk, axis=1, keepdims=True) + 1e-8

        C = len(chunk)

        if metric in ("ip", "dot"):
            # Higher dot product = closer; negate so lower = better
            chunk_dists = -(q @ chunk.T)                        # (Q, C)
        elif metric == "cos":
            chunk_dists = -(q @ chunk.T)
        else:  # l2
            q_sq  = np.einsum("ij,ij->i", q, q)[:, None]       # (Q, 1)
            c_sq  = np.einsum("ij,ij->i", chunk, chunk)[None, :]# (1, C)
            chunk_dists = q_sq + c_sq - 2.0 * (q @ chunk.T)    # (Q, C)

        # Merge chunk candidates with running top-K
        all_d = np.concatenate([top_dists, chunk_dists], axis=1)  # (Q, top_k+C)
        all_i = np.concatenate(
            [top_ids,
             np.broadcast_to(chunk_live_ids, (Q, C)).copy()],
            axis=1,
        )                                                          # (Q, top_k+C)

        k = min(top_k, all_d.shape[1])
        part = np.argpartition(all_d, k - 1, axis=1)[:, :k]       # (Q, k)
        rows = np.arange(Q)[:, None]
        top_dists = all_d[rows, part]
        top_ids   = all_i[rows, part]

        bar.update(C)

    bar.close()

    # Sort each query's top-K by distance
    order = np.argsort(top_dists, axis=1)
    gt = np.take_along_axis(top_ids, order, axis=1)
    return gt, actual_sel


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--selectivities", nargs="+", type=float,
                        default=SELECTIVITIES)
    parser.add_argument("--correlations", nargs="+",
                        default=CORRELATION_TYPES,
                        choices=CORRELATION_TYPES)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset '{args.dataset}'...")
    ds     = ds_mod.get_dataset(args.dataset)
    metric = ds["metric"]

    train_raw = ds["train"]                   # HDF5 handle — not loaded
    queries   = ds["test"].astype(np.float32)
    N = ds["num"]
    D = ds["dim"]
    Q = queries.shape[0]
    print(f"  N={N:,}  D={D}  Q={Q}  metric={metric}")

    # Compute scores lazily (anticorrelated reads train in chunks)
    print("\nComputing filter scores...")
    scores = {
        "neutral": _filter_scores_neutral(N),
    }
    if "anticorrelated" in args.correlations:
        scores["anticorrelated"] = _filter_scores_anticorrelated(
            train_raw, N, D, queries
        )

    for corr_type in args.correlations:
        # Output filename uses dataset name directly, no -test- stripping confusion
        out_path = out_dir / f"{args.dataset}-filtered-{corr_type}.hdf5"
        print(f"\n=== {corr_type} → {out_path.name} ===")

        with h5py.File(out_path, "w") as hf:
            hf.create_dataset("test",   data=queries, compression="lzf")
            hf.attrs["metric"]      = metric
            hf.attrs["source"]      = args.dataset
            hf.attrs["correlation"] = corr_type

            sc = scores[corr_type]

            for sel in args.selectivities:
                print(f"  selectivity={sel}%")
                labels  = make_filter_labels(sc, sel)
                key_sel = f"{sel}pct".replace(".", "_")

                hf.create_dataset(f"filter_{key_sel}",
                                  data=labels, compression="lzf")

                t0 = time.perf_counter()
                gt, actual_sel = compute_filtered_gt(
                    train_raw, queries, labels, metric, top_k=args.top_k
                )
                elapsed = time.perf_counter() - t0

                hf.create_dataset(f"neighbors_{key_sel}",
                                  data=gt, compression="lzf")
                hf[f"neighbors_{key_sel}"].attrs["target_selectivity_pct"] = sel
                hf[f"neighbors_{key_sel}"].attrs["actual_selectivity_pct"] = actual_sel
                print(f"    actual_sel={actual_sel:.2f}%  ({elapsed:.0f}s)")

        size_mb = out_path.stat().st_size / 1024 / 1024
        print(f"  {out_path.name}  ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
