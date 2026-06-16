"""
generate_synthetic_filtered.py — Add synthetic filter labels and pre-compute
filtered ground truth for any vsbt dataset.

Outputs one HDF5 file per (dataset, correlation_type) with:
  train               float32  (N, D)
  test                float32  (Q, D)
  filter_<sel>pct     int8     (N,)   — 1 = passes filter, 0 = filtered out
  neighbors_<sel>pct  int32    (Q, K) — exact GT within passing set

Correlation types:
  neutral        — filter labels are random, independent of embedding space
  anticorrelated — label=1 assigned to vectors FARTHEST from the query
                   centroid; the query's true neighbours mostly fail the filter

Usage (EC2):
  python utils/generate_synthetic_filtered.py --dataset laion-5m-test-ip
  python utils/generate_synthetic_filtered.py --dataset sift-128-euclidean
  python utils/generate_synthetic_filtered.py --dataset dbpedia-openai-1000k-angular
  # ... etc.

  # Then upload:
  aws s3 cp laion-5m-filtered-neutral.hdf5         s3://enterprisedb-vector-datasets/
  aws s3 cp laion-5m-filtered-anticorrelated.hdf5  s3://enterprisedb-vector-datasets/

RAM requirements (FAISS flat index at 30% selectivity):
  sift-1m   (128D)  :  ~50 MB    — trivial
  dbpedia-1m (1536D):  ~740 MB   — trivial
  laion-5m  (768D)  :  ~7 GB     — fine
  laion-20m (768D)  :  ~27 GB    — needs a medium EC2
  laion-100m (768D) :  ~135 GB   — needs r6i.4xlarge or larger
"""

import argparse
import time
from pathlib import Path

import faiss
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


# ---------------------------------------------------------------------------
# Label generation
# ---------------------------------------------------------------------------

def _filter_scores_neutral(n: int) -> np.ndarray:
    """Uniform random scores in [0, 1) — independent of embedding space."""
    rng = np.random.default_rng(SEED)
    return rng.random(n).astype(np.float32)


def _filter_scores_anticorrelated(train: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """
    Score each train vector by its distance from the query centroid.
    High score = far from query neighbourhood = most anticorrelated.

    Assigning label=1 (pass) to the highest-scoring vectors means the
    filtered set is semantically opposite to where queries look — the
    hardest possible workload for any graph or IVF index.
    """
    centroid = queries.mean(axis=0).astype(np.float32)
    centroid /= np.linalg.norm(centroid) + 1e-8
    # Negative dot product: high value = far from centroid direction
    scores = -(train.astype(np.float32) @ centroid)
    # Shift to [0, 1)
    scores -= scores.min()
    scores /= scores.max() + 1e-8
    return scores


def make_filter_labels(scores: np.ndarray, selectivity_pct: float) -> np.ndarray:
    """
    Assign label=1 to the top selectivity_pct% vectors by score.
    Returns int8 array (N,).
    """
    threshold = np.percentile(scores, 100.0 - selectivity_pct)
    labels = (scores >= threshold).astype(np.int8)
    return labels


# ---------------------------------------------------------------------------
# Filtered GT
# ---------------------------------------------------------------------------

def compute_filtered_gt(train: np.ndarray, queries: np.ndarray,
                        labels: np.ndarray, metric: str,
                        top_k: int = TOP_K) -> tuple:
    """
    Exact nearest-neighbour GT restricted to label=1 vectors.
    Returns (gt int32 (Q, top_k), actual_selectivity_pct float).
    """
    live_idx = np.where(labels == 1)[0].astype(np.int32)
    actual_sel = len(live_idx) / len(train) * 100

    live_vecs = train[live_idx].astype(np.float32)
    q = queries.astype(np.float32)
    D = live_vecs.shape[1]

    if metric in ("cos",):
        norms = np.linalg.norm(live_vecs, axis=1, keepdims=True)
        live_vecs = live_vecs / np.maximum(norms, 1e-8)
        norms_q = np.linalg.norm(q, axis=1, keepdims=True)
        q = q / np.maximum(norms_q, 1e-8)
        index = faiss.IndexFlatIP(D)
    elif metric in ("ip", "dot"):
        index = faiss.IndexFlatIP(D)
    else:  # l2, euclidean
        index = faiss.IndexFlatL2(D)

    index.add(live_vecs)
    _, positions = index.search(q, min(top_k, len(live_idx)))

    gt = np.full((len(queries), top_k), -1, dtype=np.int32)
    for qi in range(len(queries)):
        valid = positions[qi][positions[qi] >= 0]
        gt[qi, :len(valid)] = live_idx[valid]

    return gt, actual_sel


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        help="Dataset name as registered in datasets.py")
    parser.add_argument("--out-dir", default=".",
                        help="Output directory for generated HDF5 files")
    parser.add_argument("--top-k",  type=int, default=TOP_K)
    parser.add_argument("--selectivities", nargs="+", type=float,
                        default=SELECTIVITIES,
                        help="Selectivity percentages to generate GT for")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset '{args.dataset}'...")
    ds = ds_mod.get_dataset(args.dataset)
    metric = ds["metric"]

    # Load train into RAM (needed for FAISS)
    print("  Reading train vectors into RAM...")
    train_raw = ds["train"]
    train = train_raw[:].astype(np.float32) if hasattr(train_raw, "__getitem__") else np.array(list(train_raw), dtype=np.float32)
    queries = ds["test"].astype(np.float32)
    N, D = train.shape
    Q = queries.shape[0]
    print(f"  train={train.shape}  queries={queries.shape}  metric={metric}")

    # Pre-compute scores for both correlation types
    scores = {
        "neutral":        _filter_scores_neutral(N),
        "anticorrelated": _filter_scores_anticorrelated(train, queries),
    }

    for corr_type in CORRELATION_TYPES:
        base_name = args.dataset.replace("-test-", "-").replace("-train-", "-")
        out_path = out_dir / f"{base_name}-filtered-{corr_type}.hdf5"
        print(f"\n=== {corr_type} → {out_path} ===")

        with h5py.File(out_path, "w") as hf:
            # train is not written — load from the original dataset at benchmark time.
            hf.create_dataset("test",   data=queries, compression="lzf")
            hf.attrs["metric"]      = metric
            hf.attrs["source"]      = args.dataset
            hf.attrs["correlation"] = corr_type

            sc = scores[corr_type]

            for sel in args.selectivities:
                print(f"  selectivity={sel}%")
                labels = make_filter_labels(sc, sel)

                key_sel = f"{sel}pct".replace(".", "_")

                # Store filter labels so the benchmark can load the column
                hf.create_dataset(f"filter_{key_sel}", data=labels, compression="lzf")

                # Compute filtered GT
                t0 = time.perf_counter()
                gt, actual_sel = compute_filtered_gt(
                    train, queries, labels, metric, top_k=args.top_k
                )
                elapsed = time.perf_counter() - t0
                hf.create_dataset(f"neighbors_{key_sel}", data=gt, compression="lzf")
                hf[f"neighbors_{key_sel}"].attrs["target_selectivity_pct"] = sel
                hf[f"neighbors_{key_sel}"].attrs["actual_selectivity_pct"] = actual_sel
                print(f"    actual_sel={actual_sel:.2f}%  GT done in {elapsed:.1f}s")

        size_mb = out_path.stat().st_size / 1024 / 1024
        print(f"  Wrote {out_path}  ({size_mb:.0f} MB)")
        print(f"  Upload: aws s3 cp {out_path} s3://enterprisedb-vector-datasets/")


if __name__ == "__main__":
    main()
