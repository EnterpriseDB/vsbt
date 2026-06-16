"""
process_yfcc_filtered.py — One-time EC2 job to build yfcc-10m-filtered.hdf5.

Reads raw files from /data/yfcc-10m (or --data-dir), computes exact
filtered ground truth for each selectivity level using per-query numpy
L2 brute-force, and writes a self-contained HDF5 that vsbt can load
like any other dataset.

File formats (big-ann-benchmarks):
  .u8bin  : uint32 nrows, uint32 ncols, then nrows*ncols uint8
  .spmat  : int64 nrow, int64 ncol, int64 nnz;
            (nrow+1)*int64 indptr; nnz*int32 indices; nnz*float32 data

Filter semantics (NeurIPS 2023): AND — a base vector matches query tags
[t1, t2] only if it has BOTH t1 and t2.

Usage (on EC2):
  python utils/process_yfcc_filtered.py --workers 32
  aws s3 cp yfcc-10m-filtered.hdf5 s3://enterprisedb-vector-datasets/

Output: yfcc-10m-filtered.hdf5
  train          float32  (10M, 192)
  test           float32  (100K, 192)
  neighbors      int32    (100K, 100)  — filtered GT (computed once; GT is
                                         determined by each query's tags, not
                                         by a selectivity parameter)
  match_counts   int32    (100K,)      — actual matching set size per query;
                                         use to select queries at benchmark
                                         time: e.g. keep queries where
                                         match_counts / 10M is in [0.5%, 2%]
                                         for a ~1% selectivity benchmark
"""

import argparse
import multiprocessing as mp
import os
import struct
import time
from pathlib import Path

# Prevent numpy/BLAS from spawning threads inside each worker.
# Must be set before numpy is imported in the worker processes.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import h5py
import numpy as np
from tqdm import tqdm


SELECTIVITIES = [0.1, 1.0, 5.0, 10.0, 30.0]
TOP_K = 100

# Module-level globals inherited by worker processes via fork (Linux COW).
# Set once in main() before the Pool is created — never written by workers.
_BASE = None
_QUERIES = None
_INV_INDEX = None
_Q_INDPTR = None
_Q_INDICES = None


# ---------------------------------------------------------------------------
# Binary format parsers
# ---------------------------------------------------------------------------

def read_u8bin(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        nrows, ncols = struct.unpack("<II", f.read(8))
        data = np.frombuffer(f.read(nrows * ncols), dtype=np.uint8)
    return data.reshape(nrows, ncols).astype(np.float32)


def read_spmat(path: Path):
    """Read CSR sparse matrix. Returns (indptr int64, indices int32, shape)."""
    with open(path, "rb") as f:
        nrow, ncol, nnz = struct.unpack("<qqq", f.read(24))
        indptr  = np.frombuffer(f.read((nrow + 1) * 8), dtype=np.int64)
        indices = np.frombuffer(f.read(nnz * 4),        dtype=np.int32)
        f.read(nnz * 4)   # skip float32 values — treated as binary
    return indptr, indices, (int(nrow), int(ncol))


# ---------------------------------------------------------------------------
# Per-query L2 nearest neighbours (no per-query FAISS index)
# ---------------------------------------------------------------------------

def _l2_top_k(sub: np.ndarray, q: np.ndarray, k: int) -> np.ndarray:
    """
    Indices of the k nearest rows in sub to q (L2).
    Uses ||a-b||² = ||a||² - 2a·b + ||b||² to avoid a full diff allocation.
    """
    sub_sq = np.einsum('ij,ij->i', sub, sub)
    dists  = sub_sq - 2.0 * (sub @ q) + float(np.dot(q, q))
    k = min(k, len(dists))
    part = np.argpartition(dists, k - 1)[:k]
    return part[np.argsort(dists[part])]


# ---------------------------------------------------------------------------
# Worker — runs in a forked child process, reads globals as COW shared memory
# ---------------------------------------------------------------------------

def _worker(args):
    """Process a chunk of queries. Returns (qi_start, partial_gt, per_query_match_counts)."""
    qi_start, qi_end, top_k = args

    base      = _BASE
    queries   = _QUERIES
    inv_index = _INV_INDEX
    q_indptr  = _Q_INDPTR
    q_indices = _Q_INDICES

    n = qi_end - qi_start
    gt           = np.full((n, top_k), -1, dtype=np.int32)
    match_counts = np.zeros(n, dtype=np.int32)

    for bi, qi in enumerate(range(qi_start, qi_end)):
        q_tags = q_indices[q_indptr[qi]: q_indptr[qi + 1]]
        if len(q_tags) == 0:
            continue

        # AND: intersect posting lists, smallest first
        tag_order = sorted(range(len(q_tags)), key=lambda i: len(inv_index[q_tags[i]]))
        matching = inv_index[q_tags[tag_order[0]]]
        for i in tag_order[1:]:
            matching = np.intersect1d(matching, inv_index[q_tags[i]], assume_unique=True)
            if len(matching) == 0:
                break

        if len(matching) == 0:
            continue

        match_counts[bi] = len(matching)
        k   = min(top_k, len(matching))
        sub = base[matching]
        positions = _l2_top_k(sub, queries[qi], k)
        gt[bi, :k] = matching[positions]

    return qi_start, gt, match_counts


# ---------------------------------------------------------------------------
# Inverted index build
# ---------------------------------------------------------------------------

def build_inverted_index(indptr, indices, n_tags: int):
    print("Building inverted tag index...", flush=True)
    inv = [[] for _ in range(n_tags)]
    n_base = len(indptr) - 1
    for base_id in tqdm(range(n_base), desc="  Inverted index", ncols=80):
        for tag_id in indices[indptr[base_id]: indptr[base_id + 1]]:
            inv[tag_id].append(base_id)
    return [np.array(v, dtype=np.int32) for v in inv]


# ---------------------------------------------------------------------------
# Parallel GT computation
# ---------------------------------------------------------------------------

_CHUNK = 200   # queries per task; small enough for frequent progress updates


def compute_filtered_gt(n_queries: int, top_k: int, n_workers: int) -> tuple:
    """
    Compute GT once for all queries. Each query's GT is determined solely
    by its tag filter — selectivity is an output, not an input.

    Returns (gt int32 (n_queries, top_k), match_counts int32 (n_queries,)).
    """
    tasks = [
        (start, min(start + _CHUNK, n_queries), top_k)
        for start in range(0, n_queries, _CHUNK)
    ]

    gt           = np.full((n_queries, top_k), -1, dtype=np.int32)
    match_counts = np.zeros(n_queries, dtype=np.int32)

    ctx = mp.get_context("fork")
    with ctx.Pool(processes=n_workers) as pool:
        bar = tqdm(total=n_queries, desc="  Computing GT", ncols=80)
        for qi_start, partial_gt, partial_counts in pool.imap_unordered(_worker, tasks):
            n = partial_gt.shape[0]
            gt[qi_start: qi_start + n]           = partial_gt
            match_counts[qi_start: qi_start + n] = partial_counts
            bar.update(n)
        bar.close()

    return gt, match_counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/data/yfcc-10m")
    parser.add_argument("--out",      default="yfcc-10m-filtered.hdf5")
    parser.add_argument("--top-k",    type=int, default=TOP_K)
    parser.add_argument("--workers",  type=int,
                        default=min(os.cpu_count() or 1, 32),
                        help="Parallel worker processes (default: min(cpu_count, 32))")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    print("Loading base vectors...")
    base = read_u8bin(data_dir / "base.10M.u8bin")
    print(f"  base: {base.shape}")

    print("Loading query vectors...")
    queries = read_u8bin(data_dir / "query.public.100K.u8bin")
    print(f"  queries: {queries.shape}")

    print("Loading base metadata (.spmat)...")
    b_indptr, b_indices, b_shape = read_spmat(data_dir / "base.metadata.10M.spmat")
    print(f"  base metadata: {b_shape[0]} vectors, {b_shape[1]} tags, nnz={len(b_indices):,}")

    print("Loading query metadata (.spmat)...")
    q_indptr, q_indices, q_shape = read_spmat(data_dir / "query.metadata.public.100K.spmat")
    print(f"  query metadata: {q_shape[0]} queries")

    print("Building inverted tag index...")
    inv_index = build_inverted_index(b_indptr, b_indices, n_tags=b_shape[1])

    # Set globals before forking so workers inherit them via COW
    global _BASE, _QUERIES, _INV_INDEX, _Q_INDPTR, _Q_INDICES
    _BASE      = base
    _QUERIES   = queries
    _INV_INDEX = inv_index
    _Q_INDPTR  = q_indptr
    _Q_INDICES = q_indices

    n_queries = queries.shape[0]
    print(f"\nUsing {args.workers} worker processes")

    t0 = time.perf_counter()
    gt, match_counts = compute_filtered_gt(n_queries, args.top_k, args.workers)
    elapsed = time.perf_counter() - t0

    n_base = base.shape[0]
    sel_pcts = match_counts / n_base * 100
    print(f"\n  Done in {elapsed:.0f}s")
    print(f"  Selectivity distribution across queries:")
    for lo, hi in [(0, 0.5), (0.5, 2), (2, 7), (7, 15), (15, 100)]:
        n = int(((sel_pcts >= lo) & (sel_pcts < hi)).sum())
        print(f"    [{lo:.1f}%, {hi:.0f}%): {n:,} queries")

    with h5py.File(args.out, "w") as hf:
        hf.create_dataset("train",        data=base,         compression="lzf")
        hf.create_dataset("test",         data=queries,      compression="lzf")
        hf.create_dataset("neighbors",    data=gt,           compression="lzf")
        hf.create_dataset("match_counts", data=match_counts, compression="lzf")
        hf.attrs["metric"]   = "l2"
        hf.attrs["dim"]      = int(base.shape[1])
        hf.attrs["n_base"]   = n_base

    print(f"\nWrote {args.out}")
    print(f"  aws s3 cp {args.out} s3://enterprisedb-vector-datasets/")


if __name__ == "__main__":
    main()
