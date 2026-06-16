"""
process_yfcc_filtered.py — One-time EC2 job to build yfcc-10m-filtered.hdf5.

Reads raw files from /data/yfcc-10m (or --data-dir), computes exact
filtered ground truth for each selectivity level using FAISS, and writes
a self-contained HDF5 that vsbt can load like any other dataset.

File formats (big-ann-benchmarks):
  .u8bin  : uint32 nrows, uint32 ncols, then nrows*ncols uint8
  .ibin   : int32  nrows, int32  ncols, then nrows*ncols int32
  .spmat  : int64 nrow, int64 ncol, int64 nnz;
            (nrow+1)*int64 indptr; nnz*int32 indices; nnz*float32 data
            (CSR, values are weights — for tag matching we treat as binary)

Filter semantics (NeurIPS 2023): AND — a base vector matches query tags
[t1, t2] only if it has BOTH t1 and t2.

Usage (on EC2):
  python utils/process_yfcc_filtered.py
  aws s3 cp yfcc-10m-filtered.hdf5 s3://enterprisedb-vector-datasets/

Output: yfcc-10m-filtered.hdf5
  train                  float32  (10M, 192)
  test                   float32  (100K, 192)
  neighbors_<sel>pct     int32    (100K, 100)   — filtered GT
  selectivities                                 — attrs recording actual values
"""

import argparse
import struct
import time
from pathlib import Path

import faiss
import h5py
import numpy as np
from tqdm import tqdm


SELECTIVITIES = [0.1, 1.0, 5.0, 10.0, 30.0]
TOP_K = 100


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
        # values (float32) — treat as binary for tag matching, skip
        f.read(nnz * 4)
    return indptr, indices, (int(nrow), int(ncol))


# ---------------------------------------------------------------------------
# Filtered GT computation
# ---------------------------------------------------------------------------

def compute_filtered_gt(base: np.ndarray, queries: np.ndarray,
                        base_indptr, base_indices,
                        query_indptr, query_indices,
                        selectivity_pct: float,
                        top_k: int = TOP_K) -> np.ndarray:
    """
    For each query, find its tag filter, identify matching base vectors
    (AND semantics), compute exact FAISS nearest neighbours within that set.

    Returns gt: (n_queries, top_k) int32 — base row IDs.
    """
    n_queries = queries.shape[0]
    gt = np.full((n_queries, top_k), -1, dtype=np.int32)

    hits = 0
    miss = 0

    for qi in tqdm(range(n_queries), desc=f"  GT@{selectivity_pct}%", ncols=80):
        # Tags for this query
        q_tags = set(
            query_indices[query_indptr[qi]: query_indptr[qi + 1]].tolist()
        )
        if not q_tags:
            miss += 1
            continue

        # Base vectors that have ALL query tags (AND semantics)
        # Start from smallest tag set to minimise work
        tag_list = sorted(q_tags)
        # For each tag, collect base vector IDs that have it
        # We use the transpose: tag→base_ids would require an inverted index.
        # For 10M base with sparse tags we scan in chunks using the CSR.
        # Practical shortcut: for each base vector check if it contains all tags.
        # This is O(N) per query — too slow for 100K queries × 10M base.
        # Instead we build a per-tag inverted index once (below this function).
        raise RuntimeError("Use compute_filtered_gt_with_invindex instead")

    return gt


def build_inverted_index(indptr, indices, n_tags: int):
    """Build tag→list[base_id] inverted index from CSR matrix."""
    print("Building inverted tag index...", flush=True)
    inv = [[] for _ in range(n_tags)]
    n_base = len(indptr) - 1
    for base_id in tqdm(range(n_base), desc="  Inverted index", ncols=80):
        for tag_id in indices[indptr[base_id]: indptr[base_id + 1]]:
            inv[tag_id].append(base_id)
    # Convert to sorted numpy arrays for fast intersection
    inv_np = [np.array(v, dtype=np.int32) for v in inv]
    return inv_np


def compute_filtered_gt_fast(base: np.ndarray, queries: np.ndarray,
                              base_indptr, base_indices,
                              query_indptr, query_indices,
                              inv_index,
                              selectivity_pct: float,
                              top_k: int = TOP_K) -> tuple:
    """
    Compute filtered GT using a per-tag inverted index (AND semantics).

    Returns (gt, actual_selectivity_pct).
    """
    n_queries = queries.shape[0]
    gt = np.full((n_queries, top_k), -1, dtype=np.int32)
    total_match_count = 0

    for qi in tqdm(range(n_queries), desc=f"  GT@{selectivity_pct}%", ncols=80):
        q_tags = query_indices[query_indptr[qi]: query_indptr[qi + 1]]
        if len(q_tags) == 0:
            continue

        # Intersect posting lists (AND)
        matching = inv_index[q_tags[0]]
        for tag_id in q_tags[1:]:
            matching = np.intersect1d(matching, inv_index[tag_id], assume_unique=True)

        if len(matching) < top_k:
            # Not enough candidates; record what we have
            if len(matching) > 0:
                sub = base[matching]
                q = queries[qi: qi + 1].astype(np.float32)
                dists = np.sum((sub - q) ** 2, axis=1)
                order = np.argsort(dists)[:top_k]
                gt[qi, :len(order)] = matching[order]
            continue

        total_match_count += len(matching)

        # FAISS brute-force on matching subset
        sub = base[matching].astype(np.float32)
        index = faiss.IndexFlatL2(sub.shape[1])
        index.add(sub)
        q = queries[qi: qi + 1].astype(np.float32)
        _, positions = index.search(q, top_k)
        gt[qi] = matching[positions[0]]

    actual_sel = total_match_count / (n_queries * base.shape[0]) * 100
    return gt, actual_sel


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",  default="/data/yfcc-10m")
    parser.add_argument("--out",       default="yfcc-10m-filtered.hdf5")
    parser.add_argument("--top-k",     type=int, default=TOP_K)
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
    print(f"  query metadata: {q_shape[0]} queries, tags vocab={b_shape[1]}")

    inv_index = build_inverted_index(b_indptr, b_indices, n_tags=b_shape[1])

    with h5py.File(args.out, "w") as hf:
        hf.create_dataset("train", data=base,    compression="lzf")
        hf.create_dataset("test",  data=queries, compression="lzf")
        hf.attrs["metric"] = "l2"
        hf.attrs["dim"]    = int(base.shape[1])

        for sel in SELECTIVITIES:
            t0 = time.perf_counter()
            gt, actual_sel = compute_filtered_gt_fast(
                base, queries,
                b_indptr, b_indices,
                q_indptr, q_indices,
                inv_index,
                selectivity_pct=sel,
                top_k=args.top_k,
            )
            elapsed = time.perf_counter() - t0
            key = f"neighbors_{sel}pct".replace(".", "_")
            hf.create_dataset(key, data=gt, compression="lzf")
            hf[key].attrs["target_selectivity_pct"] = sel
            hf[key].attrs["actual_selectivity_pct"] = actual_sel
            print(f"  {key}: actual_sel={actual_sel:.2f}%  ({elapsed:.0f}s)")

    print(f"\nWrote {args.out}")
    print(f"Upload with:")
    print(f"  aws s3 cp {args.out} s3://enterprisedb-vector-datasets/")


if __name__ == "__main__":
    main()
