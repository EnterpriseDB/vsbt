"""Generate small CI/CD subsets of the openai / cohere / laion datasets.

Slices the first N base vectors (5k / 50k by default) out of each family's
smallest locally-available source dataset, reuses a slice of its query set,
and **recomputes** ground-truth neighbours against each subset (correctness
requires this regardless of whether the parent's GT happens to overlap).

Source per family (see FAMILIES below):
    openai  <- openai-500k (parquet, dim 1536, metric cos)
    cohere  <- cohere-1m   (parquet, dim 768,  metric cos)
    laion   <- laion-5m-test-ip (HDF5, dim 768, metric ip)

Whatever the source format, output is always the three-file parquet shape
`_load_parquet` in datasets.py expects. `<DATA_DIR>/<family>/<label>` is
exactly the `base_dir` datasets.py's `<family>-<label>-<metric>` entry
points at:
    <DATA_DIR>/<family>/<label>/shuffle_train.parquet   (id, emb)
    <DATA_DIR>/<family>/<label>/test.parquet            (id, emb)
    <DATA_DIR>/<family>/<label>/neighbors.parquet       (id, neighbors_id)

This is a one-time, offline prep tool. If a family's source dataset is
not present locally it is downloaded from S3 first (reusing datasets.py's
downloaders), so a clean checkout can generate subsets unattended. The
generated files are then uploaded to S3 so CI can download them.

Usage:
    python utils/derive_datasets.py
    python utils/derive_datasets.py --families openai,cohere --sizes 5000

Ground truth is stored for the full query set by default (1000 — the
largest size shared by all three sources) — it costs nothing to keep them
all here. Trim to fewer queries at run time instead, e.g. via the suite's
`--max-queries` flag, rather than regenerating a smaller dataset.
"""
import argparse
import os
import sys

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# datasets.py lives at the repo root and does `from utils...` itself, so make
# the repo root importable before pulling in its S3 downloaders / metadata.
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from datasets import DATASETS, _download_parquet_from_s3, download_http_file
# Mirrors datasets.py's DATA_DIR/DATASET_LOCAL_DIR so this tool and the loader
# agree on where a subset's files live, even if DATASET_LOCAL_DIR is overridden.
DATA_DIR = os.environ.get("DATASET_LOCAL_DIR", os.path.join(REPO_ROOT, "datasets"))
DEFAULT_SIZES = [5_000, 50_000]

# Per-family source description. `source` is a parquet dir (containing
# shuffle_train.parquet/test.parquet) or a single HDF5 file, matching how
# datasets.py locates each family's smallest dataset. `source_dataset` names
# the datasets.py entry the source is downloaded from when absent locally, so
# S3 coordinates (prefix / url / shard count) live in one place only.
FAMILIES = {
    "openai": {
        "source_type": "parquet",
        "source": os.path.join(DATA_DIR, "openai", "500k"),
        "source_dataset": "openai-500k-cos",
        "dim": 1536,
        "metric": "cos",
    },
    "cohere": {
        "source_type": "parquet",
        "source": os.path.join(DATA_DIR, "cohere", "1m"),
        "source_dataset": "cohere-1m-cos",
        "dim": 768,
        "metric": "cos",
    },
    "laion": {
        "source_type": "hdf5",
        "source": os.path.join(DATA_DIR, "laion-5m-test-ip.hdf5"),
        "source_dataset": "laion-5m-test-ip",
        "dim": 768,
        "metric": "ip",
    },
}


def _ensure_source_present(family, family_cfg):
    """Download a family's source dataset from S3 if it is not already local.

    Delegates to datasets.py's downloaders (both are no-ops when the target
    already exists) and pulls the S3 coordinates from its DATASETS metadata so
    they are never duplicated here.
    """
    ds = DATASETS[family_cfg["source_dataset"]]
    source = family_cfg["source"]

    if family_cfg["source_type"] == "parquet":
        if os.path.exists(os.path.join(source, "test.parquet")):
            return
        print(f"  source for {family!r} missing; downloading {ds['s3_prefix']} ...")
        _download_parquet_from_s3(ds["s3_prefix"], source, ds["num"],
                                  ds.get("num_shards"))
    elif family_cfg["source_type"] == "hdf5":
        if os.path.exists(source):
            return
        print(f"  source for {family!r} missing; downloading {ds['url']} ...")
        download_http_file(ds["url"], source)
    else:
        raise ValueError(f"Unknown source_type: {family_cfg['source_type']!r}")


def _size_label(n: int) -> str:
    if n % 1_000 != 0:
        raise ValueError(f"size {n} is not a multiple of 1000")
    return f"{n // 1_000}k"


def _read_first_rows(path: str, n: int) -> pa.Table:
    """Read the first `n` rows of a parquet file, preserving its arrow schema."""
    pf = pq.ParquetFile(path)
    batches = []
    remaining = n
    for batch in pf.iter_batches(batch_size=10_000):
        if remaining <= 0:
            break
        if batch.num_rows > remaining:
            batch = batch.slice(0, remaining)
        batches.append(batch)
        remaining -= batch.num_rows
    if remaining > 0:
        raise ValueError(
            f"{path} has fewer than {n} rows (short by {remaining})"
        )
    return pa.Table.from_batches(batches)


def _embs_to_numpy(table: pa.Table) -> np.ndarray:
    """Extract the `emb` list-column into a (rows, dim) float64 array."""
    return np.asarray(table.column("emb").to_pylist(), dtype=np.float64)


def _load_parquet_source(source_dir, max_n, num_queries):
    """Load (base_ids, base_embs, query_embs) from a parquet-family source dir."""
    train_path = os.path.join(source_dir, "shuffle_train.parquet")
    test_path = os.path.join(source_dir, "test.parquet")
    for p in (train_path, test_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Source file not found: {p}")

    train_table = _read_first_rows(train_path, max_n)
    base_ids = train_table.column("id").to_pylist()
    base_embs = _embs_to_numpy(train_table)

    query_table = _read_first_rows(test_path, num_queries)
    query_embs = _embs_to_numpy(query_table)

    return base_ids, base_embs, query_embs


def _load_hdf5_source(source_path, max_n, num_queries):
    """Load (base_ids, base_embs, query_embs) from an HDF5 family source.

    HDF5 (ann-benchmarks-style) files have no `id` column — rows are
    identified by their position, matching the convention `_load_hdf5_dataset`
    in datasets.py relies on. So base_ids here are just 0..max_n-1.
    """
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source file not found: {source_path}")

    with h5py.File(source_path, "r") as f:
        base_embs = np.asarray(f["train"][:max_n], dtype=np.float64)
        available_queries = f["test"].shape[0]
        n_queries = min(num_queries, available_queries)
        query_embs = np.asarray(f["test"][:n_queries], dtype=np.float64)

    base_ids = list(range(max_n))
    return base_ids, base_embs, query_embs


def _ground_truth(base_vecs, base_ids, query_vecs, depth, metric):
    """Return, per query, the `depth` nearest base `id`s.

    Both cosine and inner-product ranking are "higher score = nearer", so
    a single descending-argsort over a similarity/dot-product matrix covers
    both — cosine just needs its vectors L2-normalised first.
    """
    def _l2_normalize(a):
        norms = np.linalg.norm(a, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return a / norms

    if metric == "cos":
        base_use = _l2_normalize(base_vecs)
        query_use = _l2_normalize(query_vecs)
    elif metric == "ip":
        base_use = base_vecs
        query_use = query_vecs
    else:
        raise ValueError(f"Unsupported metric for ground truth: {metric!r}")

    scores = query_use @ base_use.T  # (num_queries, N)
    depth = min(depth, base_vecs.shape[0])
    order = np.argsort(-scores, axis=1)[:, :depth]
    base_ids = np.asarray(base_ids)
    return [base_ids[row].tolist() for row in order]


def generate_subset(family, family_cfg, base_ids_full, base_embs_full,
                    query_embs, n, gt_depth):
    label = _size_label(n)
    out_dir = os.path.join(DATA_DIR, family, label)
    os.makedirs(out_dir, exist_ok=True)

    base_ids = base_ids_full[:n]
    base_embs = base_embs_full[:n]

    train_table = pa.table({
        "id": pa.array(base_ids, type=pa.int64()),
        "emb": pa.array(base_embs.tolist(), type=pa.list_(pa.float64())),
    })
    pq.write_table(train_table, os.path.join(out_dir, "shuffle_train.parquet"))

    query_ids = list(range(len(query_embs)))
    query_table = pa.table({
        "id": pa.array(query_ids, type=pa.int64()),
        "emb": pa.array(query_embs.tolist(), type=pa.list_(pa.float64())),
    })
    pq.write_table(query_table, os.path.join(out_dir, "test.parquet"))

    neighbors = _ground_truth(base_embs, base_ids, query_embs, gt_depth,
                              family_cfg["metric"])
    neighbors_table = pa.table({
        "id": pa.array(query_ids, type=pa.int64()),
        "neighbors_id": pa.array(neighbors, type=pa.list_(pa.int64())),
    })
    pq.write_table(neighbors_table, os.path.join(out_dir, "neighbors.parquet"))

    metric = family_cfg["metric"]
    print(f"  {family}-{label}-{metric}: train={n} queries={len(query_ids)} "
          f"gt_depth={len(neighbors[0])} -> {out_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--families", default=",".join(FAMILIES),
                        help="comma-separated families to generate (default: %(default)s)")
    parser.add_argument("--sizes", default=",".join(str(s) for s in DEFAULT_SIZES),
                        help="comma-separated base-vector counts (default: %(default)s)")
    parser.add_argument("--num-queries", type=int, default=1_000,
                        help="number of test queries to slice (default: %(default)s, "
                             "the largest query set shared by all sources — trim at "
                             "run time via the suite's --max-queries instead of "
                             "lowering this)")
    parser.add_argument("--gt-depth", type=int, default=100,
                        help="ground-truth neighbours stored per query (default: %(default)s)")
    args = parser.parse_args()

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    for family in families:
        if family not in FAMILIES:
            raise ValueError(f"Unknown family {family!r}; expected one of {sorted(FAMILIES)}")

    sizes = sorted(int(s) for s in args.sizes.split(",") if s.strip())
    max_n = max(sizes)

    for family in families:
        family_cfg = FAMILIES[family]
        _ensure_source_present(family, family_cfg)
        print(f"Reading first {max_n} base vectors and {args.num_queries} queries "
              f"for {family!r} from {family_cfg['source']} ...")

        if family_cfg["source_type"] == "parquet":
            base_ids, base_embs, query_embs = _load_parquet_source(
                family_cfg["source"], max_n, args.num_queries)
        elif family_cfg["source_type"] == "hdf5":
            base_ids, base_embs, query_embs = _load_hdf5_source(
                family_cfg["source"], max_n, args.num_queries)
        else:
            raise ValueError(f"Unknown source_type: {family_cfg['source_type']!r}")

        print(f"Generating {family} subsets:")
        for n in sizes:
            generate_subset(family, family_cfg, base_ids, base_embs, query_embs,
                            n, args.gt_depth)

    print("Done.")


if __name__ == "__main__":
    main()
