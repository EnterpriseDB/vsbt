"""
workload.py — Incremental insert/delete workload benchmark.

Models a production index under continuous write pressure:

  1. Load start_pct% of the dataset and build the index.
  2. At each checkpoint (step_pct% increments up to 100%):
       - Insert insert_ratio * step_rows new vectors.
       - Delete delete_ratio * step_rows random live vectors.
       - Recompute exact ground truth with FAISS against the live set,
         or load it from the per-checkpoint cache on re-runs.
       - Run each configured benchmark point and record recall/QPS.
  3. Write per-checkpoint results to workload_results.csv.
  4. Print a summary table with drift% vs. the first checkpoint.

Requires 'faiss' (faiss-cpu or faiss-gpu) installed alongside the usual
vsbt dependencies.

YAML config example (workload key added to standard suite config):

  vc-laion-5m-workload:
    mode: workload           # triggers this runner (default: readonly)
    dataset: laion-5m-test-ip
    metric: dot
    lists: [50, 8000]
    samplingFactor: 256
    residual_quantization: true
    pg_parallel_workers: 32
    top: 10
    workload:
      start_pct: 50        # load this fraction first
      step_pct: 10         # grow by this much each checkpoint
      insert_ratio: 0.7    # fraction of step rows that are inserts
      delete_ratio: 0.3    # fraction of step rows that are deletes
      autovacuum: false    # disable autovacuum during the run
      rng_seed: 0          # seed for delete selection (reproducibility)
    benchmarks:
      "10-20-1.9":
        nprob: "10,20"
        epsilon: 1.9

The suite runners (pgvector_suite.py, vectorchord_suite.py) dispatch to
run_workload() automatically when they see mode: workload.  This script
can also be run standalone with the same YAML.

GT caching
----------
Ground truth is expensive (FAISS brute-force over all live vectors).
After the first run, GT arrays are cached to
  results/{suite_name}/gt_cache/gt_{pct}pct.npy
  results/{suite_name}/gt_cache/live_ids_{pct}pct.npy

On re-runs, if the cached live_ids match the current live set (which is
guaranteed when rng_seed is fixed), the GT is loaded from disk instead of
recomputed.  Delete the gt_cache/ directory to force recomputation.
"""

import argparse
import csv
import time
from pathlib import Path

import faiss
import numpy as np
import psycopg
import pgvector.psycopg
import yaml
from tqdm import tqdm

import datasets
from loader import pack_copy_binary_chunk


# ---------------------------------------------------------------------------
# Metric operator string for ORDER BY clause
# ---------------------------------------------------------------------------

_METRIC_OPS = {
    "l2": "<->",
    "euclidean": "<->",
    "cos": "<=>",
    "ip": "<#>",
    "dot": "<#>",
}


def _metric_op(metric: str) -> str:
    op = _METRIC_OPS.get(metric)
    if op is None:
        raise ValueError(f"Unknown metric: {metric}")
    return op


# ---------------------------------------------------------------------------
# Ground-truth computation
# ---------------------------------------------------------------------------

def compute_gt(live_ids: np.ndarray, train_data, queries: np.ndarray,
               top_k: int, metric: str) -> np.ndarray:
    """
    Compute exact nearest-neighbour ground truth using FAISS brute-force.

    live_ids: sorted int32 array of live row IDs  (FAISS position i → DB id live_ids[i])
    train_data: sliceable dataset array           (train_data[id] = float32 vector)
    queries: (n_queries, D) float32
    Returns gt: (n_queries, top_k) int32 — DB row IDs for each query's neighbours
    """
    vecs = train_data[live_ids]
    if vecs.dtype != np.float32:
        vecs = vecs.astype(np.float32)
    D = vecs.shape[1]
    q = queries.astype(np.float32)

    if metric in ("cos",):
        index = faiss.IndexFlatIP(D)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / np.maximum(norms, 1e-8)
        index.add(vecs)
        q_norms = np.linalg.norm(q, axis=1, keepdims=True)
        q = q / np.maximum(q_norms, 1e-8)
    elif metric in ("ip", "dot"):
        index = faiss.IndexFlatIP(D)
        index.add(vecs)
    else:  # l2, euclidean
        index = faiss.IndexFlatL2(D)
        index.add(vecs)

    _, positions = index.search(q, top_k)       # positions within vecs
    gt = live_ids[positions]                     # map back to DB ids
    return gt


# ---------------------------------------------------------------------------
# GT cache helpers
# ---------------------------------------------------------------------------

def _gt_cache_paths(cache_dir: Path, checkpoint_pct: int):
    return (cache_dir / f"gt_{checkpoint_pct}pct.npy",
            cache_dir / f"live_ids_{checkpoint_pct}pct.npy")


def _load_gt_cache(cache_dir: Path, checkpoint_pct: int,
                   live_ids: np.ndarray, top_k: int):
    """Return cached GT array if live_ids match, else None."""
    gt_path, ids_path = _gt_cache_paths(cache_dir, checkpoint_pct)
    if not (gt_path.exists() and ids_path.exists()):
        return None
    cached_ids = np.load(ids_path)
    if cached_ids.shape != live_ids.shape or not np.array_equal(cached_ids, live_ids):
        return None
    gt = np.load(gt_path)
    if gt.shape[1] < top_k:
        return None
    return gt[:, :top_k]


def _save_gt_cache(cache_dir: Path, checkpoint_pct: int,
                   live_ids: np.ndarray, gt: np.ndarray) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    gt_path, ids_path = _gt_cache_paths(cache_dir, checkpoint_pct)
    np.save(ids_path, live_ids)
    np.save(gt_path, gt)


# ---------------------------------------------------------------------------
# Index creation (vectorchord / pgvector HNSW)
# ---------------------------------------------------------------------------

def create_index(conn, table_name: str, config: dict, pg_parallel_workers: int = None) -> None:
    """Create the appropriate index based on config keys."""
    if pg_parallel_workers:
        conn.execute(f"SET max_parallel_maintenance_workers = {pg_parallel_workers}")
        conn.execute(f"SET max_parallel_workers = {pg_parallel_workers}")

    metric = config["metric"]

    if "lists" in config:
        # VectorChord vchordrq
        lists = config["lists"]
        metric_ops_map = {"l2": "vector_l2_ops", "euclidean": "vector_l2_ops",
                          "cos": "vector_cosine_ops", "ip": "vector_ip_ops", "dot": "vector_ip_ops"}
        ops = metric_ops_map[metric]
        sf = config.get("samplingFactor", 256)
        rq = config.get("residual_quantization", True)
        lists_str = f"[{lists[0]}, {lists[1]}]" if isinstance(lists, list) else str(lists)
        sql = (
            f"CREATE INDEX ON {table_name} USING vchordrq (embedding {ops}) "
            f"WITH (lists = '{lists_str}', sampling_factor = {sf}, "
            f"residual_quantization = {str(rq).lower()})"
        )
    elif "m" in config:
        # pgvector HNSW
        m = config["m"]
        ef = config.get("efConstruction", 128)
        metric_ops_map = {"l2": "vector_l2_ops", "euclidean": "vector_l2_ops",
                          "cos": "vector_cosine_ops", "ip": "vector_ip_ops", "dot": "vector_ip_ops"}
        ops = metric_ops_map[metric]
        sql = (
            f"CREATE INDEX ON {table_name} USING hnsw (embedding {ops}) "
            f"WITH (m = {m}, ef_construction = {ef})"
        )
    else:
        raise ValueError("Config must have 'lists' (vectorchord) or 'm' (pgvector HNSW)")

    print(f"Building index: {sql}")
    conn.execute(sql)


# ---------------------------------------------------------------------------
# Per-checkpoint benchmark
# ---------------------------------------------------------------------------

def _apply_gucs(conn, benchmark: dict) -> None:
    """Apply suite-specific GUCs from a benchmark config."""
    if "nprob" in benchmark:
        conn.execute(f'SET vchordrq.probes = \'{benchmark["nprob"]}\'')
    if "epsilon" in benchmark:
        conn.execute(f'SET vchordrq.epsilon = {benchmark["epsilon"]}')
    if "efSearch" in benchmark:
        conn.execute(f'SET hnsw.ef_search = {benchmark["efSearch"]}')
    if "probes" in benchmark:
        conn.execute(f'SET ivfflat.probes = {benchmark["probes"]}')


def run_benchmark_point(conn, table_name: str, queries: np.ndarray, gt: np.ndarray,
                        top_k: int, metric_op: str, benchmark: dict) -> dict:
    """Run a single benchmark point; return {recall, qps, p50_ms, p99_ms}."""
    _apply_gucs(conn, benchmark)
    conn.execute("SET enable_seqscan = off")

    query_sql = f"SELECT id FROM {table_name} ORDER BY embedding {metric_op} %s LIMIT {top_k}"
    cur = conn.cursor()
    latencies = []
    hits = 0

    for q_vec, gt_row in zip(queries, gt):
        t0 = time.perf_counter()
        cur.execute(query_sql, (q_vec,))
        result = cur.fetchall()
        t1 = time.perf_counter()
        latencies.append(t1 - t0)
        result_ids = {r[0] for r in result[:top_k]}
        hits += len(result_ids & set(gt_row[:top_k].tolist()))

    cur.close()
    n = len(queries)
    lat_ms = np.array(latencies) * 1000
    return {
        "recall": hits / (top_k * n),
        "qps": n / sum(latencies),
        "p50_ms": float(np.percentile(lat_ms, 50)),
        "p99_ms": float(np.percentile(lat_ms, 99)),
    }


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _print_summary(suite_name: str, all_results: list) -> None:
    print(f"\n{'=' * 72}")
    print(f"Workload Summary: {suite_name}")
    print(f"{'=' * 72}")

    # Group by benchmark name preserving insertion order
    benches: dict[str, list] = {}
    for r in all_results:
        benches.setdefault(r["benchmark"], []).append(r)

    for bench_name, rows in benches.items():
        print(f"\nBenchmark: {bench_name}")
        print(f"{'chkpt%':>7}  {'n_live':>10}  {'recall':>8}  {'recall_Δ':>9}  "
              f"{'qps':>8}  {'qps_Δ':>8}  {'p50_ms':>7}  {'p99_ms':>7}")
        print("-" * 72)
        base_recall = rows[0]["recall"]
        base_qps    = rows[0]["qps"]
        for i, row in enumerate(rows):
            if i == 0:
                r_drift = "baseline"
                q_drift = "baseline"
            else:
                r_drift = f"{(row['recall'] - base_recall) / (base_recall or 1) * 100:+.2f}%"
                q_drift = f"{(row['qps']    - base_qps)    / (base_qps    or 1) * 100:+.2f}%"
            print(f"{row['checkpoint_pct']:>7}  {row['n_live']:>10,}  "
                  f"{row['recall']:>8.4f}  {r_drift:>9}  "
                  f"{row['qps']:>8.1f}  {q_drift:>8}  "
                  f"{row['p50_ms']:>7.2f}  {row['p99_ms']:>7.2f}")


# ---------------------------------------------------------------------------
# Main workload runner
# ---------------------------------------------------------------------------

def run_workload(suite_name: str, config: dict, url: str,
                 chunk_size: int = 200_000, num_threads: int = 4) -> None:
    workload_cfg = config.get("workload", {})
    start_pct   = workload_cfg.get("start_pct", 50)
    step_pct    = workload_cfg.get("step_pct", 10)
    ins_ratio   = workload_cfg.get("insert_ratio", 1.0)
    del_ratio   = workload_cfg.get("delete_ratio", 0.0)
    autovacuum  = workload_cfg.get("autovacuum", True)
    rng_seed    = workload_cfg.get("rng_seed", 0)
    metric      = config["metric"]
    top_k       = config.get("top", 10)
    benchmarks  = config.get("benchmarks", {})
    metric_op   = _metric_op(metric)

    ds = datasets.get_dataset(config["dataset"])
    train = ds["train"]
    queries = ds["test"].astype(np.float32)
    N = ds["num"]

    table_name = suite_name.replace("-", "_")

    conn = psycopg.connect(url, autocommit=True)
    pgvector.psycopg.register_vector(conn)

    # --- Table setup ---
    conn.execute(f"DROP TABLE IF EXISTS {table_name}")
    conn.execute(f"CREATE TABLE {table_name} (id INTEGER PRIMARY KEY, embedding vector({ds['dim']}))")
    if not autovacuum:
        conn.execute(f"ALTER TABLE {table_name} SET (autovacuum_enabled = false)")
        print("autovacuum disabled for this table")

    # --- Initial load at start_pct ---
    n_start = int(N * start_pct / 100)
    print(f"\nLoading initial {start_pct}% ({n_start:,} vectors)...")
    pbar = tqdm(total=n_start, unit=" rows", ncols=80)
    _insert_range(conn, table_name, train, start_id=0, count=n_start,
                  chunk_size=chunk_size, progress=pbar)
    pbar.close()

    live_ids = np.arange(n_start, dtype=np.int32)

    # --- Build index ---
    pg_workers = config.get("pg_parallel_workers")
    print(f"Building index...")
    t0 = time.perf_counter()
    create_index(conn, table_name, config, pg_workers)
    print(f"Index built in {time.perf_counter() - t0:.1f}s")

    # --- Result collection ---
    results_path = Path(f"./results/{suite_name}/workload_results.csv")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    gt_cache_dir = results_path.parent / "gt_cache"

    csv_fields = ["checkpoint_pct", "n_live", "benchmark", "recall", "qps", "p50_ms", "p99_ms"]
    csv_file = open(results_path, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    writer.writeheader()

    all_results = []   # for end-of-run summary

    # --- Checkpoints: start_pct, start_pct+step, ..., 100 ---
    next_insert_id = n_start
    live_set = set(live_ids.tolist())
    rng = np.random.default_rng(seed=rng_seed)

    checkpoints = list(range(start_pct, 101, step_pct))
    if checkpoints[0] != start_pct:
        checkpoints.insert(0, start_pct)

    for i, checkpoint_pct in enumerate(checkpoints):
        if i > 0:
            # Compute how many rows this step covers
            prev_pct = checkpoints[i - 1]
            step_rows = int(N * (checkpoint_pct - prev_pct) / 100)
            n_insert = int(step_rows * ins_ratio)
            n_delete = int(step_rows * del_ratio)

            # Insert new vectors
            if n_insert > 0 and next_insert_id < N:
                n_insert = min(n_insert, N - next_insert_id)
                print(f"  Inserting {n_insert:,} vectors (ids {next_insert_id}–{next_insert_id+n_insert-1})...")
                pbar = tqdm(total=n_insert, unit=" rows", ncols=80)
                _insert_range(conn, table_name, train, start_id=next_insert_id,
                               count=n_insert, chunk_size=chunk_size, progress=pbar)
                pbar.close()
                for new_id in range(next_insert_id, next_insert_id + n_insert):
                    live_set.add(new_id)
                next_insert_id += n_insert

            # Delete random live vectors (deterministic via rng_seed)
            if n_delete > 0 and live_set:
                n_delete = min(n_delete, len(live_set))
                live_arr = np.array(sorted(live_set), dtype=np.int32)
                del_indices = rng.choice(len(live_arr), size=n_delete, replace=False)
                to_delete = live_arr[del_indices].tolist()
                print(f"  Deleting {n_delete:,} vectors...")
                batch = 10_000
                for start in range(0, len(to_delete), batch):
                    ids_batch = to_delete[start: start + batch]
                    placeholders = ",".join(["%s"] * len(ids_batch))
                    conn.execute(f"DELETE FROM {table_name} WHERE id IN ({placeholders})", ids_batch)
                for d_id in to_delete:
                    live_set.discard(d_id)

        live_ids = np.array(sorted(live_set), dtype=np.int32)
        n_live = len(live_ids)
        print(f"\n=== Checkpoint {checkpoint_pct}% — {n_live:,} live vectors ===")

        # Recompute GT (or load from cache)
        gt = _load_gt_cache(gt_cache_dir, checkpoint_pct, live_ids, top_k)
        if gt is not None:
            print("  Ground truth loaded from cache.")
        else:
            print("  Computing ground truth (FAISS brute-force)...")
            gt = compute_gt(live_ids, train, queries, top_k, metric)
            _save_gt_cache(gt_cache_dir, checkpoint_pct, live_ids, gt)

        # Run each benchmark point
        for bench_name, bench_cfg in benchmarks.items():
            metrics = run_benchmark_point(conn, table_name, queries, gt, top_k, metric_op, bench_cfg)
            print(f"  {bench_name:20s}  recall={metrics['recall']:.4f}  "
                  f"QPS={metrics['qps']:.1f}  P50={metrics['p50_ms']:.2f}ms")
            row = {
                "checkpoint_pct": checkpoint_pct,
                "n_live": n_live,
                "benchmark": bench_name,
                **metrics,
            }
            writer.writerow(row)
            csv_file.flush()
            all_results.append(row)

    csv_file.close()
    if not autovacuum:
        conn.execute(f"ALTER TABLE {table_name} RESET (autovacuum_enabled)")
    conn.close()
    print(f"\nResults written to {results_path}")

    _print_summary(suite_name, all_results)


def _insert_range(conn, table_name: str, train_data, start_id: int,
                  count: int, chunk_size: int, progress=None) -> None:
    """Insert train_data[start_id : start_id+count] with DB ids start_id…+count."""
    for offset in range(0, count, chunk_size):
        end = min(offset + chunk_size, count)
        chunk_len = end - offset
        chunk = train_data[start_id + offset: start_id + end]
        if chunk.dtype != np.float32:
            chunk = chunk.astype(np.float32)
        ids = np.arange(start_id + offset, start_id + end, dtype=np.int32)
        buf = pack_copy_binary_chunk(ids, chunk)
        with conn.cursor().copy(
            f"COPY {table_name} (id, embedding) FROM STDIN WITH (FORMAT BINARY)"
        ) as copy:
            copy.write(buf)
        if progress is not None:
            progress.update(chunk_len)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Workload benchmark for vector search indexes")
    parser.add_argument("-s", "--suite", required=True, help="YAML config file")
    parser.add_argument("--url", default="postgresql://postgres@localhost:5432/postgres",
                        help="Database connection URL")
    parser.add_argument("--chunk-size", type=int, default=200_000)
    parser.add_argument("--max-load-threads", type=int, default=4)
    args = parser.parse_args()

    with open(args.suite) as f:
        full_config = yaml.safe_load(f)

    for suite_name, config in full_config.items():
        if "workload" not in config:
            print(f"Skipping {suite_name}: no 'workload' key in config")
            continue
        print(f"\nRunning workload benchmark: {suite_name}")
        run_workload(suite_name, config, args.url,
                     chunk_size=args.chunk_size,
                     num_threads=args.max_load_threads)


if __name__ == "__main__":
    main()
