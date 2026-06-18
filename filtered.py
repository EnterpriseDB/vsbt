"""
filtered.py — Filtered ANN benchmark runner.

Benchmarks vector search with a scalar pre-filter:
  SELECT id FROM t WHERE filter_label = 1
  ORDER BY embedding <op> $1 LIMIT K

Key behaviour by extension:
  pgvector (0.8+): SET hnsw.iterative_scan = relaxed_order
    Extends the HNSW graph walk until K qualifying results are found.
    Without this, recall collapses at low selectivity (<5%).
  VectorChord:     SET vchordrq.prefilter = on
    Builds a bitvector of qualifying rows before the IVF scan.

Invoked automatically from common.py when mode: filtered appears in a
YAML config. Can also be run standalone for debugging.

YAML example (see README for full reference):

  pgvector-laion-5m-neutral-1pct:
    mode: filtered
    dataset: laion-5m-filtered-neutral
    selectivity: 1.0
    metric: ip
    pg_parallel_workers: 32
    top: 10
    m: 16
    efConstruction: 128
    benchmarks:
      ef40:  { efSearch: 40 }
      ef80:  { efSearch: 80 }
      ef200: { efSearch: 200 }

  vc-laion-5m-neutral-1pct:
    mode: filtered
    dataset: laion-5m-filtered-neutral
    selectivity: 1.0
    metric: ip
    lists: [50, 8000]
    samplingFactor: 256
    residual_quantization: true
    pg_parallel_workers: 32
    top: 10
    benchmarks:
      10-20-1.9: { nprob: "10,20", epsilon: 1.9 }
"""

import csv
import time
from pathlib import Path

import numpy as np
import psycopg
import pgvector.psycopg
from tqdm import tqdm

import datasets
from loader import load_vectors_with_labels


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

_METRIC_OPS = {
    "l2": "<->", "euclidean": "<->",
    "cos": "<=>",
    "ip":  "<#>", "dot": "<#>",
}

_METRIC_INDEX_OPS = {
    "l2": "vector_l2_ops", "euclidean": "vector_l2_ops",
    "cos": "vector_cosine_ops",
    "ip":  "vector_ip_ops",  "dot": "vector_ip_ops",
}


def _detect_suite(config: dict) -> str:
    """Return 'pgvector' or 'vectorchord' based on config keys."""
    if "m" in config:
        return "pgvector"
    if "lists" in config:
        return "vectorchord"
    raise ValueError(
        "Cannot detect suite type: config must have 'm' (pgvector HNSW) "
        "or 'lists' (VectorChord)."
    )


# ---------------------------------------------------------------------------
# Table + index creation
# ---------------------------------------------------------------------------

def _create_table(conn, table_name: str, dim: int, pg_parallel_workers: int = None):
    conn.execute(f"DROP TABLE IF EXISTS {table_name}")
    conn.execute(
        f"CREATE TABLE {table_name} "
        f"(id INTEGER, embedding vector({dim}), filter_label SMALLINT)"
    )
    if pg_parallel_workers:
        conn.execute(
            f"ALTER TABLE {table_name} "
            f"SET (parallel_workers = {pg_parallel_workers})"
        )


def _create_index(conn, table_name: str, config: dict, suite: str,
                  pg_parallel_workers: int = None):
    metric   = config["metric"]
    ops      = _METRIC_INDEX_OPS[metric]
    mwm      = config.get("maintenance_work_mem")
    workers  = pg_parallel_workers or config.get("pg_parallel_workers")

    if workers:
        conn.execute(f"SET max_parallel_maintenance_workers = {workers}")
        conn.execute(f"SET max_parallel_workers = {workers}")
    if mwm:
        conn.execute(f"SET maintenance_work_mem = '{mwm}'")

    print("Building vector index...", flush=True)
    t0 = time.perf_counter()

    if suite == "pgvector":
        m   = config["m"]
        efc = config.get("efConstruction", 128)
        conn.execute(
            f"CREATE INDEX ON {table_name} USING hnsw (embedding {ops}) "
            f"WITH (m = {m}, ef_construction = {efc})"
        )
    else:  # vectorchord
        lists = config["lists"]
        sf    = config.get("samplingFactor", 256)
        rq    = str(config.get("residual_quantization", True)).lower()
        lstr  = f"[{lists[0]}, {lists[1]}]" if isinstance(lists, list) else str(lists)
        spherical = "true" if metric in ("cos", "ip", "dot") else "false"
        ivf_config = f"""
residual_quantization = {rq}
build.pin = 2

[build.internal]
lists = {lstr}
sampling_factor = {sf}
spherical_centroids = {spherical}
"""
        conn.execute(
            f"CREATE INDEX ON {table_name} USING vchordrq (embedding {ops}) "
            f"WITH (options = $${ivf_config}$$)"
        )

    elapsed = time.perf_counter() - t0
    print(f"  done in {elapsed:.1f}s")

    # B-tree on filter_label so the planner can use an index scan for the WHERE
    conn.execute(f"CREATE INDEX ON {table_name} (filter_label)")


# ---------------------------------------------------------------------------
# Per-benchmark GUCs
# ---------------------------------------------------------------------------

def _apply_gucs(conn, benchmark: dict, suite: str):
    if suite == "pgvector":
        ef = benchmark.get("efSearch", 40)
        conn.execute(f"SET hnsw.ef_search = {ef}")
        conn.execute("SET hnsw.iterative_scan = relaxed_order")
        mst = benchmark.get("max_scan_tuples")
        if mst:
            conn.execute(f"SET hnsw.max_scan_tuples = {mst}")
        conn.execute("SET enable_seqscan = off")
    else:  # vectorchord
        nprob   = benchmark.get("nprob", "10,20")
        epsilon = benchmark.get("epsilon", 1.0)
        conn.execute(f"SET vchordrq.probes = '{nprob}'")
        conn.execute(f"SET vchordrq.epsilon = {epsilon}")
        conn.execute("SET vchordrq.prefilter = on")
        conn.execute("SET enable_seqscan = off")


# ---------------------------------------------------------------------------
# Benchmark execution
# ---------------------------------------------------------------------------

def _run_benchmark_point(conn, table_name: str, queries: np.ndarray,
                         gt: np.ndarray, top_k: int, metric_op: str,
                         benchmark: dict, suite: str) -> dict:
    _apply_gucs(conn, benchmark, suite)

    query_sql = (
        f"SELECT id FROM {table_name} "
        f"WHERE filter_label = 1 "
        f"ORDER BY embedding {metric_op} %s LIMIT {top_k}"
    )
    cur = conn.cursor()
    latencies, hits = [], 0

    for q_vec, gt_row in zip(queries, gt):
        t0 = time.perf_counter()
        cur.execute(query_sql, (q_vec,))
        result = cur.fetchall()
        t1 = time.perf_counter()
        latencies.append(t1 - t0)

        result_ids = {r[0] for r in result[:top_k]}
        gt_ids     = set(gt_row[:top_k].tolist())
        # Exclude -1 padding (queries with fewer than top_k true neighbors)
        gt_ids.discard(-1)
        if gt_ids:
            hits += len(result_ids & gt_ids) / len(gt_ids)

    cur.close()
    n      = len(queries)
    lat_ms = np.array(latencies) * 1000
    return {
        "recall": hits / n,
        "qps":    n / sum(latencies),
        "p50_ms": float(np.percentile(lat_ms, 50)),
        "p99_ms": float(np.percentile(lat_ms, 99)),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_filtered(suite_name: str, config: dict, url: str,
                 chunk_size: int = 500_000, num_threads: int = 4) -> None:
    selectivity      = config.get("selectivity")
    if selectivity is None:
        raise ValueError(f"{suite_name}: filtered mode requires a 'selectivity' key")

    metric           = config["metric"]
    top_k            = config.get("top", 10)
    pg_workers       = config.get("pg_parallel_workers")
    benchmarks       = config.get("benchmarks", {})
    suite            = _detect_suite(config)
    metric_op        = _METRIC_OPS[metric]
    table_name       = suite_name.replace("-", "_").replace(".", "_")

    print(f"\n[filtered] {suite_name}  suite={suite}  "
          f"selectivity={selectivity}%  metric={metric}")

    # --- Load dataset ---
    ds = datasets.get_dataset(config["dataset"], selectivity=selectivity)
    train        = ds["train"]
    queries      = ds["test"].astype(np.float32)
    gt           = ds["neighbors"]          # (Q, top_k) — pre-computed filtered GT
    filter_labels = ds["filter_labels"]     # (N,) int8 — 1 = passes filter
    N, D         = ds["num"], ds["dim"]

    if filter_labels is None:
        raise ValueError(
            f"{suite_name}: dataset '{config['dataset']}' has no filter_labels "
            f"at selectivity={selectivity}%"
        )

    print(f"  N={N:,}  D={D}  Q={queries.shape[0]}  "
          f"live={int(filter_labels.sum()):,} ({float(filter_labels.mean())*100:.2f}%)")

    conn = psycopg.connect(url, autocommit=True)
    pgvector.psycopg.register_vector(conn)

    # --- Table ---
    _create_table(conn, table_name, D, pg_workers)

    # --- Load vectors + labels ---
    print("Loading vectors with filter labels...")
    pbar = tqdm(total=N, unit=" rows", ncols=80)
    load_vectors_with_labels(
        conn_factory=lambda: psycopg.connect(url, autocommit=True),
        table_name=table_name,
        data=train,
        labels=filter_labels,
        n=N,
        chunk_size=chunk_size,
        num_threads=num_threads,
        progress=pbar,
    )
    pbar.close()

    # --- Index ---
    _create_index(conn, table_name, config, suite, pg_workers)

    # --- Prewarm ---
    try:
        conn.execute("SELECT pg_prewarm(%s)", (table_name + "_embedding_idx",))
    except Exception:
        pass

    # --- Benchmarks ---
    results_path = Path(f"./results/{suite_name}/filtered_results.csv")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    csv_fields = ["selectivity_pct", "benchmark", "recall", "qps", "p50_ms", "p99_ms"]

    with open(results_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
        writer.writeheader()

        print(f"\nRunning {len(benchmarks)} benchmark point(s)...")
        for bench_name, bench_cfg in benchmarks.items():
            metrics = _run_benchmark_point(
                conn, table_name, queries, gt, top_k, metric_op, bench_cfg, suite
            )
            print(f"  {bench_name:20s}  recall={metrics['recall']:.4f}  "
                  f"QPS={metrics['qps']:.1f}  P50={metrics['p50_ms']:.2f}ms")
            writer.writerow({"selectivity_pct": selectivity,
                             "benchmark": bench_name, **metrics})
            csv_file.flush()

    conn.close()
    print(f"\nResults → {results_path}")
