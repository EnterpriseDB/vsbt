# Vector Search Benchmark Suite

A comprehensive benchmarking tool for PostgreSQL vector search extensions. Compare performance across **pgvector**, **VectorChord**, and **pgpu** (GPU-accelerated) on datasets ranging from 1M to 1B vectors.

## Supported Extensions

| Extension | Index Type | Description |
|-----------|------------|-------------|
| **[pgvector](https://github.com/pgvector/pgvector)** | [HNSW](https://arxiv.org/abs/1603.09320) | Standard CPU-based approximate nearest neighbor search |
| **[vchordq](https://github.com/tensorchord/VectorChord)** | [IVF-RaBitQ](https://arxiv.org/abs/2405.12497) ([VectorChord](https://blog.vectorchord.ai/scaling-vector-search-to-1-billion-on-postgresql)) | High dimensionality & high performance vector quantization & compression |
| **[pgpu](https://github.com/EnterpriseDB/pgpu)** | IVF-RaBitQ (VectoChord) | GPU-accelerated index building for VectorChord |

## Supported Datasets

| Dataset | Vectors | Dimensions | Metric | Type |
|---------|---------|------------|--------|------|
| laion-1m-test-ip | 1M | 768 | Inner Product | HDF5 |
| laion-5m-test-ip | 5M | 768 | Inner Product | HDF5 |
| laion-20m-test-ip | 20M | 768 | Inner Product | HDF5 |
| laion-100m-test-ip | 100M | 768 | Inner Product | HDF5 |
| laion-400m-test-ip | 400M | 512 | Inner Product | NPY (multipart) |
| deep1b-test-l2 | 1B | 96 | L2 | NPY (mmap) |
| sift-128-euclidean | 1M | 128 | L2 | HDF5 |
| glove-test-cos | 1.2M | 100 | Cosine | HDF5 |

Datasets are automatically downloaded on first use.

## Installation

### Prerequisites

- Python 3.10+
- PostgreSQL 15+ with one of the supported extensions installed

### Install Dependencies

```bash
# Create a virtual environment (recommended, required on RHEL 9+ and similar)
python3.10 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

> **Note:** Some systems (e.g. RHEL 9, Fedora 38+) restrict installing packages globally with pip.
> If your system ships with Python < 3.10, install a newer version (e.g. via `dnf install python3.11`)
> and create the venv with that: `python3.11 -m venv .venv`.

### Required PostgreSQL Extensions

Depending on which benchmark suite you want to run:

```sql
-- For pgvector benchmarks
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_prewarm;

-- For VectorChord benchmarks
CREATE EXTENSION IF NOT EXISTS vchord CASCADE;

-- For PGPU benchmarks
CREATE EXTENSION IF NOT EXISTS vchord CASCADE;
CREATE EXTENSION IF NOT EXISTS pgpu;
```

## Usage

### Basic Command Structure

```bash
python <suite>.py -s config/<config>.yaml [options]
```

### Common Options

| Option | Description | Default |
|--------|-------------|---------|
| `-s, --suite` | YAML configuration file (required) | - |
| `--url` | PostgreSQL connection URL | `postgresql://postgres@localhost:5432/postgres` |
| `--devices` | Block devices to monitor | Auto-detected |
| `--chunk-size` | Chunk size for loading data | `1000000` |
| `--query-clients` | Number of parallel client sessions for querying | `1` |
| `--max-load-threads` | Threads for loading embeddings | `4` |
| `--skip-add-embeddings` | Skip data loading step | `false` |
| `--skip-index-creation` | Skip index build step | `false` |
| `--build-only` | Build index only, skip query benchmarks | `false` |
| `--no-fs-cache` | Drop filesystem cache after prewarm (requires sudo) | `false` |
| `--overwrite-table` | Drop existing table first | `false` |
| `--debug` | Enable debug logging | `false` |
| `--debug-single-query` | Repeat same query to diagnose latency issues | `false` |

### Running pgvector Benchmarks

```bash
# Run with default 5M dataset configuration
python pgvector_suite.py -s config/pgvector_suite.yaml \
    --url "postgresql://postgres@localhost:5432/postgres" \

# Skip loading if data already exists
python pgvector_suite.py -s config/pgvector_suite.yaml \
    --skip-add-embeddings \
    --url "postgresql://postgres@localhost:5432/postgres"
```

### Running VectorChord Benchmarks

```bash
# Run 20M dataset benchmark
python vectorchord_suite.py -s config/vectorchord_suite_20m.yaml \
    --url "postgresql://postgres@localhost:5432/postgres"

# Run 100M dataset benchmark
python vectorchord_suite.py -s config/vectorchord_suite_100m.yaml \
    --url "postgresql://postgres@localhost:5432/postgres"

# Run 1B dataset benchmark
python vectorchord_suite.py -s config/vectorchord_suite_1B.yaml \
    --url "postgresql://postgres@localhost:5432/postgres"
```

### Running PGPU (GPU-Accelerated) Benchmarks

```bash
# Run GPU-accelerated index build with 5M dataset
python pgpu_suite.py -s config/pgpu_suite_5m.yaml \
    --url "postgresql://postgres@localhost:5432/postgres"

# Run with 100M dataset
python pgpu_suite.py -s config/pgpu_suite_100m.yaml \
    --url "postgresql://postgres@localhost:5432/postgres"
```

### Using External Centroids

For VectorChord and PGPU, you can provide pre-computed centroids:

```bash
# Using a centroids file
python vectorchord_suite.py -s config/vectorchord_suite_100m.yaml \
    --centroids-file centroids.npy \
    --url "postgresql://postgres@localhost:5432/postgres"

# Using an existing centroids table
python vectorchord_suite.py -s config/vectorchord_suite_100m.yaml \
    --centroids-table public.my_centroids \
    --url "postgresql://postgres@localhost:5432/postgres"
```

### Parallel Query Benchmarking

Run queries in parallel to measure throughput under load:

```bash
python pgvector_suite.py -s config/pgvector_suite.yaml \
    --query-clients 8 \
    --skip-add-embeddings \
    --skip-index-creation
```

## Configuration Files

### pgvector Configuration Example

```yaml
pgvector-laion-5m-m16-256:
  dataset: laion-5m-test-ip
  datasetType: hdf5
  pg_parallel_workers: 63  # PostgreSQL parallel workers for index build
  metric: dot
  m: 16                    # HNSW M parameter
  efConstruction: 256      # HNSW ef_construction
  top: 10                  # Top-K results
  benchmarks:
    "40": { efSearch: 40 }
    "80": { efSearch: 80 }
    "200": { efSearch: 200 }
```

### VectorChord Configuration Example

```yaml
vc-laion-5m-8192-test-ip:
  dataset: laion-5m-test-ip
  datasetType: hdf5
  metric: ip
  lists: [90, 8192]        # Hierarchical IVF lists
  samplingFactor: 64
  pg_parallel_workers: 63  # PostgreSQL parallel workers for index build
  kmeans_hierarchical: true
  kmeans_dimension: 100
  residual_quantization: true
  top: 10
  benchmarks:
    20-1.0:
      nprob: 2,20          # Probes per level
      epsilon: 1.0
    50-1.5:
      nprob: 16,50
      epsilon: 1.5
```

### PGPU Configuration Example

```yaml
pgpu-laion-100m-160000-test-ip:
  dataset: laion-100m-test-ip
  datasetType: hdf5
  metric: dot
  lists: [500, 160000]     # Hierarchical IVF lists
  samplingFactor: 256
  batchSize: 10000000
  pg_parallel_workers: 63  # PostgreSQL parallel workers for index build
  residual_quantization: true
  top: 10
  benchmarks:
    50-1.0:
      nprob: 50,100
      epsilon: 1.0
```

## Output

Results are organized per test name, with each test accumulating results across multiple runs:

```
results/
├── all_results.csv                        # Global CSV (append-only, all runs)
├── {test_name}/                           # One folder per YAML test name
│   ├── report.md                          # Incremental report (all runs for this test)
│   ├── runs/                              # Raw data per run
│   │   └── {run_id}.json
│   ├── charts/                            # Charts (latest run)
│   │   ├── recall_vs_qps.png
│   │   ├── latency.png
│   │   ├── build_times.png
│   │   └── system_dashboard.png
│   └── index_build/                       # Index build monitoring data
└── comparisons/                           # Cross-test comparison charts
    ├── recall_vs_qps_{timestamp}.png
    └── recall_vs_p99_{timestamp}.png
```

The `report.md` is **incremental** — it is regenerated from all stored run JSONs each time a benchmark completes. Running the same test with different `shared_buffers` or `--no-fs-cache` adds a new run section to the existing report, building a complete history.

### Generated Reports

Each benchmark run generates:

| Output | Description |
|--------|-------------|
| **Incremental Report** | `report.md` with system info, configuration, build metrics across all runs, and benchmark tables per run |
| **Recall vs QPS Chart** | Scatter plot showing recall/throughput tradeoff |
| **Latency Chart** | Bar chart comparing P50/P99 latencies across configs |
| **Build Time Chart** | Horizontal bar showing load/clustering/index build breakdown |
| **Raw JSON** | Complete results data per run for programmatic access |
| **Consolidated CSV** | Append-only CSV with shared_buffers, maintenance_work_mem, and fs_cache columns |

### Comparing Runs

Use `chart_compare.py` to generate comparison charts across different runs:

```bash
# List all available runs
python chart_compare.py --list

# Compare two specific runs by ID
python chart_compare.py --runs 20260309010000 20260309020000

# Compare latest runs for specific tests (e.g., pgvector vs vectorchord)
python chart_compare.py --tests pgvector-laion-1B-m16-128 vc-deep1B-test-l2

# Filter by shared_buffers size
python chart_compare.py --tests pgvector-... vc-... --sb 700GB

# Filter by cache mode
python chart_compare.py --tests pgvector-... vc-... --cache-mode with
```

Generates Recall vs QPS and Recall vs P99 charts with labeled data points (efSearch or nprob/epsilon) and distinct colors/markers per series.

### Build-Only and No-FS-Cache Modes

```bash
# Build the index without running query benchmarks
python pgvector_suite.py -s config/pgvector_suite_1B.yaml --build-only --skip-add-embeddings

# Run benchmarks with filesystem cache dropped after prewarm
# (isolates shared_buffers performance from OS page cache)
python pgvector_suite.py -s config/pgvector_suite_1B.yaml --no-fs-cache --skip-add-embeddings --skip-index-creation
```

### Metrics Reported

| Metric | Description |
|--------|-------------|
| **Recall@K** | Fraction of true nearest neighbors found |
| **QPS** | Queries per second |
| **P50 Latency** | Median query latency (ms) |
| **P99 Latency** | 99th percentile latency (ms) |
| **Index Size** | On-disk index size |
| **Build Time** | Total time to create the index |
| **Clustering Time** | Time spent on k-means clustering (PGPU only) |
| **Load Time** | Time to load embeddings into database |

## Utility Scripts

### Compare Benchmark Runs

Compare results between different benchmark runs:

```bash
# List all available runs
python compare_runs.py --list

# Show details for a specific run
python compare_runs.py --show 5

# Compare two runs
python compare_runs.py --compare 3 7

# Export comparison as CSV
python compare_runs.py --compare 3 7 --format csv
```

### Convert Deep1B Dataset

Convert Deep1B binary files to NPY format:

```bash
cd utils
python convert_deep1b.py
```

### Verify Deep1B Files

Check integrity of downloaded Deep1B files:

```bash
cd utils
python verify_deep1B.py
```

## Project Structure

```
vector-search/
├── common.py                 # Base test suite class
├── datasets.py               # Dataset download and loading
├── results.py                # Results management and visualization
├── compare_runs.py           # Historical benchmark comparison utility
├── chart_compare.py          # Cross-run comparison chart generator
├── pgvector_suite.py         # pgvector HNSW benchmarks
├── vectorchord_suite.py      # VectorChord IVF benchmarks
├── pgpu_suite.py             # GPU-accelerated benchmarks
├── requirements.txt          # Python dependencies
├── config/                   # Benchmark configurations
│   ├── pgvector_suite.yaml
│   ├── pgpu_suite_5m.yaml
│   ├── pgpu_suite_100m.yaml
│   ├── vectorchord_suite_100m.yaml
│   ├── vectorchord_suite_1B.yaml
│   └── ...
├── monitor/
│   ├── __init__.py           # Monitor package
│   ├── system_monitor.py     # System metrics (psutil-based)
│   └── pg_stats.py           # PostgreSQL statistics collector
├── utils/
│   ├── convert_deep1b.py     # Deep1B format converter
│   └── verify_deep1B.py      # File integrity checker
└── results/                  # Output directory (generated)
    ├── all_results.csv       # Global CSV (all runs)
    ├── {test_name}/          # Per-test results
    │   ├── report.md         # Incremental report
    │   ├── runs/*.json       # Raw data per run
    │   └── charts/*.png      # Charts
    └── comparisons/          # Cross-test comparison charts
```

## pgvector: Memory Tuning for Large HNSW Index Builds

When building HNSW indexes on large tables (hundreds of millions to billions of rows), PostgreSQL's memory management can work against you. Understanding how memory is used during an index build is critical to avoiding I/O thrashing.

### The Problem: `shared_buffers` Goes Unused

PostgreSQL uses a **Buffer Access Strategy** (specifically `BAS_BULKREAD`) for large sequential scans, including the heap scan that feeds an HNSW index build. This strategy restricts the scan to a small ring buffer within `shared_buffers`, preventing it from evicting hot pages that other queries might need.

This is great for busy OLTP systems — but on a dedicated machine building an index, it means your `shared_buffers` sits almost entirely empty while the build runs. The heap data flows through a tiny window and is immediately discarded from PostgreSQL's perspective.

Meanwhile, `maintenance_work_mem` holds the HNSW graph being constructed in memory. This is anonymous memory that the OS cannot reclaim.

### Estimating Graph Memory

The HNSW graph is held in `maintenance_work_mem` during the build. Each node at level L consumes:

```
MAXALIGN(~128 bytes)              HnswElementData struct
+ MAXALIGN(8 + 4 × dim)          vector value (varlena header + floats)
+ MAXALIGN(8 × (L+1))            neighbor list pointers
+ MAXALIGN(8 + 32 × m)           layer 0 neighbor array  (2×m candidates × 16 bytes)
+ L × MAXALIGN(8 + 16 × m)      upper layer arrays       (m candidates × 16 bytes each)
```

Levels are assigned randomly with `P(level ≥ L) = (1/m)^L`, so the expected upper-layer overhead per node is `1/(m−1) × (8 + MAXALIGN(8 + 16×m))`.

`ef_construction` does **not** affect graph memory — it only adds transient per-worker search buffers that are freed after each tuple insertion.

**Quick estimates** (average bytes per node including upper-layer overhead):

| dim | m=16 | m=32 | m=64 |
|-----|------|------|------|
| 96  | ~1,066 | ~1,577 | ~2,601 |
| 768 | ~3,754 | ~4,265 | ~5,289 |

For example, 1B vectors with dim=96 and m=16 requires ~**993 GB** of `maintenance_work_mem`. The benchmark suite prints this estimate before starting the index build.

If the graph exceeds `maintenance_work_mem`, pgvector flushes the in-memory graph to disk and switches to a much slower on-disk insertion mode for remaining tuples.

### Estimating On-Disk Index Size

The on-disk HNSW index is approximately **65% of the in-memory graph size**. This ratio is useful for planning `shared_buffers` for query serving:

```
on-disk index ≈ 0.65 × in-memory graph
```

For example, 1B vectors with dim=96 and m=16: 0.65 × 993 GB ≈ **645 GB** (observed: 646 GB).

### Sizing `shared_buffers` for Query Serving

For best query performance, the entire index should fit in `shared_buffers`. If it doesn't, avoid the middle ground — due to double buffering between `shared_buffers` and OS page cache, a partially-filled `shared_buffers` performs **worse** than a tiny one.

Benchmark results on a 1B vector HNSW index (646 GB) at various `shared_buffers` sizes:

| shared_buffers | efSearch=200 QPS | efSearch=800 QPS | Notes |
|---|---|---|---|
| 16GB | 59.04 | 17.79 | Index in OS page cache — good |
| 128GB | 37.65 | 12.85 | Double buffering — worst |
| 256GB | 29.61 | 14.37 | Double buffering — worst |
| 700GB | 95.14 | 36.52 | Index fully in shared_buffers — best |

When `shared_buffers` is too small to hold the index, PostgreSQL and the OS cache overlapping copies of the same pages, wasting RAM. With 16GB, nearly all RAM goes to page cache and the index fits. With 700GB, shared_buffers holds the full index with faster access than page cache. The middle ground wastes RAM on double copies and neither cache is large enough.

### Example

Consider a machine with 512GB of RAM building an HNSW index on a 200GB table:

```
shared_buffers       = 128GB   (pinned, but nearly empty during the build)
maintenance_work_mem = 256GB   (pinned, holds the HNSW graph)
─────────────────────────────
Total pinned         = 384GB
Available for OS page cache = ~100GB  (512 - 384 - OS overhead)
Table on disk        = 200GB
```

The table doesn't fit in the remaining page cache. The OS constantly evicts and re-reads pages, `kswapd` runs at 100% CPU trying to reclaim memory, and most parallel workers end up blocked on `DataFileRead` — waiting for disk instead of doing useful work.

You can verify this during a build with `pg_buffercache`:

```sql
-- Check what's actually in shared_buffers
CREATE EXTENSION IF NOT EXISTS pg_buffercache;

SELECT c.relname, count(*) AS buffers,
       pg_size_pretty(count(*) * 8192::bigint) AS size
FROM pg_buffercache b
JOIN pg_class c ON b.relfilenode = c.relfilenode
WHERE b.reldatabase = (SELECT oid FROM pg_database WHERE datname = current_database())
GROUP BY c.relname ORDER BY count(*) DESC LIMIT 10;
```

If you see your table occupying only a few MB out of many GB of `shared_buffers`, that confirms the ring buffer strategy is active.

### The Fix: Lower `shared_buffers` for Index Builds

Since the heap scan bypasses `shared_buffers`, that memory is better given to the OS page cache, which will happily cache the entire table without any ring buffer restriction.

For the example above, setting `shared_buffers = 8GB` changes the picture:

```
shared_buffers       = 8GB
maintenance_work_mem = 256GB
─────────────────────────────
Total pinned         = 264GB
Available for OS page cache = ~220GB  (512 - 264 - OS overhead)
Table on disk        = 200GB   ← now fits entirely in page cache
```

The entire table stays cached, `kswapd` goes quiet, parallel workers stop waiting on disk, and the build runs significantly faster — despite *lower* PostgreSQL settings.

### Recommendations

- **Before the index build:** temporarily lower `shared_buffers` (e.g., 8–16GB) and restart PostgreSQL.
- **After the build:** raise `shared_buffers` back to its normal value for query serving, where it will be used effectively.
- **`maintenance_work_mem`** should be sized to fit the HNSW graph. If it's too small the build will spill to disk; if it's too large it starves page cache.
- **`max_parallel_maintenance_workers`** — more workers than your storage can feed just adds I/O contention. Monitor `pg_stat_activity` for `DataFileRead` waits and reduce workers if most are blocked on I/O.

> **Note:** This applies to PostgreSQL through version 17. PostgreSQL 18 introduces `io_method=io_uring` with Direct I/O support, which bypasses the OS page cache entirely and may change these trade-offs.

## Contributors

- **Alessandro Ferraresi** - Initial creator
- **Huan Zhang**
- **Tim Waizenegger**

## License

See LICENSE file for details.
