# Vector Search Benchmark Suite

A comprehensive benchmarking tool for PostgreSQL vector search extensions. Compare performance across **pgvector**, **VectorChord**, and **pgpu** (GPU-accelerated) on datasets ranging from 1M to 1B vectors.

## Supported Extensions

| Extension | Index Type | Description |
|-----------|------------|-------------|
| **[pgvector](https://github.com/pgvector/pgvector)** | [HNSW](https://arxiv.org/abs/1603.09320) | Standard CPU-based approximate nearest neighbor search |
| **[vchordq](https://github.com/tensorchord/VectorChord)** | [IVF-RaBitQ](https://arxiv.org/abs/2405.12497) ([VectorChord](https://blog.vectorchord.ai/scaling-vector-search-to-1-billion-on-postgresql)) | High dimensionality & high performance vector quantization & compression |
| **[pgpu](https://github.com/EnterpriseDB/pgpu)** | IVF-RaBitQ (VectorChord) | GPU-accelerated index building for VectorChord |

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
| glove-100-angular | 1.2M | 100 | Cosine | HDF5 |
| gist-960-euclidean | 1M | 960 | L2 | HDF5 |

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
python <suite>.py -s config/<dataset>/<config>.yaml [options]
```

Configs are grouped by dataset (one directory per dataset, e.g. `config/sift-128-euclidean/`, `config/laion-5m-test-ip/`). Each directory contains extension-specific YAMLs (`pgvector-*.yaml`, `vectorchord-*.yaml`, `pgpu.yaml`).

### Common Options

| Option | Description | Default |
|--------|-------------|---------|
| `-s, --suite` | YAML configuration file (required) | - |
| `--url` | PostgreSQL connection URL | `postgresql://postgres@localhost:5432/postgres` |
| `--devices` | Block devices to monitor | Auto-detected |
| `--chunk-size` | Chunk size for loading data | `1000000` |
| `--query-clients` | Number of parallel client sessions for querying | `1` |
| `--max-queries` | Limit number of queries per benchmark | Full test set |
| `--max-load-threads` | Threads for loading embeddings | `4` |
| `--skip-add-embeddings` | Skip data loading step | `false` |
| `--skip-index-creation` | Skip index build step | `false` |
| `--build-only` | Build index only, skip query benchmarks | `false` |
| `--overwrite-table` | Drop existing table first | `false` |
| `--debug` | Enable debug logging | `false` |
| `--debug-single-query` | Repeat same query to diagnose latency issues | `false` |

### Running pgvector Benchmarks

```bash
# Run with default 5M dataset configuration
python pgvector_suite.py -s config/laion-5m-test-ip/pgvector-m16-128.yaml

# Skip loading if data already exists
python pgvector_suite.py -s config/laion-5m-test-ip/pgvector-m16-128.yaml --skip-add-embeddings
```

### Running VectorChord Benchmarks

```bash
# Run 100M dataset benchmark
python vectorchord_suite.py -s config/laion-100m-test-ip/vectorchord-570-320k.yaml

# Run 1B dataset benchmark
python vectorchord_suite.py -s config/deep1b-test-l2/vectorchord-800-640k.yaml
```

### Running PGPU (GPU-Accelerated) Benchmarks

```bash
# Run GPU-accelerated index build with 5M dataset
python pgpu_suite.py -s config/laion-5m-test-ip/pgpu.yaml

# Run with 100M dataset
python pgpu_suite.py -s config/laion-100m-test-ip/pgpu.yaml
```

### Using External Centroids

For VectorChord and PGPU, you can provide pre-computed centroids:

```bash
# Using a centroids file
python vectorchord_suite.py -s config/laion-100m-test-ip/vectorchord-570-320k.yaml \
    --centroids-file centroids.npy

# Using an existing centroids table
python vectorchord_suite.py -s config/laion-100m-test-ip/vectorchord-570-320k.yaml \
    --centroids-table public.my_centroids
```

### Parallel Query Benchmarking

Run queries in parallel to measure throughput under load:

```bash
python pgvector_suite.py -s config/laion-5m-test-ip/pgvector-m16-128.yaml \
    --query-clients 32 \
    --skip-add-embeddings \
    --skip-index-creation
```

## Automated Benchmark Script

`utils/run_benchmarks.sh` automates running benchmarks across simulated server sizes using huge pages to constrain available memory. This provides realistic performance data for different production server configurations without needing separate hardware.

`utils/run_full_pipeline.sh` orchestrates the complete workflow: build index, benchmark across server tiers, and park the index — for each configuration. It chains `run_benchmarks.sh` internally and handles PostgreSQL restarts between build and query phases.

### Usage

```bash
# Run benchmarks across server tiers for a single config
utils/run_benchmarks.sh <config.yaml> [5m|100m|1b|SB:RAM,...] [query-clients] [max-queries]

# Run the full build → benchmark → park pipeline for all configs
utils/run_full_pipeline.sh vectorchord   # All VectorChord configs
utils/run_full_pipeline.sh pgvector      # All pgvector configs
utils/run_full_pipeline.sh all           # Everything
```

### Server Tier Presets

Each preset defines a series of simulated servers with appropriate shared_buffers and filesystem cache:

**5M** (19-21 GB indexes): 8 GB through 256 GB servers
```bash
utils/run_benchmarks.sh config/laion-5m-test-ip/pgvector-m16-128.yaml 5m
```

**100M** (367-405 GB indexes): 64 GB through full machine
```bash
utils/run_benchmarks.sh config/laion-100m-test-ip/vectorchord-570-320k.yaml 100m
```

**1B** (469-646 GB indexes): 128 GB through full machine (750 GB shared_buffers)
```bash
utils/run_benchmarks.sh config/deep1b-test-l2/pgvector-m16-128.yaml 1b
```

### Multiple Client Counts

Run both single-client and parallel benchmarks in one pass:

```bash
# Run with 1 and 32 clients at each server tier
utils/run_benchmarks.sh config/deep1b-test-l2/pgvector-m16-128.yaml 1b 1,32 1000
```

### Custom Server Tiers

Specify custom `SB:RAM` pairs (shared_buffers GB : total RAM GB):

```bash
utils/run_benchmarks.sh config/laion-100m-test-ip/pgvector-m16-128.yaml 32:128,64:256,128:512
```

### How It Works

For each server tier, the script:
1. Releases any previous huge page allocations
2. Configures `shared_buffers` via `ALTER SYSTEM`
3. Stops PostgreSQL
4. Allocates huge pages to lock away excess memory (simulating a smaller server)
5. Starts PostgreSQL with the constrained memory
6. Runs benchmarks for each client count
7. Moves to the next tier

The last tier in each preset uses the full machine without memory constraints.

## Configuration Files

### pgvector Configuration Example

```yaml
pgvector-laion-5m-m16-128:
  dataset: laion-5m-test-ip
  datasetType: hdf5
  pg_parallel_workers: 32
  metric: dot
  m: 16
  efConstruction: 128
  top: 10
  benchmarks:
    "20": { efSearch: 20 }
    "40": { efSearch: 40 }
    "58": { efSearch: 58 }
    "80": { efSearch: 80 }
    "120": { efSearch: 120 }
    "200": { efSearch: 200 }
```

### VectorChord Configuration Example

```yaml
vc-laion-5m-190-35k:
  dataset: laion-5m-test-ip
  datasetType: hdf5
  metric: dot
  samplingFactor: 256
  lists: [190, 35000]
  build_threads: 32
  residual_quantization: true
  kmeans_hierarchical: true
  pg_parallel_workers: 32
  top: 10
  benchmarks:
    15-30-1.9:
      nprob: 15,30
      epsilon: 1.9
    38-76-1.9:
      nprob: 38,76
      epsilon: 1.9
```

### PGPU Configuration Example

```yaml
pgpu-laion-100m-160000-test-ip:
  dataset: laion-100m-test-ip
  datasetType: hdf5
  metric: dot
  lists: [500, 160000]
  samplingFactor: 256
  batchSize: 10000000
  pg_parallel_workers: 32
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
│   ├── report.md                          # Aggregated report (build metrics + all benchmark results)
│   ├── runs/                              # Raw data per run
│   │   └── {test_name}_{run_id}/
│   │       ├── run.json
│   │       └── report.md
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

The `report.md` is **incremental** — it is regenerated from all stored run JSONs each time a benchmark completes. Running the same test with different `shared_buffers` or client counts adds new rows to the existing report.

### Generated Reports

Each benchmark run generates:

| Output | Description |
|--------|-------------|
| **Aggregated Report** | `report.md` with build metrics and benchmark results across all runs |
| **Recall vs QPS Chart** | Scatter plot showing recall/throughput tradeoff |
| **Latency Chart** | Bar chart comparing P50/P99 latencies across configs |
| **Build Time Chart** | Horizontal bar showing load/clustering/index build breakdown |
| **Raw JSON** | Complete results data per run for programmatic access |
| **Consolidated CSV** | Append-only CSV with all benchmark parameters and results |

### Comparing Runs

Use `chart_compare.py` to generate comparison charts across different runs:

```bash
# List all available runs
python chart_compare.py --list

# Compare two specific runs by ID
python chart_compare.py --runs 20260309010000 20260309020000

# Compare latest runs for specific tests (e.g., pgvector vs vectorchord)
python chart_compare.py --tests pgvector-laion-5m-m16-128 vc-laion-5m-190-35k

# Filter by shared_buffers size
python chart_compare.py --tests pgvector-... vc-... --sb 32GB
```

### Build-Only Mode

```bash
# Build the index without running query benchmarks
python pgvector_suite.py -s config/deep1b-test-l2/pgvector-m16-128.yaml --build-only --skip-add-embeddings
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
├── config/                   # Benchmark configurations, grouped by dataset
│   ├── sift-128-euclidean/
│   │   ├── pgvector-m16-64.yaml
│   │   └── vectorchord-32-2k.yaml
│   ├── glove-100-angular/
│   ├── gist-960-euclidean/
│   ├── laion-5m-test-ip/
│   │   ├── pgvector-m16-64.yaml
│   │   ├── pgvector-m16-128.yaml
│   │   ├── vectorchord-50-8k.yaml
│   │   ├── vectorchord-190-35k.yaml
│   │   └── pgpu.yaml
│   ├── laion-20m-test-ip/
│   ├── laion-100m-test-ip/
│   ├── laion-400m-test-ip/
│   └── deep1b-test-l2/
├── monitor/
│   ├── __init__.py           # Monitor package
│   ├── system_monitor.py     # System metrics (psutil-based)
│   └── pg_stats.py           # PostgreSQL statistics collector
├── utils/
│   ├── run_benchmarks.sh     # Automated benchmark runner with server simulation
│   ├── run_full_pipeline.sh  # Full build → benchmark → park pipeline
│   ├── convert_deep1b.py     # Deep1B format converter
│   └── verify_deep1B.py      # File integrity checker
└── results/                  # Output directory (generated)
    ├── all_results.csv       # Global CSV (all runs)
    ├── {test_name}/          # Per-test results
    │   ├── report.md         # Aggregated report
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

### Prewarming for Query Benchmarks

For consistent benchmark results, the **index** should be warmed before running queries:

- **Index → shared_buffers:** `pg_prewarm('index_name')` loads index pages into shared_buffers (default `'buffer'` mode). For VectorChord, use `vchordrq_prewarm()` instead.

HNSW queries need the heap for distance reranking — every candidate node requires a vector lookup from the heap. If the heap is cold (not in page cache), queries hit disk on these lookups, reducing QPS. The heap warms organically during the first queries.

The benchmark suite handles index prewarming automatically. When using the `utils/run_benchmarks.sh` script with simulated server sizes (via huge pages), the filesystem cache is naturally constrained to realistic levels — no manual cache management needed.

### Recommendations

- **Before the index build:** temporarily lower `shared_buffers` (e.g., 8–16GB) and restart PostgreSQL.
- **After the build:** raise `shared_buffers` back to its normal value for query serving, where it will be used effectively.
- **`maintenance_work_mem`** should be sized to fit the HNSW graph. If it's too small the build will spill to disk; if it's too large it starves page cache.
- **`max_parallel_maintenance_workers`** — more workers than your storage can feed just adds I/O contention. Monitor `pg_stat_activity` for `DataFileRead` waits and reduce workers if most are blocked on I/O.

> **Note:** This applies to PostgreSQL through version 17. PostgreSQL 18 introduces `io_method=io_uring` with Direct I/O support, which bypasses the OS page cache entirely and may change these trade-offs.

### NUMA Considerations for Multi-Socket Machines

On multi-socket servers (common for large-memory machines used in vector search), NUMA (Non-Uniform Memory Access) can cause significant and hard-to-diagnose performance variation.

Each CPU socket has its own memory controller and local RAM. Accessing local memory takes ~80-100ns, while accessing memory on the other socket crosses the inter-socket link (Intel UPI) and takes ~130-170ns — roughly 1.5-1.7x slower.

**The problem:** When PostgreSQL allocates `shared_buffers`, the kernel uses a first-touch policy — pages are placed on the NUMA node of the core that first accesses them. If `pg_prewarm` runs on a single core, the entire index (hundreds of GB) ends up physically on one NUMA node. Queries running on cores of the other node pay the remote access penalty on every buffer access.

For HNSW graph traversal, a single query performs thousands of random memory accesses (following neighbor pointers, loading vectors for distance computation). At efSearch=800 with a 646GB index, the difference between a query running on a "local" core vs a "remote" core can be **30-40% in QPS** — with no change in configuration.

**The fix:** Start PostgreSQL with interleaved memory allocation:

```bash
numactl --interleave=all pg_ctl start -D /path/to/pgdata

# Or via systemd override:
# [Service]
# ExecStart=numactl --interleave=all /usr/pgsql-17/bin/postgres -D /data/pgdata
```

This round-robins shared_buffers pages across all NUMA nodes, ensuring consistent average latency regardless of which core runs the query. You lose ~10-15% vs the best case (all local), but gain reproducible results and eliminate the worst case.

**Verify NUMA topology:**

```bash
numactl --hardware          # Show NUMA nodes and memory per node
numastat                    # Show hit/miss counters per node
numastat -p $(pgrep -d, postgres)  # Per-process NUMA stats
```

If `numa_miss` / `numa_foreign` counters are high relative to `numa_hit`, queries are frequently accessing remote memory.

> **Note:** AMD EPYC processors can have multiple NUMA nodes within a single socket (up to 4 or 8 depending on NPS configuration). Even "single socket" EPYC servers may exhibit NUMA effects. Check `numactl --hardware` to verify.

## Contributors

- **Alessandro Ferraresi** - Initial creator
- **Huan Zhang**
- **Tim Waizenegger**

## License

See LICENSE file for details.
