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
# Install Python dependencies
pip install -r requirements.txt
```

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

Results are organized in the following structure:

```
results/
├── raw/                                    # Individual run data (JSON)
│   └── {suite_name}_{timestamp}.json
├── consolidated/                           # Accumulated results across runs
│   └── all_results.csv
├── reports/                                # Generated reports
│   ├── {suite_name}_report.md              # Markdown report with tables
│   └── charts/
│       ├── {suite_name}_recall_vs_qps.png  # Recall vs QPS scatter plot
│       ├── {suite_name}_latency.png        # P50/P99 latency comparison
│       └── {suite_name}_build_times.png    # Build time breakdown
└── {suite_name}/                           # OS monitoring data
    ├── system_report.txt                   # Hardware information
    ├── cpu_utilization.csv/png             # CPU metrics over time
    ├── *_io_iops.csv                       # IOPS data per device
    ├── *_io_bandwidth.csv                  # Bandwidth data per device
    └── *_io_latency.csv                    # IO latency data
```

### Generated Reports

Each benchmark run generates:

| Output | Description |
|--------|-------------|
| **Markdown Report** | Configuration, build metrics, results table, and embedded charts |
| **Recall vs QPS Chart** | Scatter plot showing recall/throughput tradeoff |
| **Latency Chart** | Bar chart comparing P50/P99 latencies across configs |
| **Build Time Chart** | Horizontal bar showing load/clustering/index build breakdown |
| **Raw JSON** | Complete results data for programmatic access |
| **Consolidated CSV** | Append-only CSV accumulating results across all runs |

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
    ├── raw/                  # JSON data per run
    ├── consolidated/         # Accumulated CSV
    └── reports/              # Markdown reports and charts
```

## Contributors

- **Alessandro Ferraresi** - Initial creator
- **Huan Zhang**
- **Tim Waizenegger**

## License

See LICENSE file for details.
