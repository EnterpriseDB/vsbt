# Vector Search Benchmark Toolkit (vsbt) - User Guide

This guide walks through setting up and running PostgreSQL vector search benchmarks using the `vsbt` toolkit. 
It covers installing PostgreSQL, configuring it for vector search workloads, installing the `pgvector` extension, and running benchmarks with various datasets and index configurations.

To demonstrate the process, we will configure a new environment to run the toolkit, setup a PostgreSQL 17 cluster, and execute a benchmark suite using the `laion-5m-test-ip` dataset with HNSW index (m=16) and 64-bit quantization.
In this exercise, the path `/home` is the mountpoint for a large filesystem, and will be used for the PostgreSQL data directory and `vsbt` repository to ensure sufficient disk space for datasets and indexes.

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Repository Setup](#2-repository-setup)
3. [Python Installation](#3-python-installation)
4. [PostgreSQL Installation](#4-postgresql-installation)
5. [PostgreSQL Configuration](#5-postgresql-configuration)
6. [Install pgvector Extension](#6-install-pgvector-extension)
7. [vsbt Installation](#7-vsbt-installation)
8. [Running Benchmarks](#8-running-benchmarks)
9. [Troubleshooting](#9-troubleshooting)

For detailed usage instructions, configuration options, and result interpretation, see [README.md](README.md).

---

## 1. Prerequisites

**Hardware Requirements:**
- Sufficient RAM for your dataset size (see [README.md](README.md#pgvector-memory-tuning-for-large-hnsw-index-builds) for memory sizing)
- Sufficient disk space for datasets and PostgreSQL data (see [README.md](README.md#supported-datasets) for dataset sizes)

**Software Requirements:**
- RHEL 9 (or compatible: Rocky Linux 9, AlmaLinux 9)
- Root or sudo access
- Internet connectivity (for package downloads and datasets)

**Important:** Ensure the PostgreSQL data directory and `vsbt` repository are set up on a disk with sufficient space for your target dataset.

---

## 2. Repository Setup

Install the required package repositories:

```sh
# EPEL repository (for additional packages)
sudo dnf install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm

# PGDG repository (required for pgvector extension)
# For ppc64le architecture:
sudo dnf install -y https://download.postgresql.org/pub/repos/yum/reporpms/EL-9-ppc64le/pgdg-redhat-repo-latest.noarch.rpm

# For x86_64 architecture (if applicable):
# sudo dnf install -y https://download.postgresql.org/pub/repos/yum/reporpms/EL-9-x86_64/pgdg-redhat-repo-latest.noarch.rpm
```

**(Optional) EDB Repository** - for EDB customers installing PostgreSQL from EDB:

```sh
# Replace <username> and <token> with your EDB repository credentials
curl -u "<username>:<token>" -1sLf \
  'https://downloads.enterprisedb.com/basic/enterprise/setup.rpm.sh' \
  | sudo -E bash
```

---

## 3. Python Installation

`vsbt` requires Python 3.10 or higher. RHEL 9 ships with Python 3.9, so install a newer version:

```sh
# Install Python 3.12 and development headers
sudo dnf install -y python3.12 python3.12-pip python3.12-devel

# Set Python 3.12 as the default python3 (optional)
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Verify installation
python3 --version
# Should output: Python 3.12.x
```

---

## 4. PostgreSQL Installation

This guide uses PostgreSQL 17, but versions 15+ are supported. Adjust package names accordingly (e.g., `postgresql16` for version 16).

```sh
# Install PostgreSQL 17 server, client, and libraries
sudo dnf install -y postgresql17 postgresql17-server postgresql17-contrib postgresql17-libs

# Install development tools (required for some extensions)
sudo dnf groupinstall -y "Development Tools"
```

### Configure the postgres System User

The PostgreSQL packages create a `postgres` system user automatically. Configure it for easier administration:

```sh
# Verify the postgres user exists
id postgres

# Set a password for the postgres system user (for sudo/su access)
sudo passwd postgres

# (Optional) Add postgres user to sudoers for benchmark operations
# This allows dropping filesystem cache with --no-fs-cache option
sudo tee /etc/sudoers.d/postgres << 'EOF'
postgres ALL=(ALL) NOPASSWD: /usr/bin/sync, /usr/bin/tee /proc/sys/vm/drop_caches
EOF
sudo chmod 440 /etc/sudoers.d/postgres
```

### Initialize the Database

```sh
# Create data directory (use a path on a disk with sufficient space)
sudo mkdir -p /home/postgres/data
sudo chown postgres:postgres /home/postgres/data
sudo chmod 700 /home/postgres/data

# Create runtime directory (required for Unix socket connections)
sudo mkdir -p /run/postgresql
sudo chown postgres:postgres /run/postgresql
sudo chmod 755 /run/postgresql

# Initialize the database cluster
sudo -u postgres /usr/pgsql-17/bin/initdb -D /home/postgres/data
```

**Important:** If you use a custom data directory path, replace `/home/postgres/data` with your actual path in all subsequent commands throughout this guide (`pg_ctl`, `pg_hba.conf` location, etc.).

### Configure Authentication

Edit `pg_hba.conf` file and add these lines:

```
# Allow local connections without password for postgres user
local   all             postgres                                trust
host    all             postgres        127.0.0.1/32            trust
host    all             postgres        ::1/128                 trust
```

### Start PostgreSQL

```sh
# Start the server
sudo -u postgres /usr/pgsql-17/bin/pg_ctl start -D /home/postgres/data -l /home/postgres/data/logfile

# Verify it's running
sudo -u postgres /usr/pgsql-17/bin/pg_ctl status -D /home/postgres/data
```

### Create the postgres Superuser

The `initdb` command creates a database superuser role matching the OS user (typically `postgres`). If using an existing cluster or the role doesn't exist, create it manually:

```sql
CREATE ROLE postgres WITH LOGIN SUPERUSER;
```

---

## 5. PostgreSQL Configuration

Connect to PostgreSQL and configure performance parameters:

```sh
sudo -u postgres psql
```

Run these SQL commands to tune PostgreSQL for benchmarking:

```sql
-- Disable autovacuum during benchmarks (re-enable for production)
ALTER SYSTEM SET autovacuum = 'off';

-- Parallelism settings
ALTER SYSTEM SET max_worker_processes = '128';
ALTER SYSTEM SET max_parallel_maintenance_workers = '32';
ALTER SYSTEM SET max_parallel_workers = '64';

-- I/O settings
ALTER SYSTEM SET effective_io_concurrency = '200';
ALTER SYSTEM SET random_page_cost = '1.1';

-- WAL settings (large value to avoid checkpoint pressure during index builds)
ALTER SYSTEM SET max_wal_size = '500GB';

-- Memory settings (adjust based on your system RAM)
ALTER SYSTEM SET shared_buffers = '32GB';
ALTER SYSTEM SET maintenance_work_mem = '32GB';

-- Logging
ALTER SYSTEM SET client_min_messages = 'info';
```

Restart PostgreSQL to apply changes:

```sh
sudo -u postgres /usr/pgsql-17/bin/pg_ctl restart -D /home/postgres/data
```

### Memory Sizing Guidelines

See [README.md](README.md#pgvector-memory-tuning-for-large-hnsw-index-builds) for detailed memory tuning guidance.

---

## 6. Install pgvector Extension

```sh
# Install pgvector from PGDG repository
sudo dnf install -y pgvector_17
```

Enable the extensions in PostgreSQL:

```sh
sudo -u postgres psql
```

```sql
-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable pg_prewarm (for warming index into shared_buffers)
CREATE EXTENSION IF NOT EXISTS pg_prewarm;

-- Verify extensions are installed
\dx
```

---

## 7. vsbt Installation

```sh
# Install git and HDF5 development headers (required for h5py)
sudo dnf install -y git hdf5-devel

# Ensure you are in the directory where you want to clone the repository and sufficient disk space is available for the datasets
# In this example /home is a directory with sufficient space.
cd /home

# Clone the vsbt repository
git clone https://github.com/EnterpriseDB/vsbt.git

# Create and activate a Python virtual environment undder the vsbt directory
cd vsbt
python3.12 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

## 8. Running Benchmarks

To demonstrate running a benchmark, we'll execute the pgvector suite with the 5M vector dataset. This will test pgvector's performance with HNSW indexes (m=16) and 64-bit quantization.

This test has the configuration file `config/pgvector-5m-m16-64.yaml` which specifies the dataset, index parameters, and query settings.
See the `/config` directory for other pre-defined benchmark configurations, or create your own custom configuration file.

```sh
# Navigate to the vsbt directory if not already there
cd vsbt

# Activate the virtual environment
source venv/bin/activate

# Run pgvector benchmark with 5M vectors
python pgvector_suite.py -s config/pgvector-5m-m16-64.yaml 
```

This downloads the dataset to `/datasets` (first run only), loads vectors, builds indexes, runs queries, and generates a report in `results/`.

See [README.md](README.md#output) for details on output format and generated reports.

**Example output:**

```sh
python pgvector_suite.py -s config/pgvector-5m-m16-64.yaml
Running test: pgvector-laion-5m-m16-64
Table laion_5m_test_ip already exists, using it
Dropping index laion_5m_test_ip_embedding_idx... done!
Estimated HNSW graph memory: 17.5 GB (recommended maintenance_work_mem >= '18GB')
Estimated on-disk index size: 19.0 GB (recommended shared_buffers >= '19GB' for query serving)
Building index (initializing)...
Building index (building index: loading tuples): 100%|███████████████████████████████████| [10:53<?]
Index build time: 654s
Index built successfully.
Table: 37.7 GB | Index: 18.8 GB | shared_buffers: 32.0 GB (index coverage: 100.0%)
Prewarming the index into shared_buffers... done! (1.0s)
Running benchmark: {'efSearch': 20}
Running sequential benchmark with 10000 queries
recall: 0.8628 QPS: 596.06 P50: 1.62ms:  10000/10000: 100%|████████████████████|
Top: 10 | Recall: 0.8628 | QPS: 596.06 | P50: 1.62ms | P99: 3.12ms
Running benchmark: {'efSearch': 40}
Running sequential benchmark with 10000 queries
recall: 0.9206 QPS: 430.33 P50: 2.31ms:  10000/10000: 100%|████████████████████|
Top: 10 | Recall: 0.9206 | QPS: 430.33 | P50: 2.31ms | P99: 4.08ms
Running benchmark: {'efSearch': 85}
Running sequential benchmark with 10000 queries
recall: 0.9535 QPS: 285.95 P50: 3.50ms:  10000/10000: 100%|████████████████████|
Top: 10 | Recall: 0.9535 | QPS: 285.95 | P50: 3.50ms | P99: 5.88ms
Running benchmark: {'efSearch': 120}
Running sequential benchmark with 10000 queries
recall: 0.9627 QPS: 226.58 P50: 4.44ms:  10000/10000: 100%|████████████████████|
Top: 10 | Recall: 0.9627 | QPS: 226.58 | P50: 4.44ms | P99: 7.23ms
Running benchmark: {'efSearch': 200}
Running sequential benchmark with 10000 queries
recall: 0.9712 QPS: 166.62 P50: 5.99ms:  10000/10000: 100%|████████████████████|
Top: 10 | Recall: 0.9712 | QPS: 166.62 | P50: 5.99ms | P99: 9.71ms
Running benchmark: {'efSearch': 400}
Running sequential benchmark with 10000 queries
recall: 0.9789 QPS: 100.56 P50: 9.89ms:  10000/10000: 100%|████████████████████|
Top: 10 | Recall: 0.9789 | QPS: 100.56 | P50: 9.89ms | P99: 16.20ms

=====================================================
  Results Summary: pgvector-laion-5m-m16-64
  shared_buffers: 32GB | clients: 1 | index_size: 19 GB
=====================================================
| EF Search | Recall | QPS    | P50 (ms) | P99 (ms) |
|-----------|--------|--------|----------|----------|
| 20        | 0.8628 | 596.06 |     1.62 |     3.12 |
| 40        | 0.9206 | 430.33 |     2.31 |     4.08 |
| 85        | 0.9535 | 285.95 |     3.50 |     5.88 |
| 120       | 0.9627 | 226.58 |     4.44 |     7.23 |
| 200       | 0.9712 | 166.62 |     5.99 |     9.71 |
| 400       | 0.9789 | 100.56 |     9.89 |    16.20 |


 Results available in results/
Test suite completed.
```

See README.md for [metrics reported](README.md#metrics-reported) and [comparing benchmark runs](README.md#comparing-runs).

---

## 9. Troubleshooting

### Connection Refused

**Symptom:** `connection refused` or `could not connect to server`

```sh
# Check if PostgreSQL is running
sudo -u postgres /usr/pgsql-17/bin/pg_ctl status -D /home/postgres/data

# Start if not running
sudo -u postgres /usr/pgsql-17/bin/pg_ctl start -D /home/postgres/data -l /home/postgres/data/logfile

# Check pg_hba.conf allows your connection
cat /home/postgres/data/pg_hba.conf
```

### Permission Denied on /run/postgresql

```sh
sudo mkdir -p /run/postgresql
sudo chown postgres:postgres /run/postgresql
sudo chmod 755 /run/postgresql
```

### Python Dependency Errors

If `pip install -r requirements.txt` fails or you see `ModuleNotFoundError` when running vsbt, some system dependencies may be missing. Install the required development packages, then reinstall:

```sh
source venv/bin/activate
pip install -r requirements.txt
```
