# =============================================================================
# vsbt (Vector Search Benchmark Tool) Container Image
# =============================================================================
# Contains all dependencies for running vector search benchmarks:
# psycopg3, numpy, h5py, pgvector.
#
# Used standalone or as a K8s Job image by pgperf.
#
# Build:
#   docker build -t ghcr.io/enterprisedb/vsbt:latest .
#
# Push:
#   docker push ghcr.io/enterprisedb/vsbt:latest
#
# Run:
#   docker run ghcr.io/enterprisedb/vsbt:latest \
#     python pgvector_suite.py -s /config/suite.yaml --url "postgresql://..."
# =============================================================================

FROM python:3.13-slim

# HDF5 C library for h5py
RUN apt-get update && apt-get install -y --no-install-recommends \
    libhdf5-dev \
    pkg-config \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy source
COPY . /app/vsbt/

# Install Python dependencies
WORKDIR /app/vsbt
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app/vsbt
