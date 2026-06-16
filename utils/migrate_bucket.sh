#!/bin/bash
# migrate_bucket.sh — Reorganise s3://enterprisedb-vector-datasets/
#
# All moves are server-side (same bucket); no data transfer fees.
# laion-100m is 307 GB — the mv is fast but may take a few minutes.
#
# Run: bash utils/migrate_bucket.sh [--dry-run]
#
# --dry-run prints the commands without executing them.

set -euo pipefail

BUCKET="s3://enterprisedb-vector-datasets"
DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1

mv_file() {
    local src="$BUCKET/$1" dst="$BUCKET/$2"
    echo "mv  $1"
    echo "  → $2"
    if [[ $DRY -eq 0 ]]; then
        aws s3 mv "$src" "$dst"
    fi
}

mv_prefix() {
    local src="$BUCKET/$1" dst="$BUCKET/$2"
    echo "mv  $1*"
    echo "  → $2"
    if [[ $DRY -eq 0 ]]; then
        aws s3 mv "$src" "$dst" --recursive
    fi
}

echo "=== Flat HDF5 files ==="
mv_file "laion-5m-test-ip.hdf5"             "laion/5m/base.hdf5"
mv_file "laion-20m-test-ip.hdf5"            "laion/20m/base.hdf5"
mv_file "laion-100m-test-ip.hdf5"           "laion/100m/base.hdf5"   # 307 GB — takes a few min
mv_file "laion_400m_gt.npy"                 "laion/400m/gt.npy"
mv_file "sift-128-euclidean.hdf5"           "sift/1m/base.hdf5"
mv_file "glove-100-angular.hdf5"            "glove/1m/base.hdf5"
mv_file "gist-960-euclidean.hdf5"           "gist/1m/base.hdf5"
mv_file "dbpedia-openai-1000k-angular.hdf5" "dbpedia/1m/base.hdf5"

echo ""
echo "=== deep1B → deep1b/1b/ ==="
mv_prefix "deep1B/" "deep1b/1b/"

echo ""
echo "=== laion-400m/ (parts downloaded from external; just rename prefix) ==="
mv_prefix "laion-400m/" "laion/400m/parts/"

echo ""
echo "=== openai subfolders ==="
mv_prefix "openai/openai_medium_500k/" "openai/500k/"
mv_prefix "openai/openai_small_1m/"    "openai/1m/"
mv_prefix "openai/openai_medium_2m/"   "openai/2m/"
mv_prefix "openai/openai_large_5m/"    "openai/5m/"

echo ""
echo "=== cohere subfolders ==="
mv_prefix "cohere/cohere_medium_1m/"  "cohere/1m/"
mv_prefix "cohere/cohere_small_2m/"   "cohere/2m/"
mv_prefix "cohere/cohere_medium_3m/"  "cohere/3m/"
mv_prefix "cohere/cohere_large_10m/"  "cohere/10m/"

echo ""
echo "=== yfcc raw files ==="
mv_prefix "yfcc-10m-filtered/" "yfcc/10m/raw/"

echo ""
echo "Done. Verify with: aws s3 ls s3://enterprisedb-vector-datasets/"
