#!/bin/bash
set -e

# Full benchmark pipeline: build → benchmark → park, for each config.
#
# Usage:
#   ./run_full_pipeline.sh vectorchord   # Run all VectorChord builds + benchmarks
#   ./run_full_pipeline.sh pgvector      # Run all pgvector builds + benchmarks
#   ./run_full_pipeline.sh all           # Run everything

WORKDIR="${VSBT_WORKDIR:-$(cd "$(dirname "$0")/.." && pwd)}"
TOTAL_RAM_GB=1511

# ── Pipeline definitions ──────────────────────────────────────────────
# Format: SUITE|CONFIG|SCALE|MWM|MAX_QUERIES|PARK_NAME
#   SUITE:       pgvector_suite.py or vectorchord_suite.py
#   CONFIG:      config yaml path
#   SCALE:       server tier preset (5m, 100m, 1b)
#   MWM:         maintenance_work_mem for build
#   MAX_QUERIES: query limit (empty = full test set)
#   PARK_NAME:   index rename suffix after benchmarking

VC_PIPELINE=(
  "vectorchord_suite.py|config/vectorchord-5m-50-8k.yaml|5m|64MB|1000|laion_5m_test_ip_vc_50_8k"
  "vectorchord_suite.py|config/vectorchord-5m-190-35k.yaml|5m|64MB|1000|laion_5m_test_ip_vc_190_35k"
  "vectorchord_suite.py|config/vectorchord-100m-400-160k.yaml|100m|64MB|1000|laion_100m_test_ip_vc_400_160k"
  "vectorchord_suite.py|config/vectorchord-100m-570-320k.yaml|100m|64MB|1000|laion_100m_test_ip_vc_570_320k"
  "vectorchord_suite.py|config/vectorchord-deep1B-400-160k.yaml|1b|64MB|1000|deep1b_test_l2_vc_400_160k"
  "vectorchord_suite.py|config/vectorchord-deep1B-800-640k.yaml|1b|64MB|1000|deep1b_test_l2_vc_800_640k"
)

PGV_PIPELINE=(
  "pgvector_suite.py|config/pgvector-5m-m16-64.yaml|5m|16GB|1000|laion_5m_test_ip_hnsw_m16_64"
  "pgvector_suite.py|config/pgvector-5m-m16-128.yaml|5m|250GB|1000|laion_5m_test_ip_hnsw_m16_128"
  "pgvector_suite.py|config/pgvector-100m-m16-64.yaml|100m|750GB|1000|laion_100m_test_ip_hnsw_m16_64"
  "pgvector_suite.py|config/pgvector-100m-m16-128.yaml|100m|512GB|1000|laion_100m_test_ip_hnsw_m16_128"
  "pgvector_suite.py|config/pgvector-1B-m16-64.yaml|1b|1024GB|1000|deep1b_test_l2_hnsw_m16_64"
  "pgvector_suite.py|config/pgvector-1B-m16-128.yaml|1b|1024GB|1000|deep1b_test_l2_hnsw_m16_128"
)

CLIENTS="1,32"

# ── Helper functions ──────────────────────────────────────────────────

ensure_pg_running() {
  if ! systemctl is-active --quiet postgresql-17; then
    systemctl start postgresql-17
    sleep 5
  fi
}

park_index() {
  local table_name=$1
  local park_name=$2
  local current_idx="${table_name}_embedding_idx"

  echo "Parking index: $current_idx → $park_name"
  ensure_pg_running
  # Check if index exists before trying to park
  local exists
  exists=$(psql -U postgres -tA -c "SELECT 1 FROM pg_indexes WHERE indexname = '$current_idx'")
  if [ "$exists" = "1" ]; then
    psql -U postgres -q -c "ALTER INDEX $current_idx RENAME TO $park_name"
    psql -U postgres -q -c "UPDATE pg_index SET indisvalid = false WHERE indexrelid = '$park_name'::regclass"
    echo "  Parked and disabled."
  else
    echo "  Index $current_idx not found, skipping park."
  fi
}

get_table_name() {
  # Derive table name from config: read dataset, replace - with _
  local config=$1
  local dataset
  dataset=$(grep "dataset:" "$config" | head -1 | awk '{print $2}')
  echo "$dataset" | tr '-' '_'
}

run_step() {
  local runner=$1
  local config=$2
  local scale=$3
  local mwm=$4
  local max_queries=$5
  local park_name=$6

  local table_name
  table_name=$(get_table_name "$config")

  echo ""
  echo "╔══════════════════════════════════════════════════╗"
  echo "  Config: $config"
  echo "  Runner: $runner"
  echo "  Table:  $table_name"
  echo "  Scale:  $scale | MWM: $mwm"
  echo "╚══════════════════════════════════════════════════╝"
  echo ""

  # ── BUILD ──
  echo "=== BUILD PHASE ==="
  ensure_pg_running
  psql -U postgres -q -c "ALTER SYSTEM SET shared_buffers = '16GB'"
  psql -U postgres -q -c "ALTER SYSTEM SET maintenance_work_mem = '$mwm'"
  systemctl restart postgresql-17
  sleep 10

  cd "$WORKDIR" && python3 "$runner" -s "$config" --skip-add-embeddings --build-only

  # ── BENCHMARK ──
  echo ""
  echo "=== BENCHMARK PHASE ==="

  local mq_arg=""
  if [ -n "$max_queries" ]; then
    mq_arg="$max_queries"
  fi

  cd "$WORKDIR" && utils/run_benchmarks.sh "$config" "$scale" "$CLIENTS" $mq_arg

  # ── PARK ──
  echo ""
  echo "=== PARK PHASE ==="
  park_index "$table_name" "$park_name"
  echo ""
}

# ── Main ──────────────────────────────────────────────────────────────

MODE="${1:?Usage: $0 [vectorchord|pgvector|all]}"

PIPELINE=()
case "$MODE" in
  vectorchord|vc)  PIPELINE=("${VC_PIPELINE[@]}") ;;
  pgvector|pgv)    PIPELINE=("${PGV_PIPELINE[@]}") ;;
  all)             PIPELINE=("${VC_PIPELINE[@]}" "${PGV_PIPELINE[@]}") ;;
  *)               echo "Unknown mode: $MODE"; exit 1 ;;
esac

echo "═══════════════════════════════════════════════════"
echo "  Full Pipeline: $MODE"
echo "  Steps: ${#PIPELINE[@]}"
echo "  Clients: $CLIENTS"
echo "═══════════════════════════════════════════════════"

STEP=0
for entry in "${PIPELINE[@]}"; do
  STEP=$((STEP + 1))
  IFS='|' read -r runner config scale mwm max_queries park_name <<< "$entry"
  echo ""
  echo "━━━ Step $STEP/${#PIPELINE[@]} ━━━"
  run_step "$runner" "$config" "$scale" "$mwm" "$max_queries" "$park_name"
done

echo ""
echo "═══════════════════════════════════════════════════"
echo "  Pipeline complete! ($STEP steps)"
echo "═══════════════════════════════════════════════════"
