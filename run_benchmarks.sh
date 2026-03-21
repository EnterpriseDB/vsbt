#!/bin/bash
set -e

SUITE="${1:?Usage: $0 <config.yaml> [5m|100m|1b|SB:RAM,...] [query-clients]}"
SCALE="${2:-100m}"
QUERY_CLIENTS="${3:-1}"
WORKDIR="/data/vsbt"
TOTAL_RAM_GB=1511
HUGEPAGE_SIZE_MB=2

# Infer suite runner from config filename
case "$(basename "$SUITE")" in
  pgvector*)    RUNNER="pgvector_suite.py" ;;
  vectorchord*) RUNNER="vectorchord_suite.py" ;;
  pgpu*)        RUNNER="pgpu_suite.py" ;;
  *)            echo "Cannot infer suite from config: $SUITE"; exit 1 ;;
esac

# Server tier definitions: "SB_GB:AVAILABLE_RAM_GB" pairs
# Last tier in each scale uses full machine (no huge pages)
case "$SCALE" in
  5m)
    TIERS=("2:8" "4:16" "8:32" "16:64" "32:128" "64:256")
    ;;
  100m)
    TIERS=("16:64" "32:128" "64:256" "128:512" "256:1024" "512:$TOTAL_RAM_GB")
    ;;
  1b)
    TIERS=("32:128" "64:256" "128:512" "256:1024" "512:1200" "750:$TOTAL_RAM_GB")
    ;;
  *)
    # Custom tiers: comma-separated SB:RAM pairs
    IFS=',' read -ra TIERS <<< "$SCALE"
    ;;
esac

echo "Config:   $SUITE"
echo "Runner:   $RUNNER"
echo "Scale:    $SCALE"
echo "Clients:  $QUERY_CLIENTS"
echo "Tiers:    ${TIERS[*]}"
echo ""

release_hugepages() {
  echo 0 > /proc/sys/vm/nr_hugepages 2>/dev/null || true
}

allocate_hugepages() {
  local lock_gb=$1
  if [ "$lock_gb" -le 0 ]; then
    return
  fi
  local pages=$((lock_gb * 1024 / HUGEPAGE_SIZE_MB))
  echo 3 > /proc/sys/vm/drop_caches
  echo "$pages" > /proc/sys/vm/nr_hugepages
  local actual
  actual=$(cat /proc/sys/vm/nr_hugepages)
  local actual_gb=$((actual * HUGEPAGE_SIZE_MB / 1024))
  echo "Huge pages: locked ${actual_gb}GB / ${lock_gb}GB requested"
  if [ "$actual_gb" -lt "$((lock_gb - 5))" ]; then
    echo "WARNING: Could not allocate enough huge pages (fragmentation?)"
  fi
}

# Ensure PostgreSQL has huge_pages = off so it doesn't use our reserved pages
ensure_pg_hugepages_off() {
  release_hugepages
  if ! systemctl is-active --quiet postgresql-17; then
    systemctl start postgresql-17
    sleep 5
  fi
  psql -U postgres -c "ALTER SYSTEM SET huge_pages = 'off'" 2>/dev/null || true
}

ensure_pg_hugepages_off

for TIER in "${TIERS[@]}"; do
  IFS=':' read -r SB_GB AVAIL_GB <<< "$TIER"
  LOCK_GB=$((TOTAL_RAM_GB - AVAIL_GB))

  echo "============================================"
  echo "  Simulated server: ${AVAIL_GB}GB RAM"
  echo "  shared_buffers:   ${SB_GB}GB"
  echo "  FS cache:         ~$((AVAIL_GB - SB_GB - 4))GB"
  echo "============================================"

  # 1. Release previous huge pages and clear cache for clean start
  release_hugepages
  echo 3 > /proc/sys/vm/drop_caches
  if ! systemctl is-active --quiet postgresql-17; then
    systemctl start postgresql-17
    sleep 5
  fi

  # 2. Set shared_buffers for this tier
  psql -U postgres -c "ALTER SYSTEM SET shared_buffers = '${SB_GB}GB'"
  psql -U postgres -c "ALTER SYSTEM SET maintenance_work_mem = '16GB'"

  # 3. Stop PG before locking memory
  systemctl stop postgresql-17
  sleep 5

  # 4. Lock memory via huge pages (skip for full machine)
  if [ "$LOCK_GB" -gt 0 ]; then
    allocate_hugepages "$LOCK_GB"
  else
    echo "Full machine — no memory restriction"
  fi

  # 5. Start PG with constrained memory
  if ! systemctl start postgresql-17; then
    echo "ERROR: PostgreSQL failed to start (not enough memory for SB=${SB_GB}GB with ${AVAIL_GB}GB available?)"
    release_hugepages
    continue
  fi
  sleep 10

  psql -U postgres -c "SHOW shared_buffers"

  # 6. Run benchmark
  cd "$WORKDIR" && python3 "$RUNNER" -s "$SUITE" --skip-add-embeddings --skip-index-creation --query-clients "$QUERY_CLIENTS"

  echo ""
done

# Cleanup: release all huge pages
release_hugepages
echo "All runs complete!"
