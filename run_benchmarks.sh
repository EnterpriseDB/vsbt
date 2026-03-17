#!/bin/bash
set -e

SUITE="${1:?Usage: $0 <config.yaml> [small|large|SB_LIST]}"
WORKDIR="/data/vsbt"

# Infer suite runner from config filename
case "$(basename "$SUITE")" in
  pgvector*)    RUNNER="pgvector_suite.py" ;;
  vectorchord*) RUNNER="vectorchord_suite.py" ;;
  pgpu*)        RUNNER="pgpu_suite.py" ;;
  *)            echo "Cannot infer suite from config: $SUITE"; exit 1 ;;
esac

# shared_buffers presets
SB_SMALL=(2GB 4GB 8GB 16GB 32GB)                  # 5M-20M datasets
SB_LARGE=(700GB 512GB 256GB 128GB 64GB 32GB 16GB)  # 100M-1B datasets

case "${2:-large}" in
  small) SB_LIST=("${SB_SMALL[@]}") ;;
  large) SB_LIST=("${SB_LARGE[@]}") ;;
  *)     IFS=',' read -ra SB_LIST <<< "$2" ;;
esac

echo "Config:  $SUITE"
echo "Runner:  $RUNNER"
echo "SB list: ${SB_LIST[*]}"
echo ""

for SB in "${SB_LIST[@]}"; do
  echo "============================================"
  echo "  Testing shared_buffers = $SB"
  echo "============================================"

  psql -U postgres -c "ALTER SYSTEM SET shared_buffers = '$SB'"
  psql -U postgres -c "ALTER SYSTEM SET maintenance_work_mem = '16GB'"
  systemctl restart postgresql-17
  sleep 10

  psql -U postgres -c "SHOW shared_buffers"

  echo "--- With page cache ---"
  cd $WORKDIR && python3 "$RUNNER" -s "$SUITE" --skip-add-embeddings --skip-index-creation

  echo "--- Without page cache ---"
  cd $WORKDIR && python3 "$RUNNER" -s "$SUITE" --skip-add-embeddings --skip-index-creation --no-fs-cache

  echo "Done with shared_buffers = $SB"
  echo ""
done

echo "All runs complete!"
