#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   TEST_EXEC=build/bin/mybench ./scripts/run_scalability.sh build
# If TEST_EXEC is not set, the script will try to run ctest -R scalability

BUILD_DIR=${1:-build}
TEST_EXEC=${TEST_EXEC:-}
OUT_FILE=${OUT_FILE:-scalability_results.csv}

if [ -n "$TEST_EXEC" ]; then
  echo "threads,seconds,cmd" > "$OUT_FILE"
  for t in 1 2 4 8 16; do
    echo "Running $TEST_EXEC with threads=$t"
    START=$(date +%s.%N)
    "$TEST_EXEC" --threads="$t"
    END=$(date +%s.%N)
    DIFF=$(echo "$END - $START" | bc)
    echo "$t,$DIFF,$TEST_EXEC --threads=$t" >> "$OUT_FILE"
  done
  echo "Results written to $OUT_FILE"
  cat "$OUT_FILE"
else
  echo "No TEST_EXEC provided, falling back to running ctest -R scalability (in $BUILD_DIR)"
  cd "$BUILD_DIR"
  ctest -R scalability --output-on-failure
fi
