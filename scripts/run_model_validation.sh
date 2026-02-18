#!/usr/bin/env bash
set -euo pipefail

# run_model_validation.sh [build_dir]
BUILD_DIR=${1:-build}
RESULTS_FILE=${RESULTS_FILE:-model_results.csv}
TEST_CMD=${TEST_CMD:-}
PY_VALIDATOR=${PY_VALIDATOR:-scripts/validate_model.py}
TOL=${TOL:-1e-6}

echo "Build dir: $BUILD_DIR"

if [ -n "$TEST_CMD" ]; then
  echo "Running provided test command: $TEST_CMD"
  eval "$TEST_CMD" > "$RESULTS_FILE"
  echo "Wrote results to $RESULTS_FILE"
else
  echo "No TEST_CMD provided, attempting to run ctest -R model_validation"
  cd "$BUILD_DIR"
  # Expect tests to output VTU/PVTu files; run the matching test
  ctest -R model_validation --output-on-failure || true

  # Find PVTu or VTU output files created by DataOut
  # Prefer PVTu master file if present
  cd "$PWD/.." || true
  pvtu_files=(output-*.pvtu)
  vtu_files=(output-*.vtu)

  if [ -e "${pvtu_files[0]}" ]; then
    RESULTS_FILE="${pvtu_files[0]}"
    echo "Found PVTu results: $RESULTS_FILE"
  elif [ -e "${vtu_files[0]}" ]; then
    # pick the most recently modified VTU
    RESULTS_FILE=$(ls -1t output-*.vtu | head -n 1)
    echo "Found VTU results: $RESULTS_FILE"
  else
    echo "No VTU/PVTu results found; exiting with error"
    exit 2
  fi
fi

echo "Running Python validator ($PY_VALIDATOR) with tolerance $TOL"
python "$PY_VALIDATOR" --results "$RESULTS_FILE" --tolerance "$TOL"

echo "Model validation completed"
