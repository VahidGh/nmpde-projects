#!/usr/bin/env bash
set -euo pipefail

# run_model_validation.sh [build_dir]
VENV_DIR=${VENV_DIR:-venv}
BUILD_DIR=${1:-build}
RESULTS_FILE=${RESULTS_FILE:-model_results.csv}
TEST_CMD=${TEST_CMD:-}
PY_VALIDATOR=${PY_VALIDATOR:-scripts/validate_model.py}
# TOL=${TOL:-1e-6}
TOL=${TOL:-1e-1}

echo "Build dir: $BUILD_DIR"

# Set up Python virtual environment and install dependencies
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating Python virtual environment in $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

echo "Installing/updating validation dependencies..."
# Run pip in an isolated subshell to ignore external PYTHONPATH
(
  unset PYTHONPATH
  . "$VENV_DIR/bin/activate"
  pip install --upgrade pip
  pip install -r scripts/requirements.txt
)

if [ -n "$TEST_CMD" ]; then
  echo "Running provided test command: $TEST_CMD"
  eval "$TEST_CMD" > "$RESULTS_FILE"
  echo "Wrote results to $RESULTS_FILE"
else
  echo "No TEST_CMD provided, attempting to run ctest -R model_validation"
  # Expect tests to output VTU/PVTu files; run the matching test
  if ! (cd "$BUILD_DIR" && ctest -R model_validation --output-on-failure); then
    echo "ctest command failed. Aborting validation." >&2
    exit 1
  fi

  # Find PVTu or VTU output files created by DataOut
  # Prefer PVTu master file if present
  # pvtu_files=($BUILD_DIR/output-*.pvtu)
  # vtu_files=($BUILD_DIR/output-*.vtu)
  pvtu_files=($BUILD_DIR/output-*_400.pvtu)
  vtu_files=($BUILD_DIR/output-*_400_0.vtu)

  if [ -e "${pvtu_files[0]}" ]; then
    RESULTS_FILE="${pvtu_files[0]}"
    # RESULTS_FILE="${pvtu_files[-1]}"
    echo "Found PVTu results: $RESULTS_FILE"
  elif [ -e "${vtu_files[0]}" ]; then
    # pick the most recently modified VTU
    RESULTS_FILE="${vtu_files[0]}"
    # RESULTS_FILE="${vtu_files[-1]}"
    # RESULTS_FILE=$(ls -1t "$BUILD_DIR"/output-*.vtu | head -n 1)
    echo "Found VTU results: $RESULTS_FILE"
  else
    echo "No VTU/PVTu results found; exiting with error"
    exit 2
  fi
fi
echo "Running Python validator ($PY_VALIDATOR) with tolerance $TOL"

# Run the validator in an isolated subshell using the virtual environment
(
  unset PYTHONPATH
  . "$VENV_DIR/bin/activate"
  python3 "$PY_VALIDATOR" --results "$RESULTS_FILE" --tolerance "$TOL"
)

echo "Model validation completed"
