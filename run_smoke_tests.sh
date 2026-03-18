#!/usr/bin/env bash
# Run the smoke test suite for train.py.
# Usage:
#   ./run_smoke_tests.sh              # core tests + compiled
#   ./run_smoke_tests.sh --no-compile # skip compiled test (faster)
#   ./run_smoke_tests.sh --integration # also run prepare + training-loop tests
#   SMOKE_TEST8_TIMEOUT=3600 ./run_smoke_tests.sh --integration  # if Test 8 times out (cold cache)
#   SMOKE_TEST8_LOG=/tmp/t8.log ./run_smoke_tests.sh --integration  # fixed log path
#   ./run_smoke_tests.sh --force      # skip training/GPU guard

set -e
cd "$(dirname "$0")"

if [[ -f .venv/bin/python ]]; then
  exec .venv/bin/python smoke_test.py "$@"
else
  exec python3 smoke_test.py "$@"
fi
