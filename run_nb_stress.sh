#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "Executing stress test notebook..."
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=600 \
    docs/source/examples/stress_test_real_scale.ipynb

echo "Syncing to examples/..."
cp docs/source/examples/stress_test_real_scale.ipynb examples/

echo "Building docs..."
sphinx-build -b html docs/source docs/build/html

echo "Done."
