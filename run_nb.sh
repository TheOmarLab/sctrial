#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "Executing immunotherapy notebook..."
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=600 \
    docs/source/examples/example_immunotherapy_sade_feldman.ipynb

echo "Syncing to examples/..."
cp docs/source/examples/example_immunotherapy_sade_feldman.ipynb examples/

echo "Building docs..."
sphinx-build -b html docs/source docs/build/html

echo "Done."
