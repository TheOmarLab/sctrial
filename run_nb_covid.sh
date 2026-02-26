#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "Executing COVID notebook..."
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=600 \
    docs/source/examples/example_covid19_stephenson.ipynb

echo "Syncing to examples/..."
cp docs/source/examples/example_covid19_stephenson.ipynb examples/

echo "Building docs..."
sphinx-build -b html docs/source docs/build/html

echo "Done."
