#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "Executing vaccine notebook..."
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=600 \
    docs/source/examples/example_vaccine_immport.ipynb

echo "Syncing to examples/..."
cp docs/source/examples/example_vaccine_immport.ipynb examples/

echo "Building docs..."
sphinx-build -b html docs/source docs/build/html

echo "Done."
