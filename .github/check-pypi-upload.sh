#!/bin/bash

# Get the wheel filename
WHEEL_FILE=$(ls dist/*.whl 2>/dev/null | head -1)
if [ -z "$WHEEL_FILE" ]; then
    echo "No wheel file found in dist/"
    echo "perform_pypi_upload=1" >> $GITHUB_OUTPUT
    exit 0
fi

WHEEL_NAME=$(basename "$WHEEL_FILE")
echo "Built wheel: $WHEEL_NAME"

# Extract version from wheel filename
BLIT_BUILD_VERSION=$(echo "$WHEEL_NAME" | awk -F"-" '{ print $2 }')
echo "Version: $BLIT_BUILD_VERSION"

# Check if this specific wheel already exists on PyPI
# PyPI URL pattern: https://pypi.org/simple/blocksolver/
echo "Checking PyPI for existing wheels..."

UPLOAD_TO_PYPI=1

# Fetch the list of files from PyPI simple API
PYPI_FILES=$(curl -sL https://pypi.org/simple/blocksolver/ 2>/dev/null | grep -oP '(?<=href=")[^"]+\.whl(?=")' | xargs -I {} basename {})

if [ -n "$PYPI_FILES" ]; then
    echo "Existing wheels on PyPI:"
    echo "$PYPI_FILES" | head -20
    
    # Check if our specific wheel already exists
    if echo "$PYPI_FILES" | grep -qF "$WHEEL_NAME"; then
        echo "Wheel $WHEEL_NAME already exists on PyPI"
        UPLOAD_TO_PYPI=0
    else
        echo "Wheel $WHEEL_NAME not found on PyPI"
    fi
else
    echo "No existing wheels found on PyPI (new package?)"
fi

if [ "$UPLOAD_TO_PYPI" = "1" ]; then
    echo "Will upload wheel to PyPI"
else
    echo "Skipping upload - wheel already exists"
fi

echo "perform_pypi_upload=$UPLOAD_TO_PYPI" >> $GITHUB_OUTPUT
