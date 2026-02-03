#!/bin/bash
set -e

cd /src/python/

# Build wheels for each Python version
for PYBIN in /opt/python/cp3{8,9,10,11,12,13,14}*/bin/; do
    if [ -d "$PYBIN" ]; then
        echo "========================================"
        echo "Building for: ${PYBIN}"
        echo "========================================"
        
        # Get Python version
        PYVER=$(${PYBIN}/python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        echo "Python version: ${PYVER}"
        
        # Clean previous build artifacts
        rm -rf build/ *.egg-info/ dist/ builddir/
        
        # Install build dependencies
        "${PYBIN}/pip" install --upgrade pip
        "${PYBIN}/pip" install build meson-python meson ninja numpy
        
        # Install scipy - prefer pre-built wheels to avoid source compilation
        "${PYBIN}/pip" install --only-binary=:all: scipy || \
            "${PYBIN}/pip" install scipy
        
        # Build wheel using meson-python (respects pyproject.toml)
        echo "Using meson-python build system"
        "${PYBIN}/python" -m build --wheel --outdir wheels/
        "${PYBIN}/python" -m unittest discover -v tests
    fi
done

# Check if wheels need repair (have external shared library dependencies)
rm -rf dist/
mkdir -p dist/

for WHEEL in wheels/*.whl; do
    if [ -f "$WHEEL" ]; then
        echo "Checking: ${WHEEL}"
        
        # Check if wheel has external dependencies
        NEEDS_REPAIR=$(auditwheel show "${WHEEL}" 2>&1 || true)
        
        if echo "$NEEDS_REPAIR" | grep -q "is consistent with the following platform tag"; then
            # Wheel is already self-contained (static linking worked)
            echo "Wheel is self-contained, copying without repair"
            cp "${WHEEL}" dist/
        else
            # Wheel needs external libraries bundled
            echo "Repairing wheel (bundling shared libraries)..."
            auditwheel repair "${WHEEL}" -w dist/
        fi
    fi
done

# Cleanup
rm -rf wheels/ build/ *.egg-info/ builddir/

echo "========================================"
echo "Built wheels:"
ls -lh dist/
echo "========================================"
