#!/bin/bash
set -e

cd /src/python/

# Build wheels for each Python version
for PYBIN in /opt/python/cp3{9,10,11,12,13}*/bin/; do
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
        "${PYBIN}/pip" install build meson-python meson ninja numpy
        
        # Build wheel using meson-python (respects pyproject.toml)
        echo "Using meson-python build system"
        "${PYBIN}/python" -m build --wheel --outdir wheels/
    fi
done

# Repair wheels with auditwheel
rm -rf dist/
mkdir -p dist/
for WHEEL in wheels/*.whl; do
    if [ -f "$WHEEL" ]; then
        echo "Repairing: ${WHEEL}"
        auditwheel repair "${WHEEL}" -w dist/
    fi
done

# Cleanup
rm -rf wheels/ build/ *.egg-info/ builddir/