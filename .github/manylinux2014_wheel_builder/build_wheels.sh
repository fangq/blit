#!/bin/bash
cd /src/python/

# Build wheels for each Python version
for PYBIN in /opt/python/cp3{9,10,11,12,13}*/bin/; do
    if [ -d "$PYBIN" ]; then
        echo "========================================"
        echo "Building for: ${PYBIN}"
        echo "========================================"
        
        # Get Python version
        PYVER=$(${PYBIN}/python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        PYVER_MAJOR=$(${PYBIN}/python -c "import sys; print(sys.version_info.major)")
        PYVER_MINOR=$(${PYBIN}/python -c "import sys; print(sys.version_info.minor)")
        
        echo "Python version: ${PYVER}"
        
        # Clean previous build artifacts
        rm -rf build/ *.egg-info/
        
        if [ "$PYVER_MAJOR" -eq 3 ] && [ "$PYVER_MINOR" -lt 12 ]; then
            # Python < 3.12: use numpy.distutils
            echo "Using numpy.distutils (legacy)"
            "${PYBIN}/pip" install wheel setuptools "numpy>=1.20,<2.0"
            "${PYBIN}/python" setup.py bdist_wheel -d wheels/
        else
            # Python >= 3.12: use meson
            echo "Using meson-python (modern)"
            "${PYBIN}/pip" install build meson-python meson ninja numpy
            "${PYBIN}/python" -m build --wheel --outdir wheels/
        fi
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
rm -rf wheels/ build/ *.egg-info/