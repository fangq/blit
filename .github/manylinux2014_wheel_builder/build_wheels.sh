#!/bin/bash
cd python/
for PYBIN in /opt/python/cp3{8,9,10,11,12}*/bin/; do
    if [ -d "$PYBIN" ]; then
        "${PYBIN}/pip" install wheel "setuptools<65.0" "numpy>=1.25,<2.0"
        "${PYBIN}/python" setup.py bdist_wheel -d wheels/
    fi
done
rm -rf dist/
for WHEEL in wheels/*; do
    auditwheel repair ${WHEEL} -w dist/
done
rm -rf wheels/ build/ *.egg-info/