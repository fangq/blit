#!/bin/bash
cd python/
for PYBIN in /opt/python/*/bin/; do
    "${PYBIN}/pip" install "numpy<2.0"
    "${PYBIN}/pip" wheel . -w wheels/
done
rm -rf dist/
for WHEEL in wheels/*; do
    auditwheel repair ${WHEEL} -w dist/
done
rm -rf wheels/