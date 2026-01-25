#!/bin/bash
BLIT_BUILD_VERSION=$(awk -F"-" '{ print $2 }' <<< $(ls dist/ | head -1))
BLIT_VERSIONS_STRING=$(pip index versions blocksolver | grep versions:)
BLIT_VERSIONS_STRING=${BLIT_VERSIONS_STRING#*:}
UPLOAD_TO_PYPI=1
while IFS=', ' read -ra BLIT_VERSIONS_ARRAY; do
  for VERSION in "${BLIT_VERSIONS_ARRAY[@]}"; do
    if [ "$BLIT_BUILD_VERSION" = "$VERSION" ]; then
      UPLOAD_TO_PYPI=0
    fi
  done;
done <<< "$BLIT_VERSIONS_STRING"
if [ "$UPLOAD_TO_PYPI" = 1 ]; then
  echo "Wheel version wasn't found on PyPi.";
else
  echo "Wheel was found on PyPi.";
fi
echo "perform_pypi_upload=$UPLOAD_TO_PYPI" >> $GITHUB_OUTPUT

