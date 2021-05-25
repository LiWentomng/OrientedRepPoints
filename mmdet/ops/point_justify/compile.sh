#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

echo "Building point_justify..."

if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace
