#!/usr/bin/env bash

set -x
rm -Rf ./build/
sphinx-build -M html "source" "build"
rm -Rf ../docs/*
cp -R ./build/html/* ../docs/
