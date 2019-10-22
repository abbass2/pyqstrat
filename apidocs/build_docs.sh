#!/usr/bin/env bash

set -x
cp ../build/lib.macosx-10.9-x86_64-3.7/pyqstrat/pyqstrat_cpp.cpython-37m-darwin.so ../pyqstrat/
rm -Rf ./build/
sphinx-build -M html "source" "build"
rm -Rf ../docs/*
cp -R ./build/html/* ../docs/
