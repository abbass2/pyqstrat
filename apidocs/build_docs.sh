#!/usr/bin/env bash
# Next line is commented because I write my own pyqstrat.rst file in the source directory instead of having sphinx-apidoc create it.
# It screws up on cpp code and the example submodule
# sphinx-apidoc -f -o ./source/ .. ../setup.py
set -x
make clean
make html
