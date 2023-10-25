#!/bin/bash
set -x
set -e
# generate .py files for notebooks and quit on first error
find pyqstrat -maxdepth 1 -name '*.ipynb' \( -exec jup_mini {} \; -o -quit \)

export NO_DISPLAY=1
mypy --ignore-missing-imports pyqstrat/
flake8 --ignore W291,W293,W503,E402,E701,E275,E741 --max-line-length=160 --extend-exclude notebooks pyqstrat/

# run notebooks and exit on first error
find pyqstrat/notebooks -name '*.ipynb' \( -exec sh -c 'trap "exit \$?" EXIT; ipython "$0"' {} \; -o  -quit \)

# build docs
rm -Rf ./docs/*
sphinx-build ./apidocs/source ./docs/
rm -Rf ./docs/build/
set +e
set +x
