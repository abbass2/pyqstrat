#!/bin/bash
set -x
#find pyqstrat -maxdepth 1 -name '*.ipynb' -exec jup_mini {} \;
export NO_DISPLAY=1
#mypy --ignore-missing-imports pyqstrat/
#flake8 --ignore W291,W293,W503,E402,E701,E275,E741 --max-line-length=160 --extend-exclude notebooks pyqstrat/
# run notebooks
#find pyqstrat/notebooks . -maxdepth 1 -name '*.ipynb' -exec ipython {} \; > /dev/null

set -x
# build docs
rm -Rf ./docs/*
sphinx-build ./apidocs/source ./docs/
# rm -Rf ./docs/build/
set +x
