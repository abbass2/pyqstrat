!/bin/bash
sphinx-apidoc -f -o ./source/ .. ../setup.py; make html