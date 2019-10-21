#!/bin/bash

jupyter nbconvert --to python *.ipynb
for filename in *.py
do
    grep -v 'get_ipython()' $filename > $filename
done
