#!/bin/bash

for file in *.ipynb
do
    newname="${file%.ipynb}.py"
    tmpname="$newname.tmp"
    echo processing "$file" to "$newname"
    jq -j ' .cells | map( select(.cell_type == "code") | .source + ["\n\n"] ) | .[][]' "$file" > "$tmpname"
    grep -v -e '^%' "$tmpname" | grep -v "get_ipython()" > "$newname"
    rm $tmpname
done
mv *.py ../

