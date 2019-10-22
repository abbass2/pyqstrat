#!/bin/bash

for file in *.ipynb
do
    newname="${file%.ipynb}.py"
    tmpname="$newname.tmp"
    echo processing "$file" to "$newname"
    jq -j ' .cells | map( select(.cell_type == "code") | .source + ["\n\n"] ) | .[][]' "$file" > "$tmpname"
    # remove ipython magics and trailing blank lines
    grep -v -e '^%' "$tmpname" | grep -v "get_ipython()" | sed -e :a -e '/^\n*$/{$d;N;};/\n$/ba' > "../$newname"
    # Delete all trailing blank lines at end of file (only).
    rm $tmpname
done

