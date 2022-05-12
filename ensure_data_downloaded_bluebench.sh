#!/bin/bash
adirRepo=$( cd $( dirname "$0" ) && pwd )

# http://redsymbol.net/articles/unofficial-bash-strict-mode/
set -euo pipefail
IFS=$'\n\t'

url=https://github.com/ncbi-nlp/BLUE_Benchmark/releases/download/0.1/data_v0.2.zip
sha1_expected=7a6ddc7d6e702464bb7baf19081a747890c9131d
adir_data="${adirRepo}/BLUE_Benchmark/data"

sha1_old=""
if [[ -e "${adir_data}/sha1.txt" ]]
then
    sha1_old=$(cat "${adir_data}/sha1.txt")
fi

if [[ "$sha1_old" == "$sha1_expected" ]]
then
    echo "Valid data has already been downloaded"
else
    rm -rf "${adir_data}"
    mkdir -p "${adir_data}"
    echo about to: curl -L "$url" -o "${adir_data}/data.zip"
    curl -L "$url" -o ${adir_data}/data.zip
    sha1_new=$(sha1sum "${adir_data}/data.zip" | cut -c1-40)
    (cd "${adir_data}"; unzip data.zip)
    echo "$sha1_new" > "${adir_data}/sha1.txt"
    rm "${adir_data}/data.zip"
fi
