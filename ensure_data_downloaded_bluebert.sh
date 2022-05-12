#!/bin/bash
adirRepo=$( cd $( dirname "$0" ) && pwd )

# http://redsymbol.net/articles/unofficial-bash-strict-mode/
set -euo pipefail
IFS=$'\n\t'

url=https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/NCBI-BERT/NCBI_BERT_pubmed_uncased_L-12_H-768_A-12.zip
sha1_expected=a4acf965b4a5e14c660fb2f4ffde28e98ae4a718
adir_data="${adirRepo}/bluebert/data"

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
