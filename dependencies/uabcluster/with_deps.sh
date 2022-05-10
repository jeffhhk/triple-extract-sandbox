#!/bin/bash
adirScript=$( cd $( dirname "$0" ) && pwd )

echo adirScript="$adirScript"
echo "*** about to add modules. . ."
module add Python/3.7.4-GCCcore-8.3.0; module add cuda10.2/toolkit/10.2.89
echo "*** about to install packages . . ."
bash "$adirScript/ensure_dependencies.sh" venv requirements_bluebenchmark.txt
LUSCRATCH=$(bash "$adirScript/echo_adir_scratch.sh")
source "$LUSCRATCH/venv/bin/activate"
# TODO FIX
# echo "*** installing spacy vocab"
# python -m spacy download en_core_web_sm
echo "*** about to run job . . ."
"$@"
