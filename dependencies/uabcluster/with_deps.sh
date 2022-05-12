#!/bin/bash
adirScript=$( cd $( dirname "$0" ) && pwd )

echo adirScript="$adirScript"
echo "*** about to add modules. . ."
module add Python/3.7.4-GCCcore-8.3.0; module add cuda10.2/toolkit/10.2.89
echo "*** about to install packages . . ."
bash "$adirScript/ensure_dependencies.sh"
LUSCRATCH=$(bash "$adirScript/echo_adir_scratch.sh")
source "$LUSCRATCH/venv/bin/activate"
echo "*** about to run job . . ."
"$@"
