#!/bin/bash
adirScript=$( cd $( dirname "$0" ) && pwd )

echo adirScript="$adirScript"
echo "*** about to add modules. . ."
module add Python/3.7.4-GCCcore-8.3.0
module add cuda10.0/toolkit/10.0.130
module add cuDNN/7.4.2.24-CUDA-10.0.130
# echo "*** invalidating dependencies ***"
#rm -rf /scratch/local/jeff@groovescale.com/venvtf
echo "*** about to install packages . . ."
bash "$adirScript/ensure_dependencies.sh" venvtf requirements_bluebenchmark.txt
LUSCRATCH=$(bash "$adirScript/echo_adir_scratch.sh")
source "$LUSCRATCH/venvtf/bin/activate"

echo "*** horrible necessary hack or else GPUS will not be detected: install tensorflow-gpu separately ***"
pip install -r "$adirScript/../../requirements_tensorflow-gpu.txt"
# TODO FIX
# echo "*** installing spacy vocab"
# python -m spacy download en_core_web_sm
echo "*** about to run job . . ."
"$@"
