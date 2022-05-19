#!/bin/bash
adirRepo=$( cd $( dirname "$0" ) && pwd )

# http://redsymbol.net/articles/unofficial-bash-strict-mode/
set -euo pipefail
IFS=$'\n\t'

if [[ ! -e "$adirRepo"/bluebert ]]
then
    git clone --branch master https://github.com/ncbi-nlp/bluebert "$adirRepo"/bluebert
fi
if [[ ! -e "$adirRepo"/BLUE_Benchmark ]]
then
    git clone --branch master https://github.com/ncbi-nlp/BLUE_Benchmark "$adirRepo"/BLUE_Benchmark
fi
if [[ ! -e "$adirRepo"/transformerssrc ]]
then
    git clone --branch v4.18.0 https://github.com/huggingface/transformers "$adirRepo"/transformerssrc
fi
