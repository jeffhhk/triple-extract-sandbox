#!/bin/bash
adirRepo=$( cd $( dirname "$0" ) && cd ../.. && pwd )
LUSCRATCH=$("$adirRepo/dependencies/uabcluster/echo_adir_scratch.sh")
echo "$LUSCRATCH/venv/bin/activate"