#!/bin/bash
adirRepo=$( cd $( dirname "$0" ) && cd ../.. && pwd )

# two required parameters:
venv="$1"
reqs="$2"

# http://redsymbol.net/articles/unofficial-bash-strict-mode/
set -euo pipefail
IFS=$'\n\t'

howto_activate() {
    echo ""
    echo "To activate:"
    echo "    source \"$LUSCRATCH/$venv/bin/activate\""
}

export LUSCRATCH=$("$adirRepo/dependencies/uabcluster/echo_adir_scratch.sh")

if [[ -e "$LUSCRATCH/$venv/install_done" ]]
then
    echo "Virtual environment already created at: $LUSCRATCH/$venv"
    echo "To update, delete virtual environment and rerun this script"
    howto_activate
else
    mkdir -p "$LUSCRATCH"
    rm -rf "$LUSCRATCH/$venv"
    virtualenv "$LUSCRATCH/$venv"
    source "$LUSCRATCH/$venv/bin/activate"
    # For tracing info:
    #pip install -r "$adirRepo"/"$reqs" --log /dev/stdout
    pip install -r "$adirRepo"/"$reqs"
    touch "$LUSCRATCH/$venv/install_done"
    echo "Virtual environment created."
    howto_activate
fi
