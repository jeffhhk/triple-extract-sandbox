#!/bin/bash
adirRepo=$( cd $( dirname "$0" ) && cd ../.. && pwd )

# http://redsymbol.net/articles/unofficial-bash-strict-mode/
set -euo pipefail
IFS=$'\n\t'

howto_activate() {
    echo ""
    echo "To activate:"
    echo "    source \"\$(dependencies/uabcluster/echo_adir_activate.sh)\""
}

export LUSCRATCH=$("$adirRepo/dependencies/uabcluster/echo_adir_scratch.sh")

if [[ -e "$LUSCRATCH/venv/install_done" ]]
then
    echo "Virtual environment already created at: $LUSCRATCH/venv"
    echo "To update, delete virtual environment and rerun this script"
    howto_activate
else
    mkdir -p "$LUSCRATCH"
    rm -rf "$LUSCRATCH/venv"
    virtualenv "$LUSCRATCH/venv"
    source "$LUSCRATCH/venv/bin/activate"
    # For tracing info:
    #pip install -r "$adirRepo"/requirements_lock.txt --log /dev/stdout
    pip install -r "$adirRepo"/requirements_lock.txt
    python -m spacy download en_core_web_sm
    touch "$LUSCRATCH/venv/install_done"
    echo "Virtual environment created."
    howto_activate
fi
