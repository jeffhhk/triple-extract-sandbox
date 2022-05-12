#!/bin/bash
adirScript=$( cd $( dirname "$0" ) && pwd )

reld="$1"
shift
cd "$reld"

exec "$@"
