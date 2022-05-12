#!/bin/bash
adirRepo=$( cd $( dirname "$0" ) && pwd )

docker build "$adirRepo" -t bluebert-sandbox
