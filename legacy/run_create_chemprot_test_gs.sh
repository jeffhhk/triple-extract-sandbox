#!/bin/bash
adirRepo=$( cd $( dirname "$0" ) && cd .. && pwd )

cd "$adirRepo"/BLUE_Benchmark
python blue/gs/create_chemprot_test_gs.py \
    --entities data/data/ChemProt/original/chemprot_test_gs/chemprot_test_entities_gs.tsv \
    --relations data/data/ChemProt/original/chemprot_test_gs/chemprot_test_gold_standard.tsv \
    --output data/data/ChemProt/chemprot_test_gs.tsv
