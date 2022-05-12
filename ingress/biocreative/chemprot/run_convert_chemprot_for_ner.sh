#!/bin/bash
adirRepo=$( cd $( dirname "$0" ) && cd ../../.. && pwd )

# http://redsymbol.net/articles/unofficial-bash-strict-mode/
set -euo pipefail
IFS=$'\n\t'

cd "$adirRepo"
env PYTHONPATH="$adirRepo"/BLUE_Benchmark python ingress/biocreative/chemprot/to_ner.py \
    BLUE_Benchmark/data/data/ChemProt/original/chemprot_development/chemprot_development_abstracts.tsv \
    BLUE_Benchmark/data/data/ChemProt/original/chemprot_development/chemprot_development_entities.tsv \
    BLUE_Benchmark/data/data/ChemProt/original/chemprot_development/chemprot_development_relations.tsv \
    derived_data/chemprot/ner/devel.tsv

env PYTHONPATH="$adirRepo"/BLUE_Benchmark python ingress/biocreative/chemprot/to_ner.py \
    BLUE_Benchmark/data/data/ChemProt/original/chemprot_training/chemprot_training_abstracts.tsv \
    BLUE_Benchmark/data/data/ChemProt/original/chemprot_training/chemprot_training_entities.tsv \
    BLUE_Benchmark/data/data/ChemProt/original/chemprot_training/chemprot_training_relations.tsv \
    derived_data/chemprot/ner/train.tsv

env PYTHONPATH="$adirRepo"/BLUE_Benchmark python ingress/biocreative/chemprot/to_ner.py \
    BLUE_Benchmark/data/data/ChemProt/original/chemprot_test_gs/chemprot_test_abstracts_gs.tsv \
    BLUE_Benchmark/data/data/ChemProt/original/chemprot_test_gs/chemprot_test_entities_gs.tsv \
    BLUE_Benchmark/data/data/ChemProt/original/chemprot_test_gs/chemprot_test_relations_gs.tsv \
    derived_data/chemprot/ner/test.tsv
