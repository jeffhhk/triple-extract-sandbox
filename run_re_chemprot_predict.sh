#!/bin/bash
adirRepo=$( cd $( dirname "$0" ) && pwd )

# Synopsis: test RE model using run_bluebert.py using ChemProt data from BioCreative VI
#
# Prerequisites:
#   See README.md
#

cd "$adirRepo"/bluebert
mkdir -p data/output
BlueBERT_DIR=$(pwd)/data
DATASET_DIR=$(pwd)/../BLUE_Benchmark/data/data/ChemProt
OUTPUT_DIR=$(pwd)/data/output
time env PYTHONPATH=$(pwd) python bluebert/run_bluebert.py \
    --do_train=false \
    --do_eval=false \
    --do_predict=true \
    --task_name="chemprot" \
    --vocab_file=$BlueBERT_DIR/vocab.txt \
    --bert_config_file=$BlueBERT_DIR/bert_config.json \
    --init_checkpoint=$BlueBERT_DIR/bert_model.ckpt \
    --num_train_epochs=10.0 \
    --data_dir=$DATASET_DIR \
    --output_dir=$OUTPUT_DIR \
    --do_lower_case=true \
    2>&1
