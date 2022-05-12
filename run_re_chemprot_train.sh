#!/bin/bash
adirRepo=$( cd $( dirname "$0" ) && pwd )

# Synopsis: build RE model using run_bluebert.py using ChemProt data from BioCreative VI
#
# Prerequisites:
#   See README.md
#
# Jeff's laptop has 6GB GPU ram, which fails with OOM.  Speed on machine with GPU acceleration is :
#
#   I0319 15:02:38.951204 46912496437696 tpu_estimator.py:2307] global_step/sec: 2.15557
#   INFO:tensorflow:examples/sec: 68.9783
#   I0319 15:02:38.951383 46912496437696 tpu_estimator.py:2308] examples/sec: 68.9783
#
# finishing 16k examples and num_train_epochs=10.0 in 38 minutes.  Without GPU acceleration is about 4/examples/sec,
# which would be around 11 hours of runtime.

cd "$adirRepo"/bluebert
mkdir -p data/output
BlueBERT_DIR=$(pwd)/data
DATASET_DIR=$(pwd)/../BLUE_Benchmark/data/data/ChemProt
OUTPUT_DIR=$(pwd)/data/output
time env PYTHONPATH=$(pwd) python bluebert/run_bluebert.py \
    --do_train=true \
    --do_eval=false \
    --do_predict=false \
    --task_name="chemprot" \
    --vocab_file=$BlueBERT_DIR/vocab.txt \
    --bert_config_file=$BlueBERT_DIR/bert_config.json \
    --init_checkpoint=$BlueBERT_DIR/bert_model.ckpt \
    --num_train_epochs=10.0 \
    --data_dir=$DATASET_DIR \
    --output_dir=$OUTPUT_DIR \
    --do_lower_case=true \
    2>&1
