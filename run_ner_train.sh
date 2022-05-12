#!/bin/bash
adirRepo=$( cd $( dirname "$0" ) && pwd )

# http://redsymbol.net/articles/unofficial-bash-strict-mode/
set -euo pipefail
IFS=$'\n\t'

# consumes data prepared e.g. by convert_chemprot_for_ner.sh

cd "$adirRepo"/bluebert
BlueBERT_DIR=$(pwd)/data
DATASET_DIR=$(pwd)/../derived_data/cdr5ner
OUTPUT_DIR=$(pwd)/data/output_ner
mkdir -p "${OUTPUT_DIR}"
time env PYTHONPATH=$(pwd) python3 bluebert/run_bluebert_ner.py \
    --do_train=true \
    --do_eval=false \
    --do_predict=false \
    --task_name="bc5cdr" \
    --vocab_file=$BlueBERT_DIR/vocab.txt \
    --bert_config_file=$BlueBERT_DIR/bert_config.json \
    --init_checkpoint=$BlueBERT_DIR/bert_model.ckpt \
    --num_train_epochs=10.0 \
    --data_dir=$DATASET_DIR \
    --output_dir=$OUTPUT_DIR \
    --do_lower_case=true \
    2>&1
# To see batch size vary:
#    --train_batch_size=16 \
