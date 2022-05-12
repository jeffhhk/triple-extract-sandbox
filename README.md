# bluebert-sandbox

# Synopsis:

Repository bluebert-sandbox is an attempt to reproduce some of the benchmarks published in:

> [Transfer Learning in Biomedical Natural Language Processing: An Evaluation of BERT and ELMo on Ten Benchmarking Datasets](https://arxiv.org/abs/1906.05474)

# Prerequisites

Tested with:

- python 3.7

# Installation

We recommend use of virtualenv.  If you use a virtualenv, activate it with e.g.:

    source venv/bin/activate

Install packages:

    pip install -r requirements_bluebenchmark.txt
    pip install -r requirements_tensorflow.txt

Clone dependent repositories:

    bash clone_git_repos.sh

Install data:

    bash ensure_data_downloaded_bluebench.sh
    bash ensure_data_downloaded_bluebert.sh

Optional GPU-accelerated dockerfile available.  See build_docker.sh.

# Training

Run:

    run_re_chemprot_train.sh

# Process

- If you get a result to run (or if you have trouble), check in a shell script to share the commands necessary to reproduce the success or failure.
- When possible, capture all necessary dependencies using the [installation](#Installation) instructions in README.md

# Conventions

- Bash and python scripts should have extensions .sh and .py
- Script files beginning with "run_" can be run without arguments
- Scripts should use their own location in the filesystem instead of the current working directory for locating resources.  See use of __file__ in run_torch_blue_ner_train.py or $0 in run_ner_train.sh for details.

# Directory Structure

| directory | description |
|---|---|
| ingress/ | Code for preparing or converting data for processing |
| derived_data/ |

