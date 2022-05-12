#!/bin/bash
adirRepo=$( cd $( dirname "$0" ) && pwd )

cd "$adirRepo"/BLUE_Benchmark
mkdir -p $(pwd)/test1
env PYTHONPATH=$(pwd) python $(pwd)/blue/bert/create_chemprot_bert.py $(pwd)/data/data/ChemProt/original test1

# For some reason, the very similar precomputed data is slightly different than what comes out with this script.

# bash ensure_data_downloaded_bluebench.sh
# bash run_preprocess_bluebench_chemprot.sh
#
# head -4 BLUE_Benchmark/data/data/ChemProt/train.tsv
#     index	sentence	label
#     16357751.T1.T2	Recent studies have provided consistent evidence that treatment with abatacept results in a rapid onset of efficacy that is maintained over the course of treatment in patients with inadequate response to @CHEMICAL$ and anti-@GENE$ therapies.	false
#     14967461.T1.T22	@CHEMICAL$ inhibitors currently under investigation include the small molecules @GENE$ (Iressa, ZD1839) and erlotinib (Tarceva, OSI-774), as well as monoclonal antibodies such as cetuximab (IMC-225, Erbitux).	CPR:4
#     14967461.T2.T22	@CHEMICAL$ inhibitors currently under investigation include the small molecules gefitinib (@GENE$, ZD1839) and erlotinib (Tarceva, OSI-774), as well as monoclonal antibodies such as cetuximab (IMC-225, Erbitux).	CPR:4
# head -4 BLUE_Benchmark/test1/train.tsv
#     index	sentence	label
#     16357751.T1.T2	Recent studies have provided consistent evidence that treatment with abatacept results in a rapid onset of efficacy that is maintained over the course of treatment in patients with inadequate response to @CHEMICAL$ and anti-@GENE$ therapies.	false
#     14967461.T1.T22	@CHEMICAL$ inhibitors currently under investigation include the small molecules @GENE$ (Iressa, ZD1839) and erlotinib (Tarceva, OSI-774), as well as monoclonal antibodies such as cetuximab (IMC-225, Erbitux).	CPR:4
#     14967461.T2.T22	@CHEMICAL$ inhibitors currently under investigation include the small molecules gefitinib (@GENE$, ZD1839) and erlotinib (Tarceva, OSI-774), as well as monoclonal antibodies such as cetuximab (IMC-225, Erbitux).	CPR:4
# wc -l BLUE_Benchmark/data/data/ChemProt/train.tsv
#     19461 BLUE_Benchmark/data/data/ChemProt/train.tsv
# wc -l BLUE_Benchmark/test1/train.tsv
#     19065 BLUE_Benchmark/test1/train.tsv
