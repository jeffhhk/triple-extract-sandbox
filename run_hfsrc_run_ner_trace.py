import os
import sys

# Add our example's directory to the PYTHONPATH
adirNer=os.path.join(os.path.dirname(__file__),"transformerssrc", "examples", "pytorch", "token-classification")
sys.path.append(adirNer)

# Spoof calling our example with command line arguments
import run_ner
sys.argv=[
    __file__,
    "--model_name_or_path", "nreimers/BERT-Small-L-4_H-512_A-8",
    "--dataset_name", "conll2003",
    "--output_dir", "/tmp/test-ner",
    "--do_train",
    "--do_eval"
    ]
run_ner.main()
