import os
import torch_blue_ner as tbn

# Performs the pytorch operations roughly equivalent to:
#   bash run_ner_train.sh

adirBluebert=os.path.join(os.path.dirname(__file__),"bluebert")
adirDataset=os.path.join(os.path.dirname(__file__),"derived_data","cdr5ner")
adirOutput=os.path.join(os.path.dirname(__file__),"bluebert","data","output_ner")

flags=tbn.Flags()
flags.do_train=True
flags.do_eval=False
flags.do_predict=False
flags.task_name="bc5cdr"
# bert_config is supplied by huggingface
# init_checkpoint is supplied by huggingface
flags.learning_rate=5e-5
flags.num_train_epochs=10
flags.data_dir=adirDataset
flags.output_dir=adirOutput
flags.huggingface_repo_name="bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12"
    # or "bionlp/bluebert_pubmed_uncased_L-24_H-1024_A-16"

# To vary batch size:
#flags.train_batch_size=16


tbn.run_main(flags)
