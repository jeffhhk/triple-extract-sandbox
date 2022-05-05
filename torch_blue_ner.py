import os
import logging
import pickle
import time
from transformers import BertConfig, AutoModel
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_url
import torch.optim
from torch import nn
from torch.utils.data import DataLoader, IterableDataset
FILENAME = "config.json"

logging.basicConfig(filename='torch_blue_ner.log', level=logging.INFO)

class Flags():
    def __init__(self):
        self.huggingface_repo_name="bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12"
        self.task_name=None
        self.data_dir=None
        self.output_dir=None
        self.do_lower_case=False
        self.max_seq_length=128
        self.do_train=True
        self.do_eval=False
        self.do_predict=True
        self.do_checkpoint=True
        self.train_batch_size=32
        self.eval_batch_size=8
        self.predict_batch_size=8
        self.learning_rate=5e-5
        self.num_train_epochs=10.0
        self.warmup_proportion=0.1
        self.save_checkpoint_steps=1000
        self.iterations_per_loop=1000
        self.device="cuda" if torch.cuda.is_available() else "cpu"
        self.basename_model="model.h5"


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label
    def to_dict(self):
        return {
            "guid": self.guid,
            "text": self.text,
            "label":self.label
        }
    def __repr__(self):
        return "{}".format(self.to_dict())


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        # self.label_mask = label_mask
    def to_dict(self):
        return {
            "input_ids": self.input_ids,
            "input_mask": self.input_mask,
            "segment_ids": self.segment_ids,
            "label_ids":self.label_ids
        }
    def __repr__(self):
        return "{}".format(self.to_dict())


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file):
        """Reads a BIO data."""
        print("about to open {}".format(input_file))
        with open(input_file) as f:
            lines = []
            words = []
            labels = []
            for line in f:
                contends = line.strip()
                if len(contends) == 0:
                    assert len(words) == len(labels)
                    if len(words) > 30:
                        # split if the sentence is longer than 30
                        while len(words) > 30:
                            tmplabel = labels[:30]
                            for iidx in range(len(tmplabel)):
                                if tmplabel.pop() == 'O':
                                    break
                            l = ' '.join(
                                [label for label in labels[:len(tmplabel) + 1] if len(label) > 0])
                            w = ' '.join(
                                [word for word in words[:len(tmplabel) + 1] if len(word) > 0])
                            lines.append([l, w])
                            words = words[len(tmplabel) + 1:]
                            labels = labels[len(tmplabel) + 1:]

                    if len(words) == 0:
                        continue
                    l = ' '.join([label for label in labels if len(label) > 0])
                    w = ' '.join([word for word in words if len(word) > 0])
                    lines.append([l, w])
                    words = []
                    labels = []
                    continue

                word = line.strip().split()[0]
                label = line.strip().split()[-1]
                words.append(word)
                labels.append(label)
            return lines


class BC5CDRProcessor(DataProcessor):
    def __init__(self, tokenizer) -> None:
        super().__init__()
        self.tokenizer = tokenizer

    def get_train_examples(self, data_dir):
        l1 = self._read_data(os.path.join(data_dir, "train.tsv"))
        l2 = self._read_data(os.path.join(data_dir, "devel.tsv"))
        return self._create_example(l1 + l2, "train")

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "devel.tsv")), "dev"
        )

    def get_test_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        return ["B", "I", "O", "X", "[CLS]", "[SEP]"]

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = line[1]
            label = line[0]
            #print("BC5CDRProcessor read {}".format(InputExample(guid=guid, text=text, label=label)))
            examples.append(InputExample(guid=guid, text=text, label=label))
        return examples


def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, output_dir):
    label_map = {}
    for (i, label) in enumerate(label_list, 1):
        label_map[label] = i
    os.makedirs(output_dir, exist_ok=True)
    label2id_file = os.path.join(output_dir, 'label2id.pkl')
    with open(label2id_file, "wb") as w:
        pickle.dump(label_map, w)
    textlist = example.text.split(' ')
    labellist = example.label.split(' ')
    tokens = []
    labels = []
    for i, word in enumerate(textlist):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        label_1 = labellist[i]
        for m in range(len(token)):
            if m == 0:
                labels.append(label_1)
            else:
                labels.append("X")
    # tokens = tokenizer.tokenize(example.text)
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]
        labels = labels[0:(max_seq_length - 2)]
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")
    segment_ids.append(0)
    # append("O") or append("[CLS]") not sure!
    label_ids.append(label_map["[CLS]"])
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])
    ntokens.append("[SEP]")
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    label_ids.append(label_map["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)
    # label_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # we don't concerned about it!
        label_ids.append(0)
        ntokens.append("**NULL**")
        # label_mask.append(0)
    # print(len(input_ids))
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    # assert len(label_mask) == max_seq_length

    if ex_index < 5:
        logging.info("*** Example ***")
        logging.info("guid: %s" % (example.guid))
        logging.info("tokens: %s" % " ".join(tokens))
            #TOOD? tokenizer.decode(tokens)))
        logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
        # logging.info("label_mask: %s" % " ".join([str(x) for x in label_mask]))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        # label_mask = label_mask
    )
    # write_tokens(ntokens, label_ids, mode)
    return feature


def convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_dir):
    for (ex_index, example) in enumerate(examples):
        e=convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, output_dir)
        # JH: We use the standard DataLoader/Dataset api for batching.  It requires that each input
        # be a single tensor.  This makes sense because it creates batches by contatenating said tensors.
        # Our NerModel takes input_ids, input_mask, and segment_ids as separate inputs.  After batching
        # but before evaluation, we will separate them back out.  Search for "torch.select".
        tX = torch.tensor([e.input_ids, e.input_mask, e.segment_ids], dtype=torch.int64)
        tY = torch.tensor([e.label_ids], dtype=torch.int64).sum(dim=0)
        yield (tX, tY)

class NerIterableDataset(IterableDataset):
    def __init__(self, examples, label_list, max_seq_length, tokenizer, output_dir) -> None:
        super().__init__()
        self.examples = examples
        self.label_list = label_list
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.output_dir = output_dir

    def __iter__(self):
        return convert_examples_to_features(self.examples, self.label_list, self.max_seq_length, self.tokenizer,
                                            self.output_dir)

    def __len__(self):
        return len(self.examples)

def filed_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file, mode=None):
    pass
# JH: no need to write a TFRecord file in the current implementation.

def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    pass
# JH: no need to decode a TFRecord file in the current implementation.

def create_loss_fn(num_classes):
    def fn(logits,y):
        log_probs=nn.Softmax(dim=-1)(logits)
        one_hot_labels = nn.functional.one_hot(y, num_classes=(num_classes+1))
        per_example_loss = (one_hot_labels * log_probs).sum(dim=-1)
        total_loss = per_example_loss.sum()
        return (per_example_loss, total_loss)
    return fn

def eval_fn(pred,y):
        # batch_size=y.shape[0]
        # max_tok=y.shape[1]
        eq = (pred.argmax(2) == y)
        return eq

class NerModel(nn.Module):
    def __init__(self, url_model, device, is_training, batch_size, num_classes) -> None:
        super().__init__()
        self.is_training = is_training
        self.batch_size = batch_size
        self.bert_config = BertConfig.from_pretrained(url_model)
        self.model0 = AutoModel.from_config(self.bert_config)
        self.model0.to(device)

        self.encoder = list(self.model0.modules())[-5]
        self.encoder_parameters = [p
                                   for m in list(self.model0.modules())[0:-4]
                                   for p in m.parameters()]

        self.encoder_out = None
        def hook(module, input_, output):
            self.encoder_out = output
        self.encoder.register_forward_hook(hook)

        self.hidden_size = self.encoder.bias.shape[0]                   # e.g. 768
        self.lin = nn.Linear(self.hidden_size, (num_classes+1))
        self.lin.to(device)
        # Where did this nn.Linear thing come from?
        # Well, in bluebert/run_bluebert_ner.py, in create_model, two calls are made
        # to tf.get_variable, creating variables "output_weights" and "output_bias".
        # Following the variable creation, the weights and bias are applied to form
        # a linear tranformation of the thing we have here called "model0".
        #
        # In this implementation, we have opted instead to use the pytorch framework
        # by instantiating a nn.Linear.

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, input_ids, attention_mask=None, segment_ids=None):
        token_type_ids = None
        self.model0(input_ids, attention_mask, token_type_ids , segment_ids)
        x = self.encoder_out
        if self.is_training:
            x = self.dropout(x)
        logits = self.lin.forward(x)
        return logits

    def parameters(self):
        # TODO:
        #   Lin.parameters obviously needs to be optimized.  Does encoder_parameters also need to be optimized?
        return list(self.encoder_parameters) + list(self.lin.parameters())

# The pytorch way to do TF1 optimization.create_optimizer.
def create_optimizer(nermodel, init_lr, loss=None, num_train_steps=None, num_warmup_steps=None):
    """Creates an optimizer training op."""
    # TODO?: Implements linear warmup. I.e., if global_step < num_warmup_steps, the
    # learning rate will be `global_step/num_warmup_steps * init_lr`.
    params={}
    optimizer = torch.optim.AdamW(nermodel.parameters(), lr=init_lr, betas=(0.9,0.999), eps=1e-6,
        weight_decay=0.01, amsgrad=False)

    # TODO?: optimizer.apply_gradients

    return optimizer

# The pytorch way to do the TF1 estimator.train and estimator.test is to roll
# ordinary functions for train and test.  See e.g.:
#   see https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
def train(device, dataloader, model, loss_fn, optimizer):
    size = len(dataloader)
    model.train()
    for batch, (X,y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        input_ids = torch.select(X,1,0)
        input_mask = torch.select(X,1,1)
        segment_ids = torch.select(X,1,2)

        # Compute prediction error
        t0=time.time()
        pred = model.forward(input_ids, attention_mask=input_mask, segment_ids=segment_ids)
        #print("dt={}".format(time.time()-t0))
        (per_example_loss, loss) = loss_fn(pred, y)
        #print("pred.shape={} y.shape={}".format(pred.shape, y.shape))
        #pred.shape=torch.Size([32, 128, 7]) y.shape=torch.Size([32, 1, 128])

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"batch: {batch} loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(device, dataloader, model, loss_fn):
    size = 0
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            input_ids = torch.select(X, 1, 0)
            input_mask = torch.select(X, 1, 1)
            segment_ids = torch.select(X, 1, 2)

            # Compute prediction error
            t0=time.time()
            pred = model.forward(input_ids, attention_mask=input_mask, segment_ids=segment_ids)
            #print("dt={}".format(time.time()-t0))
            (per_example_loss, loss) = loss_fn(pred, y)
            test_loss += loss.item()
            #print("pred.shape={} y.shape={}".format(pred.shape, y.shape))
            # pred.shape=torch.Size([32, 128, 7]) y.shape=torch.Size([32, 1, 128])
            s = eval_fn(pred,y)
            #print("score.shape={} score={}".format(s.shape, s))
            correct += s.sum().item()
            size += input_mask.count_nonzero().item()
    test_loss /= num_batches
    acc = correct / size
    print(f"Test set: Accuracy: {(100*acc):>0.1f}%, {correct} of {size}, Avg loss: {test_loss:>8f}")

def run_main(flags):
    # TODO?: interpret directory relative to script
    # if flags.output_dir[0] != "/":
    #     flags.output_dir = os.path.join(os.path.dirname(__file__), flags.output_dir)

    device = flags.device
    # TODO:
    #   RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0! (when checking argument for argument index in method wrapper__index_select)
    #device="cpu"
    print(f"Using {device} device")

    tokenizer = AutoTokenizer.from_pretrained(flags.huggingface_repo_name)
    processor = BC5CDRProcessor(tokenizer)
    label_list = processor.get_labels()

    url_model=hf_hub_url(flags.huggingface_repo_name, FILENAME)

    batch_size = flags.train_batch_size if flags.do_train else flags.eval_batch_size
    nermodel = NerModel(url_model, device, flags.do_train, batch_size, len(label_list))
    loss_fn=create_loss_fn(len(label_list))

    optimizer = create_optimizer(nermodel, flags.learning_rate)


    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    test_examples = None

    if flags.do_checkpoint and os.path.exists(os.path.join(flags.output_dir, flags.basename_model)):
        checkpoint = torch.load(os.path.join(flags.output_dir, flags.basename_model))
        nermodel.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Loaded PyTorch Model State from {}".format(flags.basename_model))

    dldr_test=None
    if flags.do_train or flags.do_eval:
        test_examples = processor.get_test_examples(flags.data_dir)
        dldr_test=DataLoader(
            NerIterableDataset(test_examples, label_list, flags.max_seq_length, tokenizer, flags.output_dir),
            batch_size=flags.eval_batch_size, num_workers=0) # single process

    if flags.do_train:
        print("hello main.processor.get_train_examples")
        train_examples = processor.get_train_examples(flags.data_dir)
        num_train_steps = int(
            len(train_examples) / flags.train_batch_size * flags.num_train_epochs)
        num_warmup_steps = int(num_train_steps * flags.warmup_proportion)

        print("***** Running training *****")
        print("  Num examples = %d" % (len(train_examples)))
        print("  Batch size = %d" % (flags.train_batch_size))
        print("  Num steps = %d" % (num_train_steps))

        dldr_train=DataLoader(
            NerIterableDataset(train_examples, label_list, flags.max_seq_length, tokenizer, flags.output_dir),
            batch_size=flags.train_batch_size, num_workers=0) # single process
        test(device, dldr_test, nermodel, loss_fn)
        for t in range(flags.num_train_epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train(device, dldr_train, nermodel, loss_fn, optimizer)
            test(device, dldr_test, nermodel, loss_fn)
            if flags.do_checkpoint:
                torch.save({
                            # 'epoch': epoch,
                            # 'loss': loss,
                            'model_state_dict': nermodel.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()
                            }, os.path.join(flags.output_dir, flags.basename_model))
                print("Saved PyTorch Model State to {}".format(flags.basename_model))

    if flags.do_eval:
        test(device, dldr_test, nermodel, loss_fn)
    if flags.do_predict:
        # TODO
        with open(os.path.join(flags.output_dir, 'label2id.pkl'), 'rb') as rf:
            label2id = pickle.load(rf)
            id2label = {value: key for key, value in label2id.items()}

    print("Done!")

def main():
    flags=Flags()
    run_main(flags)

