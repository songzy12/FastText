# examples/text_classification/rnn/train.py
from functools import partial
import argparse
import os
import random

import numpy as np
import paddle
from paddlenlp.data import JiebaTokenizer, Pad, Stack, Tuple, Vocab
from paddlenlp.datasets import load_dataset

from model import BoWModel
from utils import convert_example, create_dataloader

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--vocab_path", type=str, default="./data/senta_word_dict.txt", help="The directory to dataset.")
parser.add_argument("--save_dir", type=str, default='./checkpoints/', help="Directory to save model checkpoint")
parser.add_argument("--init_from_ckpt", type=str, default=None, help="The path of checkpoint to be loaded.")
parser.add_argument("--epochs", type=int, default=10, help="Number of epoches for training.")
parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate used to train.")
parser.add_argument("--batch_size", type=int, default=64, help="Total examples' number of a batch for training.")
args = parser.parse_args()
# yapf: enable


def set_seed(seed=1000):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


if __name__ == "__main__":
    paddle.set_device(args.device)
    set_seed()

    # Loads dataset.
    train_ds, dev_ds = load_dataset("chnsenticorp", splits=["train", "dev"])
    num_classes = len(train_ds.label_list)

    # Loads vocab.
    if not os.path.exists(args.vocab_path):
        raise RuntimeError('The vocab_path  can not be found in the path %s' %
                           args.vocab_path)
    vocab = Vocab.load_vocabulary(args.vocab_path,
                                  unk_token='[UNK]',
                                  pad_token='[PAD]')
    vocab_size = len(vocab)
    pad_token_id = vocab.to_indices('[PAD]')

    # Constructs the newtork.
    model = BoWModel(vocab_size, num_classes, padding_idx=pad_token_id)
    model = paddle.Model(model)

    # Reads data and generates mini-batches.
    tokenizer = JiebaTokenizer(vocab)
    trans_fn = partial(convert_example, tokenizer=tokenizer, is_test=False)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=vocab.token_to_idx.get('[PAD]', 0)),  # input_ids
        Stack(dtype="int64"),  # seq len
        Stack(dtype="int64")  # label
    ): [data for data in fn(samples)]
    train_loader = create_dataloader(train_ds,
                                     trans_fn=trans_fn,
                                     batch_size=args.batch_size,
                                     mode='train',
                                     batchify_fn=batchify_fn)
    dev_loader = create_dataloader(dev_ds,
                                   trans_fn=trans_fn,
                                   batch_size=args.batch_size,
                                   mode='validation',
                                   batchify_fn=batchify_fn)

    optimizer = paddle.optimizer.Adam(parameters=model.parameters(),
                                      learning_rate=args.lr)

    # Defines loss and metric.
    criterion = paddle.nn.CrossEntropyLoss()
    metric = paddle.metric.Accuracy()

    model.prepare(optimizer, criterion, metric)

    # Loads pre-trained parameters.
    if args.init_from_ckpt:
        model.load(args.init_from_ckpt)
        print("Loaded checkpoint from %s" % args.init_from_ckpt)

    # Starts training and evaluating.
    callback = paddle.callbacks.ProgBarLogger(log_freq=10, verbose=3)
    model.fit(train_loader,
              dev_loader,
              epochs=args.epochs,
              save_dir=args.save_dir,
              callbacks=callback)