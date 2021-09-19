# examples/text_classification/rnn/train.py
from functools import partial
import argparse
import random

import numpy as np
import paddle
from paddlenlp.data import Pad, Stack, Tuple
import sentencepiece as spm

from model import FastText
from utils import convert_example, create_dataloader
from datasets.yahoo_answers import YahooAnswers
from datasets.amazon_reviews import AmazonReviews

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--emb_dim", type=int, default=10, help="Size of word embeddings.")
parser.add_argument("--epochs", type=int, default=5, help="Number of epoches for training.")
parser.add_argument("--lr", type=float, default=0.05, help="Learning rate used to train.")
parser.add_argument("--batch_size", type=int, default=1024, help="Total examples' number of a batch for training.")
parser.add_argument("--spm_model_file", type=str, default='./data/fast_text_spm.model', help="Path to the SentencePiece tokenizer model.")
parser.add_argument("--save_dir", type=str, default='./checkpoints/', help="Directory to save model checkpoint")
parser.add_argument("--init_from_ckpt", type=str, default=None, help="The path of checkpoint to be loaded.")
parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument('--dataset', choices=['yahoo_answers', 'amazon_reviews'], default="yahoo_answers", help="Select which dataset to train model, defaults to yahoo_answers.")
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
    if args.dataset == "yahoo_answers":
        train_ds, dev_ds = YahooAnswers().read_datasets(
            splits=['train', 'test'])
    elif args.dataset == "amazon_reviews":
        train_ds, dev_ds = AmazonReviews().read_datasets(
            splits=['train', 'test'])

    num_classes = len(train_ds.label_list)
    vocab_size = 500294

    # Constructs the newtork.
    model = FastText(vocab_size, num_classes, args.emb_dim)
    model = paddle.Model(model)

    # Reads data and generates mini-batches.
    tokenizer = spm.SentencePieceProcessor(model_file=args.spm_model_file)
    trans_fn = partial(convert_example, tokenizer=tokenizer, is_test=False)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=0),  # input_ids
        Stack(dtype="int64"),  # seq len
        Stack(dtype="int64")  # label
    ): [data for data in fn(samples)]
    train_loader = create_dataloader(
        train_ds,
        trans_fn=trans_fn,
        batch_size=args.batch_size,
        mode='train',
        batchify_fn=batchify_fn)
    dev_loader = create_dataloader(
        dev_ds,
        trans_fn=trans_fn,
        batch_size=args.batch_size,
        mode='validation',
        batchify_fn=batchify_fn)

    optimizer = paddle.optimizer.Adam(
        parameters=model.parameters(), learning_rate=args.lr)

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