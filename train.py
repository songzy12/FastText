# examples/text_classification/rnn/train.py
import argparse
from functools import partial
import random
import sys
import time

import numpy as np
import paddle
from paddlenlp.data import Pad, Stack, Tuple
import sentencepiece as spm

from datasets.yahoo_answers import YahooAnswers
from datasets.amazon_reviews import AmazonReviews
from model import FastText
from optimizer.lr import LinearDecay
import utils

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--emb_dim", type=int, default=10, help="Size of word embeddings.")
parser.add_argument("--epochs", type=int, default=5, help="Number of epoches for training.")
parser.add_argument("--lr", type=float, default=1, help="Learning rate used to train.")
parser.add_argument("--batch_size", type=int, default=1, help="Total examples' number of a batch for training.")
parser.add_argument("--spm_model_file", type=str, default='./data/yahoo_answers.unigram.500000.model', help="Path to the SentencePiece tokenizer model.")
parser.add_argument("--save_dir", type=str, default='./checkpoints/', help="Directory to save model checkpoint")
parser.add_argument("--init_from_ckpt", type=str, default=None, help="The path of checkpoint to be loaded.")
parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument('--dataset', choices=['yahoo_answers', 'amazon_reviews'], default="yahoo_answers", help="Select which dataset to train model, defaults to yahoo_answers.")
parser.add_argument("--log_dir", type=str, default='./log/', help="Directory to save log files.")
args = parser.parse_args()
# yapf: enable

SEED = 1000
VOCAB_SIZE = 500000


def init_paddle(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)

    paddle.set_device(device)
    print("paddle.in_dynamic_mode:", paddle.in_dynamic_mode())


def load_dataset(dataset):
    if dataset == "yahoo_answers":
        train_ds, dev_ds = YahooAnswers().read_datasets(
            splits=['train', 'test'])
    elif dataset == "amazon_reviews":
        train_ds, dev_ds = AmazonReviews().read_datasets(
            splits=['train', 'test'])
    return train_ds, dev_ds


def get_dataloader(train_ds, dev_ds, tokenizer, batch_size):
    trans_fn = partial(
        utils.convert_example, tokenizer=tokenizer, is_test=False)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=-1),  # input_ids
        Stack(dtype="int64"),  # seq len
        Stack(dtype="int64")  # label
    ): [data for data in fn(samples)]
    train_loader = utils.create_dataloader(
        train_ds,
        trans_fn=trans_fn,
        batch_size=batch_size,
        mode='train',
        batchify_fn=batchify_fn)
    dev_loader = utils.create_dataloader(
        dev_ds,
        trans_fn=trans_fn,
        batch_size=batch_size,
        mode='validation',
        batchify_fn=batchify_fn)

    return train_loader, dev_loader


def build_model(vocab_size, num_classes, emb_dim):
    model = FastText(vocab_size, num_classes, emb_dim)
    model = paddle.Model(model)
    return model


def get_optimizer(train_size, epochs, batch_size, lr):
    total_epoch = train_size * epochs / batch_size
    lr_scheduler = LinearDecay(total_epoch=int(total_epoch), learning_rate=lr)
    optimizer = paddle.optimizer.SGD(learning_rate=lr_scheduler,
                                     parameters=model.parameters())
    return optimizer


def init_model(model, optimizer, criterion, metric, init_from_ckpt):
    model.prepare(optimizer, criterion, metric)
    print(model.summary(input_size=[(1, 1), (1, 1)], dtype='int64'))

    if init_from_ckpt:
        model.load(init_from_ckpt)
        print("Loaded checkpoint from %s" % init_from_ckpt)
    return model


if __name__ == "__main__":
    print(args)
    init_paddle(seed=SEED, device=args.device)

    train_ds, dev_ds = load_dataset(args.dataset)
    model = build_model(VOCAB_SIZE, len(train_ds.label_list), args.emb_dim)

    # TODO(songzy): update the tokenizer.
    tokenizer = spm.SentencePieceProcessor(model_file=args.spm_model_file)
    train_loader, dev_loader = get_dataloader(train_ds, dev_ds, tokenizer,
                                              args.batch_size)

    criterion = paddle.nn.CrossEntropyLoss()
    metric = paddle.metric.Accuracy()
    optimizer = get_optimizer(
        len(train_loader), args.epochs, args.batch_size, args.lr)
    model = init_model(model, optimizer, criterion, metric,
                       args.init_from_ckpt)

    callbacks = [
        paddle.callbacks.ProgBarLogger(
            log_freq=1000, verbose=3),
        paddle.callbacks.VisualDL(log_dir=args.log_dir),
        paddle.callbacks.LRScheduler(by_step=True),
    ]
    model.fit(train_loader,
              dev_loader,
              epochs=args.epochs,
              save_dir=args.save_dir,
              callbacks=callbacks)