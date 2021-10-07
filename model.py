# examples/text_classification/rnn/model.py

import paddle
import paddle.nn as nn

import paddlenlp as nlp

INF = 1. * 1e12


class FastText(nn.Layer):
    """
    This class implements the Bag of Words Classification Network model to classify texts.
    At a high level, the model starts by embedding the tokens and running them through
    a word embedding. Then, we encode these representations with a `BoWEncoder`.
    Lastly, we take the output of the encoder to create a final representation,
    which is passed through some feed-forward layers to output a logits (`output_layer`).
    """

    def __init__(self, vocab_size, num_classes, emb_dim, padding_idx=0):
        super().__init__()
        self.embedder = nn.Embedding(
            vocab_size, emb_dim, padding_idx=padding_idx)
        self.bow_encoder = nlp.seq2vec.BoWEncoder(emb_dim)
        self.output_layer = nn.Linear(emb_dim, num_classes)

    def forward(self, text, seq_len=None):
        # Shape: (batch_size, seq_len, embedding_dim)
        embedded_text = self.embedder(text)

        # Shape: (batch_size, embedding_dim)
        summed = self.bow_encoder(embedded_text)

        # Shape: (batch_size, num_classes)
        logits = self.output_layer(summed)
        return logits
