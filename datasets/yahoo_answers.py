import os
import collections

import pandas as pd
from paddlenlp.utils.env import DATA_HOME
from paddlenlp.datasets import DatasetBuilder

__all__ = ['YahooAnswers']


class YahooAnswers(DatasetBuilder):
    """
    The data is from https://arxiv.org/pdf/1509.01626.pdf.
    """
    META_INFO = collections.namedtuple('META_INFO', ('file', 'md5'))
    SPLITS = {
        'train': META_INFO(
            os.path.join('yahoo_answers', 'train.csv'), None),
        'test':
        META_INFO(os.path.join('yahoo_answers', 'test.csv'), None)
    }

    def _get_data(self, mode, **kwargs):
        default_root = os.path.join(DATA_HOME, self.__class__.__name__)
        filename, _ = self.SPLITS[mode]
        fullname = os.path.join(default_root, filename)

        return fullname

    def _read(self, filename, *args):
        data = pd.read_csv(filename, header=None)
        for class_index, question_title, question_content, best_answer in data.values:
            text = question_title
            if type(question_content) is str:
                text += '\n' + question_content
            if type(best_answer) is str:
                text += '\n' + best_answer
            yield {"text": text, "label": class_index}

    def get_labels(self):
        return range(1, 11)