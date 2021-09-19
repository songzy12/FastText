import os
import collections

import pandas as pd
from paddlenlp.utils.env import DATA_HOME
from paddlenlp.datasets import DatasetBuilder

__all__ = ['AmazonReviews']


class AmazonReviews(DatasetBuilder):
    """
    The data is from http://snap.stanford.edu/data/web-Amazon-links.html.

    >>> all_ds = AmazonReviews().read_datasets(splits=['all'])
    >>> len(all_ds.data) == 34686770

    >>> from collections import Counter
    >>> counter = Counter([x['label'] for x in all_ds.data])
    """
    META_INFO = collections.namedtuple('META_INFO', ('file', 'md5'))
    SPLITS = {
        'train': META_INFO(os.path.join('amazon_reviews', 'train.csv'), None),
        'test': META_INFO(os.path.join('amazon_reviews', 'test.csv'), None),
    }

    def _get_data(self, mode, **kwargs):
        default_root = os.path.join(DATA_HOME, self.__class__.__name__)
        filename, _ = self.SPLITS[mode]
        fullname = os.path.join(default_root, filename)

        return fullname

    def _read(self, filename, *args):
        data = pd.read_csv(filename, header=None)
        for score, review_title, review_text in data.values:
            text = ''
            if type(review_title) is str:
                text += '\n' + review_title
            if type(review_text) is str:
                text += '\n' + review_text
            yield {"text": text, "label": score}

    def get_labels(self):
        return range(1, 6)
