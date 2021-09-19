import os
import collections

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
        'all': META_INFO(os.path.join('amazon_reviews', 'all.txt'), None),
    }

    def _get_data(self, mode, **kwargs):
        default_root = os.path.join(DATA_HOME, self.__class__.__name__)
        filename, _ = self.SPLITS[mode]
        fullname = os.path.join(default_root, filename)

        return fullname

    def _is_valid(self, entry):
        return ('review/score' in entry) and ('review/summary' in entry or
                                              'review/text' in entry)

    def _get_example(self, entry):
        return {
            'text': entry.get('review/summary', '') + '\n' + \
                entry.get('review/text', ''),
            'label': entry['review/score']
        }

    def _read(self, filename, *args):
        with open(filename, 'r') as f:
            entry = {}
            for l in f:
                l = l.strip()
                colonPos = l.find(':')
                if colonPos == -1:
                    if self._is_valid(entry):
                        yield self._get_example(entry)
                    entry = {}
                    continue
                eName = l[:colonPos]
                rest = l[colonPos + 2:]
                entry[eName] = rest
            if self._is_valid(entry):
                yield self._get_example(entry)

    def get_labels(self):
        return ['1.0', '2.0', '3.0', '4.0', '5.0']
