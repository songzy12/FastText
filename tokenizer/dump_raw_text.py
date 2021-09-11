import argparse

from yahoo_answers import YahooAnswers

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--text_file", type=str, default='./tmp/raw_text.txt', help="File path to save raw text from training dataset.")
args = parser.parse_args()
# yapf: enable

if __name__ == '__main__':
    train_ds = YahooAnswers().read_datasets(splits=['train'])
    with open(args.text_file, 'w') as f:
        for data in train_ds.data:
            f.write(data['text'] + '\n')
