A PaddlePaddle re-implementation of fastText [1].

## Dataset

https://drive.google.com/file/d/0Bz8a_Dbh9QhbZVhsUnRWRDhETzA/view?resourcekey=0-Rp0ynafmZGZ5MflGmvwLGg

```
$ mv yahoo_answers_csv.tar.gz ~/.paddlenlp/datasets/YahooAnswers/
$ cd ~/.paddlenlp/datasets/YahooAnswers
$ tar -xzvf yahoo_answers_csv.tar.gz
$ mv yahoo_answers_csv yahoo_answers
```

https://drive.google.com/file/d/0Bz8a_Dbh9Qhbd2JNdDBsQUdocVU/view?resourcekey=0-QTryjwu31-qFjMtEAJmsvw

```
$ mv amazon_review_full_csv.tar.gz ~/.paddlenlp/datasets/AmazonReviews/
$ cd ~/.paddlenlp/datasets/AmazonReviews
$ tar -xzvf amazon_review_full_csv.tar.gz
$ mv amazon_review_full_csv amazon_reviews
```

## Install

```
$ pip intall -r requirements.txt
```

## Train

```
$ python train.py
```

## Reference

[1] <https://arxiv.org/pdf/1607.01759.pdf>

## Resource

- <https://github.com/facebookresearch/fastText#text-classification>
