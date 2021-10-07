```
python tokenizer/dump_raw_text.py
```

```
spm_train \
  --input=./tmp/raw_yahoo_answers.txt \
  --model_prefix=yahoo_answers.unigram.500000 \
  --vocab_size=500000 \
  --vocabulary_output_piece_score=false \
  --character_coverage=1.0 \
  --model_type=unigram
```