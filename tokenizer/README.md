```
python tokenizer/dump_raw_text.py
```

```
spm_train \
  --input=./tmp/raw_text.txt \
  --model_prefix=fast_text_spm \
  --vocab_size=500294 \
  --vocabulary_output_piece_score=false \
  --character_coverage=1.0 \
  --model_type=unigram
```