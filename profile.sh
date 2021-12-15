VERSION="$(date '+%Y-%m-%d-%H-%M-%S')"
python -m cProfile -s time train.py --epochs=1 | tee log/train.$VERSION.log