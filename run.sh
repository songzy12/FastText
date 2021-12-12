 VERSION="$(date '+%Y-%m-%d-%H-%M-%S')"
 python train.py | tee log/train.$VERSION.log