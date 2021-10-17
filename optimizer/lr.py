from paddle.optimizer.lr import LRScheduler


class LinearDecay(LRScheduler):
    def __init__(self,
                 total_epoch,
                 learning_rate=1.0,
                 last_epoch=-1,
                 verbose=False):
        self.total_epoch = total_epoch
        super(LinearDecay, self).__init__(learning_rate, last_epoch, verbose)

    def get_lr(self):
        return self.base_lr * (
            self.total_epoch - self.last_epoch) / self.total_epoch
