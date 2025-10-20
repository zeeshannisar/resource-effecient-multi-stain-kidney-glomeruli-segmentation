import tensorflow as tf
import numpy as np


class EarlyStoppingAtMinLoss(tf.keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
  """

    def __init__(self, patience=0):
        super(EarlyStoppingAtMinLoss, self).__init__()
        # The number of epoch it has waited when loss is no longer minimum.
        self.patience = patience
        self.stop_training = False

    def on_train_begin(self, previous_best=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = previous_best

    def on_epoch_end(self, epoch, loss=None):
        self.current = loss
        if np.less(self.current, self.best):
            self.previous_best = self.best
            self.best = self.current
            self.improved_loss = True
            self.wait = 0
        else:
            self.wait += 1
            self.improved_loss = False

            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.stop_training = True

