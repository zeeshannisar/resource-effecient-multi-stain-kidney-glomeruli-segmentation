import tensorflow as tf
import numpy as np


class ReduceLROnPlateauAtMinLoss(tf.keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
  """

    def __init__(self, optimizer_LR=None, patience=10, factor=0.1, min_lr=1e-8):
        super(ReduceLROnPlateauAtMinLoss, self).__init__()
        # The number of epoch it has waited when loss is no longer minimum.
        self.optimizer_LR = optimizer_LR
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr

        if not tf.is_tensor(self.optimizer_LR):
            raise ValueError('Need optimizer !')
        if factor >= 1.0:
            raise ValueError('ReduceLROnPlateau ' 'does not support a factor >= 1.0.')

    def on_train_begin(self):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        self.best = np.Inf

    def on_epoch_end(self, epoch, loss, name=""):
        self.current = loss
        if np.less(self.current, self.best):
            self.best = self.current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                old_lr = float(self.optimizer_LR.numpy())
                if old_lr > self.min_lr:
                    new_lr = old_lr * self.factor
                    new_lr = max(new_lr, self.min_lr)
                    print(f"Epoch {epoch:05d}: Reduce LR to from {old_lr} to {new_lr} for {name} model.")
                self.wait = 0
                self.optimizer_LR.assign(new_lr)