import math
import tensorflow as tf


class WarmUpAndCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Applies a warmup schedule on a given learning rate decay schedule."""

    def __init__(self, config, name=None):
        super(WarmUpAndCosineDecay, self).__init__()
        self.config = config
        self.base_learning_rate = self.config["training.LR"]
        self._name = name

    def __call__(self, step):
        with tf.name_scope(self._name or 'WarmUpAndCosineDecay'):
            warmup_steps = self.config["training.WarmupSteps"]

            if self.config["training.LRScaling"].lower() == 'linear':
                scaled_lr = self.base_learning_rate * self.config["training.BatchSize"] / 256.
            elif self.config["training.LRScaling"].lower() == 'sqrt':
                scaled_lr = self.base_learning_rate * math.sqrt(self.config["training.BatchSize"])
            else:
                raise ValueError('Unknown LR scaling {}'.format(self.config["training.LRScaling"]))

            learning_rate = (step / float(warmup_steps) * scaled_lr if warmup_steps else scaled_lr)

            # Cosine decay LR schedule
            total_steps = self.config["training.MaxTrainSteps"]

            # TODO(srbs): Cache this object.
            cosine_decay = tf.keras.experimental.CosineDecay(scaled_lr, total_steps - warmup_steps)
            learning_rate = tf.where(step < warmup_steps, learning_rate, cosine_decay(step - warmup_steps))
            return learning_rate

    def get_config(self):
        return {'config': self.config}
