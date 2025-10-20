import tensorflow as tf


class CustomTrainStep(tf.keras.Model):
    def __init__(self, n_gradients, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_gradients = tf.constant(n_gradients, dtype=tf.int32)
        self.n_acum_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.gradient_accumulation = [tf.Variable(tf.zeros_like(v, dtype=tf.float32), trainable=False) for v in
                                      self.trainable_variables]

    def train_step(self, data):
        self.n_acum_step.assign_add(1)

        x, y = data
        # Gradient Tape
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        # Calculate batch gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        # Accumulate batch gradients
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign_add(gradients[i])

        # If n_acum_step reach the n_gradients then we apply accumulated gradients to update the variables otherwise do nothing
        tf.cond(tf.equal(self.n_acum_step, self.n_gradients), self.apply_accu_gradients, lambda: None)

        # update metrics
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def apply_accu_gradients(self):
        # apply accumulated gradients
        # Accumulate batch gradients
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i] / tf.cast(self.n_acum_step, dtype=tf.float32)
        self.optimizer.apply_gradients(zip(self.gradient_accumulation, self.trainable_variables))

        # reset
        self.n_acum_step.assign(0)
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign(tf.zeros_like(self.trainable_variables[i], dtype=tf.float32))


# Model
input = tf.keras.Input(shape=(28, 28))
base_maps = tf.keras.layers.Flatten(input_shape=(28, 28))(input)
base_maps = tf.keras.layers.Dense(128, activation='relu')(base_maps)
base_maps = tf.keras.layers.Dense(units=10, activation='softmax', name='primary')(base_maps)
custom_model = CustomTrainStep(n_gradients=10, inputs=[input], outputs=[base_maps])

# bind all
custom_model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy'],
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))

# data
(x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
x_train = tf.divide(x_train, 255)
y_train = tf.one_hot(y_train, depth=10)

# customized fit
custom_model.fit(x_train, y_train, batch_size=6, epochs=3, verbose=1)