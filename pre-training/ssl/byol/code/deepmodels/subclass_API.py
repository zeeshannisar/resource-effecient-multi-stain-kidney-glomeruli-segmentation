import tensorflow as tf
import tensorflow.keras.backend as K


def getmergeaxis():
    """
        getmergeaxis: get the correct merge axis depending on the backend (TensorFlow or Theano) used by Keras. It is
        used in the concatenation of features maps

    :return:  (int) the merge axis
    """
    # Feature maps are concatenated along last axis (for tf backend, 0 for theano)
    if tf.keras.backend.backend() == 'tensorflow':
        merge_axis = -1
    elif tf.keras.backend.backend() == 'theano':
        merge_axis = 0
    else:
        raise Exception('Merge axis for backend %s not defined' % K.backend())

    return merge_axis


class ConvDownModule(tf.keras.layers.Layer):
    def __init__(self, filters, k_size, initializer, padding, strides, batch_norm, name=None):
        super(ConvDownModule, self).__init__()

        if name is not None:
            self._name = name

        self.batch_norm = batch_norm

        self.conv_down = tf.keras.layers.Conv2D(filters=filters, kernel_size=k_size, kernel_initializer=initializer,
                                                padding=padding, strides=strides, use_bias=not batch_norm)

        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=False):
        x = self.conv_down(inputs)
        if self.batch_norm:
            x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        return x


class UnetEncoder(tf.keras.Model):
    def __init__(self, base_filter=64, k_size=3, initializer="glorot_uniform", padding="valid",
                 strides=1, batch_norm=False, modified_arch=False, name=None):
        super(UnetEncoder, self).__init__()

        if name is not None:
            self._name = name

        self.modified_arch = modified_arch

        i = 0
        filters = base_filter * 2 ** i
        self.down_0_1 = ConvDownModule(filters, k_size, initializer, padding, strides, batch_norm, name=f"down_{i}_1")
        self.down_0_2 = ConvDownModule(filters, k_size, initializer, padding, strides, batch_norm, name=f"down_{i}_2")
        self.pool_0 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool_0")

        i = i + 1
        filters = base_filter * 2 ** i
        self.down_1_1 = ConvDownModule(filters, k_size, initializer, padding, strides, batch_norm, name=f"down_{i}_1")
        self.down_1_2 = ConvDownModule(filters, k_size, initializer, padding, strides, batch_norm, name=f"down_{i}_2")
        self.pool_1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool_1")

        i = i + 1
        filters = base_filter * 2 ** i
        self.down_2_1 = ConvDownModule(filters, k_size, initializer, padding, strides, batch_norm, name=f"down_{i}_1")
        self.down_2_2 = ConvDownModule(filters, k_size, initializer, padding, strides, batch_norm, name=f"down_{i}_2")
        self.pool_2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool_2")

        i = i + 1
        filters = base_filter * 2 ** i
        self.down_3_1 = ConvDownModule(filters, k_size, initializer, padding, strides, batch_norm, name=f"down_{i}_1")
        self.down_3_2 = ConvDownModule(filters, k_size, initializer, padding, strides, batch_norm, name=f"down_{i}_2")
        self.pool_3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool_3")

        i = i + 1
        filters = base_filter * 2 ** i
        self.outputs_1 = ConvDownModule(filters, k_size, initializer, padding, strides, batch_norm, name="bridge_1")
        self.outputs_2 = ConvDownModule(filters, k_size, initializer, padding, strides, batch_norm, name="bridge_2")

        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()

    def call(self, inputs, training=False):
        x = self.down_0_1(inputs, training=training)
        x = self.down_0_2(x, training=training)
        skip_0 = x
        x = self.pool_0(x, training=training)

        x = self.down_1_1(x, training=training)
        x = self.down_1_2(x, training=training)
        skip_1 = x
        x = self.pool_1(x, training=training)

        x = self.down_2_1(x, training=training)
        x = self.down_2_2(x, training=training)
        skip_2 = x
        x = self.pool_2(x, training=training)

        x = self.down_3_1(x, training=training)
        x = self.down_3_2(x, training=training)
        skip_3 = x
        x = self.pool_3(x)

        x = self.outputs_1(x, training=training)
        if not self.modified_arch:
            x = self.outputs_2(x, training=training)

        x = self.avg_pool(x)

        return x, skip_0, skip_1, skip_2, skip_3

    def build_graph(self, input_shape=(508, 508, 3)):
        inputs = tf.keras.layers.Input(shape=input_shape)
        outputs, skip_0, skip_1, skip_2, skip_3 = self.call(inputs)
        return tf.keras.Model(inputs=[inputs], outputs=[outputs, skip_0, skip_1, skip_2, skip_3])


class ProjectionHead(tf.keras.Model):
    def __init__(self, hidden_size=2048, projection_size=256, name=None):
        super(ProjectionHead, self).__init__()

        if name is not None:
            self._name = name

        self.fc1 = tf.keras.layers.Dense(hidden_size)
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.Activation("relu")
        self.fc2 = tf.keras.layers.Dense(projection_size)

    def call(self, inputs, training=False):
        x = self.fc1(inputs)
        x = self.bn(x, training=training)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def build_graph(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape)
        outputs = self.call(inputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs)


