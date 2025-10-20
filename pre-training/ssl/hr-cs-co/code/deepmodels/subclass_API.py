import os
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_addons as tfa


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


class ConvUpModule(tf.keras.layers.Layer):
    def __init__(self, filters, k_size, initializer, padding, strides, batch_norm, name=None):
        super(ConvUpModule, self).__init__()

        if name is not None:
            self._name = name

        self.batch_norm = batch_norm

        self.conv_up = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=k_size,
                                                       kernel_initializer=initializer,
                                                       padding=padding, strides=strides, use_bias=not batch_norm)
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=False):
        x = self.conv_up(inputs)
        if self.batch_norm == "before":
            x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        if self.batch_norm == "after":
            x = self.bn(x, training=training)
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

        return x, skip_0, skip_1, skip_2, skip_3

    def build_graph(self, input_shape=(508, 508, 3)):
        inputs = tf.keras.layers.Input(shape=input_shape)
        outputs, skip_0, skip_1, skip_2, skip_3 = self.call(inputs)
        return tf.keras.Model(inputs=[inputs], outputs=[outputs, skip_0, skip_1, skip_2, skip_3])


class UnetDecoder(tf.keras.Model):
    def __init__(self, base_filter=64, k_size=3, initializer="glorot_uniform", padding="valid",
                 strides=2, batch_norm=False, num_classes=1, name=None):
        super(UnetDecoder, self).__init__()

        if name is not None:
            self._name = name

        base_crop_size = ((k_size - 1) * 2) * 2

        curr_crop_size = base_crop_size
        i = 3
        filters = base_filter * 2 ** i
        self.up_3 = ConvUpModule(filters, k_size - 1, initializer, padding, strides, batch_norm, name=f"up_{i}")
        self.crop_3 = tf.keras.layers.Cropping2D(cropping=((curr_crop_size // 2, curr_crop_size // 2),
                                                           (curr_crop_size // 2, curr_crop_size // 2)))
        self.merged_3 = tf.keras.layers.Concatenate(axis=getmergeaxis(), name="merged_3")
        self.up_3_1 = ConvDownModule(filters, k_size, initializer, padding, strides // 2, batch_norm, name=f"up_{i}_1")
        self.up_3_2 = ConvDownModule(filters, k_size, initializer, padding, strides // 2, batch_norm, name=f"up_{i}_2")

        curr_crop_size = (2 * curr_crop_size) + (2 * base_crop_size)
        i = 2
        filters = base_filter * 2 ** i
        self.up_2 = ConvUpModule(filters, k_size - 1, initializer, padding, strides, batch_norm, name=f"up_{i}")
        self.crop_2 = tf.keras.layers.Cropping2D(cropping=((curr_crop_size // 2, curr_crop_size // 2),
                                                           (curr_crop_size // 2, curr_crop_size // 2)))
        self.merged_2 = tf.keras.layers.Concatenate(axis=getmergeaxis(), name="merged_2")
        self.up_2_1 = ConvDownModule(filters, k_size, initializer, padding, strides // 2, batch_norm, name=f"up_{i}_1")
        self.up_2_2 = ConvDownModule(filters, k_size, initializer, padding, strides // 2, batch_norm, name=f"up_{i}_2")

        curr_crop_size = (2 * curr_crop_size) + (2 * base_crop_size)
        i = 1
        filters = base_filter * 2 ** i
        self.up_1 = ConvUpModule(filters, k_size - 1, initializer, padding, strides, batch_norm, name=f"up_{i}")
        self.crop_1 = tf.keras.layers.Cropping2D(cropping=((curr_crop_size // 2, curr_crop_size // 2),
                                                           (curr_crop_size // 2, curr_crop_size // 2)))
        self.merged_1 = tf.keras.layers.Concatenate(axis=getmergeaxis(), name="merged_1")
        self.up_1_1 = ConvDownModule(filters, k_size, initializer, padding, strides // 2, batch_norm, name=f"up_{i}_1")
        self.up_1_2 = ConvDownModule(filters, k_size, initializer, padding, strides // 2, batch_norm, name=f"up_{i}_2")

        curr_crop_size = (2 * curr_crop_size) + (2 * base_crop_size)
        i = 0
        filters = base_filter * 2 ** i
        self.up_0 = ConvUpModule(filters, k_size - 1, initializer, padding, strides, batch_norm, name=f"up_{i}")
        self.crop_0 = tf.keras.layers.Cropping2D(cropping=((curr_crop_size // 2, curr_crop_size // 2),
                                                           (curr_crop_size // 2, curr_crop_size // 2)))
        self.merged_0 = tf.keras.layers.Concatenate(axis=getmergeaxis(), name="merged_0")
        self.up_0_1 = ConvDownModule(filters, k_size, initializer, padding, strides // 2, batch_norm, name=f"up_{i}_1")
        self.up_0_2 = ConvDownModule(filters, k_size, initializer, padding, strides // 2, batch_norm, name=f"up_{i}_2")

        self.outputs = tf.keras.layers.Conv2D(filters=num_classes, kernel_initializer=initializer, kernel_size=1,
                                              padding=padding, activation='sigmoid', name='output')

    def call(self, inputs, skip_0, skip_1, skip_2, skip_3, training=False):
        x = self.up_3(inputs, training=training)
        skip_3 = self.crop_3(skip_3)
        x = self.merged_3([x, skip_3])
        x = self.up_3_1(x, training=training)
        x = self.up_3_2(x, training=training)

        x = self.up_2(x, training=training)
        skip_2 = self.crop_2(skip_2)
        x = self.merged_2([x, skip_2])
        x = self.up_2_1(x, training=training)
        x = self.up_2_2(x, training=training)

        x = self.up_1(x, training=training)
        skip_1 = self.crop_1(skip_1)
        x = self.merged_1([x, skip_1])
        x = self.up_1_1(x, training=training)
        x = self.up_1_2(x, training=training)

        x = self.up_0(x, training=training)
        skip_0 = self.crop_0(skip_0)
        x = self.merged_0([x, skip_0])
        x = self.up_0_1(x, training=training)
        x = self.up_0_2(x, training=training)

        x = self.outputs(x)

        return x

    def build_graph(self, input_shape, skip_1_shape, skip_2_shape, skip_3_shape, skip_4_shape):
        inputs = tf.keras.layers.Input(shape=input_shape)
        skip_1 = tf.keras.layers.Input(shape=skip_1_shape)
        skip_2 = tf.keras.layers.Input(shape=skip_2_shape)
        skip_3 = tf.keras.layers.Input(shape=skip_3_shape)
        skip_4 = tf.keras.layers.Input(shape=skip_4_shape)
        outputs = self.call(inputs, skip_1, skip_2, skip_3, skip_4)
        return tf.keras.Model(inputs=[inputs, skip_1, skip_2, skip_3, skip_4], outputs=[outputs])


class HO_UnetEncoder(tf.keras.Model):
    def __init__(self, name=None):
        super(HO_UnetEncoder, self).__init__()

        if name is not None:
            self._name = name

        self.H2O_encoder = UnetEncoder(name="H2O_encoder")
        self.O2H_encoder = UnetEncoder(name="O2H_encoder")

    def call(self, h, o, training=False):
        h_encoder_output, h_skip_0, h_skip_1, h_skip_2, h_skip_3 = self.H2O_encoder(h, training=training)
        o_encoder_output, o_skip_0, o_skip_1, o_skip_2, o_skip_3 = self.O2H_encoder(o, training=training)
        return h_encoder_output, o_encoder_output, h_skip_0, h_skip_1, h_skip_2, h_skip_3, o_skip_0, o_skip_1, o_skip_2, o_skip_3

    def build_graph(self, h_input_shape, o_input_shape):
        h_inputs = tf.keras.layers.Input(shape=h_input_shape)
        o_inputs = tf.keras.layers.Input(shape=o_input_shape)
        h_encoder_outputs, o_encoder_outputs, h_skip_0, h_skip_1, h_skip_2, h_skip_3, o_skip_0, o_skip_1, o_skip_2, o_skip_3 = self.call(
            h_inputs, o_inputs)
        return tf.keras.Model(inputs=[h_inputs, o_inputs], outputs=[h_encoder_outputs, o_encoder_outputs,
                                                                    h_skip_0, h_skip_1, h_skip_2, h_skip_3,
                                                                    o_skip_0, o_skip_1, o_skip_2, o_skip_3])


class HO_UnetDecoder(tf.keras.Model):
    def __init__(self, name=None):
        super(HO_UnetDecoder, self).__init__()

        if name is not None:
            self._name = name

        self.H2O_decoder = UnetDecoder(name="H2O_decoder")
        self.O2H_decoder = UnetDecoder(name="O2H_decoder")

    def call(self, h_encoder_output, o_encoder_output, h_skip_0, h_skip_1, h_skip_2, h_skip_3,
             o_skip_0, o_skip_1, o_skip_2, o_skip_3, training=False):
        h2o_output = self.H2O_decoder(h_encoder_output, h_skip_0, h_skip_1, h_skip_2, h_skip_3, training=training)
        o2h_output = self.O2H_decoder(o_encoder_output, o_skip_0, o_skip_1, o_skip_2, o_skip_3, training=training)
        return h2o_output, o2h_output

    def build_graph(self, h_encoder_output_shape, o_encoder_output_shape, h_skip_0_shape, h_skip_1_shape,
                    h_skip_2_shape, h_skip_3_shape, o_skip_0_shape, o_skip_1_shape, o_skip_2_shape, o_skip_3_shape):
        h_encoder_outputs = tf.keras.layers.Input(shape=h_encoder_output_shape)
        o_encoder_outputs = tf.keras.layers.Input(shape=o_encoder_output_shape)
        h_skip_0 = tf.keras.layers.Input(shape=h_skip_0_shape)
        h_skip_1 = tf.keras.layers.Input(shape=h_skip_1_shape)
        h_skip_2 = tf.keras.layers.Input(shape=h_skip_2_shape)
        h_skip_3 = tf.keras.layers.Input(shape=h_skip_3_shape)
        o_skip_0 = tf.keras.layers.Input(shape=o_skip_0_shape)
        o_skip_1 = tf.keras.layers.Input(shape=o_skip_1_shape)
        o_skip_2 = tf.keras.layers.Input(shape=o_skip_2_shape)
        o_skip_3 = tf.keras.layers.Input(shape=o_skip_3_shape)

        h2o_outputs, o2h_outputs = self.call(h_encoder_outputs, o_encoder_outputs, h_skip_0, h_skip_1, h_skip_2,
                                             h_skip_3, o_skip_0, o_skip_1, o_skip_2, o_skip_3)

        return tf.keras.Model(inputs=[h_encoder_outputs, o_encoder_outputs, h_skip_0, h_skip_1, h_skip_2, h_skip_3,
                                      o_skip_0, o_skip_1, o_skip_2, o_skip_3], outputs=[h2o_outputs, o2h_outputs])


class ProjectionHead(tf.keras.Model):
    def __init__(self, hidden_size=4096, projection_size=256, name=None):
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


class PostProcessHOEncoder(tf.keras.Model):
    def __init__(self, name=None):
        super(PostProcessHOEncoder, self).__init__()

        if name is not None:
            self._name = name

        self.avg_pool = tfa.layers.AdaptiveAveragePooling2D(output_size=(1, 1))
        self.concat = tf.keras.layers.Concatenate(axis=getmergeaxis())
        self.flatten = tf.keras.layers.Flatten()

    def call(self, h_encoder_output, o_encoder_output, training=False):
        h_encoder_output_pool = self.avg_pool(h_encoder_output)
        o_encoder_output_pool = self.avg_pool(o_encoder_output)
        output = self.concat([h_encoder_output_pool, o_encoder_output_pool])
        output = self.flatten(output)
        return output

    def build_graph(self, h_encoder_output_shape, o_encoder_output_shape):
        h_encoder_output_inputs = tf.keras.Input(shape=h_encoder_output_shape)
        o_encoder_output_inputs = tf.keras.Input(shape=o_encoder_output_shape)
        outputs = self.call(h_encoder_output_inputs, o_encoder_output_inputs)
        return tf.keras.Model(inputs=[h_encoder_output_inputs, o_encoder_output_inputs], outputs=outputs)