from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Dropout, Reshape, \
    Activation, Conv2DTranspose, Cropping2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import BatchNormalization
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow import keras
import math


def getvalidinputsize(inp_shape, depth=5, k_size=3, data_format='channels_last'):
    convolutions_per_layer = 2

    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: ', data_format)

    def calculate(dim_size):
        # Calculate what the last feature map size would be with this patch size
        for _ in range(depth - 1):
            dim_size = (dim_size - ((k_size - 1) * convolutions_per_layer)) / 2
        dim_size -= (k_size - 1) * 2

        # Minimum possible size of last feature map
        if dim_size < 4:
            dim_size = 4

        # Round to the next smallest even number
        dim_size = math.floor(dim_size / 2.) * 2
        # Calculate the original patch size to give this (valid) feature map size
        for _ in range(depth - 1):
            dim_size = (dim_size + (k_size - 1) * convolutions_per_layer) * 2
        dim_size += (k_size - 1) * 2

        return int(dim_size)

    if data_format == 'channels_last':
        spatial_dims = range(len(inp_shape))[:-1]
    elif data_format == 'channels_first':
        spatial_dims = range(len(inp_shape))[1:]

    inp_shape = list(inp_shape)
    for d in spatial_dims:
        new_inp_shape_d = calculate(inp_shape[d])
        if new_inp_shape_d > inp_shape[d]:
            raise ValueError(
                'Minimum possible image size is larger than the original image size, increase patch resolution or patch size.')
        inp_shape[d] = new_inp_shape_d

    return tuple(inp_shape)

def Conv2DLayer(input, filters, kernel_initialiser, kernel_size, padding, batchnormalisation, name=None):
    # if isinstance(batchnormalisation, str) and batchnormalisation.lower() == 'false' or batchnormalisation.lower() == 'off' or batchnormalisation.lower() == 'no':
    #    batchnormalisation = False
    use_bias = not batchnormalisation

    output = Conv2D(filters=filters, kernel_initializer=kernel_initialiser,
                    kernel_size=kernel_size, padding=padding, use_bias=use_bias, name=name)(input)
    if batchnormalisation == 'before':
        output = BatchNormalization()(output)
    output = Activation('relu')(output)
    if batchnormalisation == 'after':
        output = BatchNormalization()(output)

    return output


# @keras.saving.register_keras_serializable()
class ConvDownModule(tf.keras.layers.Layer):
    def __init__(self, filters, k_size, initializer, padding, strides, batch_norm, name=None, **kwargs):
        super(ConvDownModule, self).__init__(**kwargs)

        self.filters = filters
        self.k_size = k_size
        self.initializer = initializer
        self.padding = padding
        self.strides = strides
        self.batch_norm = batch_norm
        if name is not None:
            self._name = name

    def build(self, input_shape):
        self.conv_down = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=self.k_size,
            kernel_initializer=self.initializer,
            padding=self.padding,
            strides=self.strides,
            use_bias=not self.batch_norm
        )
        self.bn = tf.keras.layers.BatchNormalization()
        super().build(input_shape)  # Marks the layer as built

    def call(self, inputs, training=False):
        x = self.conv_down(inputs)
        if self.batch_norm:
            x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({'filters': self.filters,
                       'k_size': self.k_size,
                       'initializer': self.initializer,
                       'padding': self.padding,
                       'strides': self.strides,
                       'batch_norm': self.batch_norm,
                       'name': self.name,
                       })
        return config


def SimCLR(simclr_model_path, simclr_model_trainable, inp_shape, depth=5, filter_factor_offset=0,
           initialiser='glorot_uniform', padding='valid', batchnormalisation=False, k_size=3,
           dropout=False, hidden1=512, hidden2=128):
    """
    UNet encoder based SimCLR Network
    """
    if simclr_model_trainable:
        print("\nInitializing the SimCLR based Finetune UNet Network...")
    else:
        print("\nInitializing the SimCLR based Fixed-Features UNet Network...")

    name = 'simclr'
    if padding == 'valid':
        inp_shape = getvalidinputsize(inp_shape, depth, k_size)

    data = Input(shape=inp_shape, dtype=K.floatx())

    base_filter_power = 6
    conv_down = [None] * depth

    def unet_encoder(data, batch_norm):
        for i in range(0, depth - 1):
            filters = 2 ** (base_filter_power + i - filter_factor_offset)
            if i == 0:
                conv_down[i] = Conv2DLayer(data, filters, initialiser, k_size,
                                           padding, batch_norm, name + '_down_' + str(i) + '_1')
            else:

                conv_down[i] = Conv2DLayer(pool, filters, initialiser, k_size, padding,
                                           batch_norm, name + '_down_' + str(i) + '_1')

            conv_down[i] = Conv2DLayer(conv_down[i], filters, initialiser, k_size, padding,
                                       batch_norm, name + '_down_' + str(i) + '_2')

            if dropout and i == depth - 2:
                conv_down[i] = tf.keras.layers.Dropout(0.5)(conv_down[i])

            pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv_down[i])

        # Bridge path

        filters = 2 ** (base_filter_power + (depth - 1) - filter_factor_offset)

        conv = Conv2DLayer(pool, filters, initialiser, k_size, padding, batch_norm, name + '_bridge_1')
        if dropout:
            conv = tf.keras.layers.Dropout(0.5)(conv)

        return conv

    x = unet_encoder(data, batch_norm=batchnormalisation)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(hidden1)(x)
    x = tf.keras.layers.Activation("relu", name="ssl_activation")(x)
    outputs = tf.keras.layers.Dense(hidden2)(x)
    simclr_model = tf.keras.Model(data, outputs, name='SimCLR_model')

    simclr_model.load_weights(simclr_model_path)
    simclr_model.trainable = simclr_model_trainable
    return simclr_model




def BYOL(byol_model_path, byol_model_trainable, inp_shape, depth=5, filter_factor_offset=0, initialiser='glorot_uniform',
         padding='valid', modifiedarch=False, batchnormalisation=False, k_size=3, dropout=False):
    """
    UNet encoder based BYOL Network
    """
    if byol_model_trainable:
        print("\nInitializing the BYOL based Finetune UNet Network...")
    else:
        print("\nInitializing the BYOL based Fixed-Features UNet Network...")

    name = 'byol'
    if padding == 'valid':
        inp_shape = getvalidinputsize(inp_shape, depth, k_size)

    base_filter_power = 6
    conv_down = [None] * depth

    if not batchnormalisation:
        batch_norm = True
    strides = 1

    def unet_encoder(input_shape):
        data = Input(shape=input_shape, dtype=K.floatx())
        for i in range(0, depth - 1):
            filters = 2 ** (base_filter_power + i - filter_factor_offset)
            if i == 0:
                conv_down[i] = ConvDownModule(filters, k_size, initialiser, padding, strides, batch_norm, f"{name}_down_{i}_1")(data)
            else:
                conv_down[i] = ConvDownModule(filters, k_size, initialiser, padding, strides, batch_norm, f"{name}_down_{i}_1")(pool)

            conv_down[i] = ConvDownModule(filters, k_size, initialiser, padding, strides, batch_norm, f"{name}_down_{i}_2")(conv_down[i])

            if dropout and i == depth - 2:
                conv_down[i] = tf.keras.layers.Dropout(0.5)(conv_down[i])

            pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name=f"{name}_pool_{i}")(conv_down[i])

        filters = 2 ** (base_filter_power + (depth - 1) - filter_factor_offset)

        outputs = ConvDownModule(filters, k_size, initialiser, padding, strides, batch_norm, f"{name}_bridge_1")(pool)

        if not modifiedarch:
            outputs = ConvDownModule(filters, k_size, initialiser, padding, strides, batch_norm, f"{name}_bridge_2")(outputs)
        if dropout:
            outputs = tf.keras.layers.Dropout(0.5)(outputs)

        outputs = tf.keras.layers.GlobalAveragePooling2D()(outputs)

        skip_0, skip_1, skip_2, skip_3 = conv_down[0], conv_down[1], conv_down[2], conv_down[3]
        encoder_model = tf.keras.Model(data, [outputs, skip_0, skip_1, skip_2, skip_3], name='Byol_model')
        return encoder_model

    byol_model = unet_encoder(input_shape=inp_shape)
    byol_model.load_weights(byol_model_path)
    byol_model.trainable = byol_model_trainable
    return byol_model


def HRCSCO(hrcsco_model_path, hrcsco_model_trainable, inp_shape, depth=5, filter_factor_offset=0, initialiser='glorot_uniform',
         padding='valid', modifiedarch=False, batchnormalisation=False, k_size=3, dropout=False):
    """
    UNet encoder based HRCSCO Network
    """
    if hrcsco_model_trainable:
        print("\nInitializing the HRCSCO based Finetune UNet Network...")
    else:
        print("\nInitializing the HRCSCO based Fixed-Features UNet Network...")

    if padding == 'valid':
        inp_shape = getvalidinputsize(inp_shape, depth, k_size)

    base_filter_power = 6
    conv_down = [None] * depth

    def unet_encoder(input_shape, name="H2O"):
        data = Input(shape=input_shape, dtype=K.floatx())
        for i in range(0, depth - 1):
            filters = 2 ** (base_filter_power + i - filter_factor_offset)
            if i == 0:
                conv_down[i] = Conv2DLayer(data, filters, initialiser, k_size, padding, batchnormalisation, f"{name}_down_{i}_1")
            else:
                conv_down[i] = Conv2DLayer(pool, filters, initialiser, k_size, padding, batchnormalisation, f"{name}_down_{i}_1")

            conv_down[i] = Conv2DLayer(conv_down[i], filters, initialiser, k_size, padding, batchnormalisation, f"{name}_down_{i}_2")

            if dropout and i == depth - 2:
                conv_down[i] = tf.keras.layers.Dropout(0.5)(conv_down[i])

            pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv_down[i])

        filters = 2 ** (base_filter_power + (depth - 1) - filter_factor_offset)

        outputs = Conv2DLayer(pool, filters, initialiser, k_size, padding, batchnormalisation, f"{name}_bridge_1")
        if not modifiedarch:
            outputs = Conv2DLayer(outputs, filters, initialiser, k_size, padding, batchnormalisation, f"{name}_bridge_2")
        if dropout:
            outputs = tf.keras.layers.Dropout(0.5)(outputs)

        skip_0, skip_1, skip_2, skip_3 = conv_down[0], conv_down[1], conv_down[2], conv_down[3]
        encoder_model = tf.keras.Model(data, [outputs, skip_0, skip_1, skip_2, skip_3], name=f"HRCSCO_{name}_model")
        return encoder_model

    def HO_UnetEncoder(h_input_shape=(inp_shape[0], inp_shape[1], 1), o_input_shape=(inp_shape[0], inp_shape[1], 1)):
        H2O_encoder = unet_encoder(input_shape=h_input_shape, name="H2O")
        O2H_encoder = unet_encoder(input_shape=o_input_shape, name="O2H")

        h_input, o_input = Input(h_input_shape, dtype=K.floatx()), Input(o_input_shape, dtype=K.floatx())

        h_output, skip0_h, skip1_h, skip2_h, skip3_h = H2O_encoder(h_input)
        o_output, skip0_o, skip1_o, skip2_o, skip3_o = O2H_encoder(o_input)

        HO_encoder_model = tf.keras.Model(inputs=[h_input, o_input], outputs=[h_output, o_output,
                                                                              skip0_h, skip1_h, skip2_h, skip3_h,
                                                                              skip0_o, skip1_o, skip2_o, skip3_o],
                                          name="HRCSCO_model")
        return HO_encoder_model

    hrcsco_model = HO_UnetEncoder()
    hrcsco_model.load_weights(hrcsco_model_path)
    hrcsco_model.trainable = hrcsco_model_trainable
    # hrcsco_H2O_model, hrcsco_O2H_model = hrcsco_model.get_layer('HRCSCO_H2O_model'), hrcsco_model.get_layer('HRCSCO_O2H_model')
    # return hrcsco_H2O_model, hrcsco_O2H_model
    return hrcsco_model