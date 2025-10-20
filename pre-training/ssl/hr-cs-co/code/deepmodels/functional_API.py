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


def Conv2DLayer(inputs, filters, initializer, kernel_size, padding, batch_norm, name=None):
    use_bias = not batch_norm

    outputs = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, use_bias=use_bias,
                                     kernel_initializer=initializer, name=name)(inputs)
    if batch_norm:
        outputs = tf.keras.layers.BatchNormalization()(outputs)

    outputs = tf.keras.layers.Activation('relu')(outputs)

    return outputs


class UnetEncoder:
    def __init__(self, input_shape=(508, 508, 1), k_size=3, padding="valid", batch_norm=False,
                 initializer="glorot_uniform", depth=5, base_filter_power=6, filter_factor_offset=0,
                 modified_arch=False, dropout=False):
        self.input_shape = input_shape
        self.k_size = k_size
        self.padding = padding
        self.batch_norm = batch_norm
        self.initializer = initializer
        self.depth = depth
        self.base_filter_power = base_filter_power
        self.filter_factor_offset = filter_factor_offset
        self.modified_arch = modified_arch
        self.dropout = dropout

    def model(self):
        image_input = tf.keras.Input(shape=self.input_shape)
        conv_down = [None] * self.depth

        for i in range(0, self.depth - 1):
            filters = 2 ** (self.base_filter_power + i - self.filter_factor_offset)
            if i == 0:
                conv_down[i] = Conv2DLayer(image_input, filters, self.initializer, self.k_size,
                                           self.padding, self.batch_norm, f"down_{i}_1")
            else:
                conv_down[i] = Conv2DLayer(pool, filters, self.initializer, self.k_size,
                                           self.padding, self.batch_norm, f"down_{i}_1")

            conv_down[i] = Conv2DLayer(conv_down[i], filters, self.initializer, self.k_size,
                                       self.padding, self.batch_norm, f"down_{i}_2")

            if self.dropout and i == self.depth - 2:
                conv_down[i] = tf.keras.layers.Dropout(0.5)(conv_down[i])

            pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv_down[i])

        filters = 2 ** (self.base_filter_power + (self.depth - 1) - self.filter_factor_offset)

        outputs = Conv2DLayer(pool, filters, self.initializer, self.k_size, self.padding, self.batch_norm, "bridge_1")
        if not self.modified_arch:
            outputs = Conv2DLayer(outputs, filters, self.initializer, self.k_size, self.padding, self.batch_norm,
                                  "bridge_2")
        if self.dropout:
            outputs = tf.keras.layers.Dropout(0.5)(outputs)

        encoder_model = tf.keras.Model(image_input, [outputs, conv_down[0], conv_down[1], conv_down[2], conv_down[3]])
        return encoder_model


class UnetDecoder:
    def __init__(self, input_shape=(24, 24, 1024), skip1_shape=None, skip2_shape=None, skip3_shape=None,
                 skip4_shape=None, k_size=3, padding="valid", batch_norm=False, initializer="glorot_uniform",
                 depth=5, base_filter_power=6, filter_factor_offset=0, learn_upscale=True, num_classes=1):

        self.input_shape = input_shape
        self.skip1_shape = skip1_shape
        self.skip2_shape = skip2_shape
        self.skip3_shape = skip3_shape
        self.skip4_shape = skip4_shape
        self.k_size = k_size
        self.padding = padding
        self.batch_norm = batch_norm
        self.initializer = initializer
        self.depth = depth
        self.base_filter_power = base_filter_power
        self.filter_factor_offset = filter_factor_offset
        self.learn_upscale = learn_upscale
        self.num_classes = num_classes

    def model(self):
        image_input = tf.keras.Input(shape=self.input_shape)
        skip1_input = tf.keras.Input(shape=self.skip1_shape)
        skip2_input = tf.keras.Input(shape=self.skip2_shape)
        skip3_input = tf.keras.Input(shape=self.skip3_shape)
        skip4_input = tf.keras.Input(shape=self.skip4_shape)

        conv_down = [skip1_input, skip2_input, skip3_input, skip4_input]
        conv = image_input
        # kernel_size -1 reduced on each convolution, two convolutions, 2 x 2 maxpooling
        base_crop_size = ((self.k_size - 1) * 2) * 2
        curr_crop_size = base_crop_size
        for i in range(self.depth - 2, -1, -1):
            filters = 2 ** (self.base_filter_power + i - self.filter_factor_offset)

            if self.learn_upscale:
                up = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_initializer=self.initializer,
                                                     kernel_size=(2, 2), strides=(2, 2), padding=self.padding,
                                                     use_bias=not self.batch_norm, name=f"up_trans_{i}")(conv)
                if self.batch_norm == 'before':
                    up = tf.keras.layers.BatchNormalization()(up)
                up = tf.keras.layers.Activation('relu')(up)
                if self.batch_norm == 'after':
                    up = tf.keras.layers.BatchNormalization()(up)
            else:
                up = tf.keras.layers.UpSampling2D(size=(2, 2))(conv)

            if self.padding == 'valid':
                conv_down[i] = tf.keras.layers.Cropping2D(cropping=((curr_crop_size // 2, curr_crop_size // 2),
                                                                    (curr_crop_size // 2, curr_crop_size // 2)))(
                    conv_down[i])
                # 2 x 2 maxpooling mutiplies range of previous crop by 2, plus current crops
                curr_crop_size = (2 * curr_crop_size) + (2 * base_crop_size)

            merged = tf.keras.layers.concatenate([conv_down[i], up], axis=getmergeaxis())
            conv = Conv2DLayer(merged, filters, self.initializer, self.k_size, self.padding, self.batch_norm,
                               name=f"up_{i}_1")
            conv = Conv2DLayer(conv, filters, self.initializer, self.k_size, self.padding, self.batch_norm,
                               name=f"up_{i}_2")

        # Output layer
        outputs = tf.keras.layers.Conv2D(filters=self.num_classes, kernel_initializer=self.initializer, kernel_size=1,
                                         padding=self.padding, activation='sigmoid', name='output')(conv)

        decoder_model = tf.keras.Model([image_input, skip1_input, skip2_input, skip3_input, skip4_input], outputs)
        return decoder_model


class HO_UnetEncoder:
    def __init__(self, h_input_shape=(508, 508, 1), o_input_shape=(508, 508, 1)):
        self.h_input_shape = h_input_shape
        self.o_input_shape = o_input_shape
        self.H2O_encoder = UnetEncoder(input_shape=h_input_shape).model()
        self.O2H_encoder = UnetEncoder(input_shape=o_input_shape).model()

    def model(self):
        h_input, o_input = tf.keras.Input(self.h_input_shape), tf.keras.Input(self.o_input_shape)

        h_output, skip1_h, skip2_h, skip3_h, skip4_h = self.H2O_encoder(h_input)
        o_output, skip1_o, skip2_o, skip3_o, skip4_o = self.O2H_encoder(o_input)

        HO_encoder_model = tf.keras.Model(inputs=[h_input, o_input],
                                          outputs=[h_output, o_output,
                                                   skip1_h, skip2_h, skip3_h, skip4_h,
                                                   skip1_o, skip2_o, skip3_o, skip4_o])
        return HO_encoder_model


class HO_UnetDecoder:
    def __init__(self, h_output_shape=(24, 24, 1024), o_output_shape=(24, 24, 1024),
                 skip1_h_shape=(504, 504, 64), skip2_h_shape=(248, 248, 128),
                 skip3_h_shape=(120, 120, 256), skip4_h_shape=(56, 56, 512),
                 skip1_o_shape=(504, 504, 64), skip2_o_shape=(248, 248, 128),
                 skip3_o_shape=(120, 120, 256), skip4_o_shape=(56, 56, 512), num_classes=1):

        self.h_output_shape, self.o_output_shape = h_output_shape, o_output_shape
        self.skip1_h_shape, self.skip2_h_shape, self.skip3_h_shape, self.skip4_h_shape = skip1_h_shape, skip2_h_shape, \
                                                                                         skip3_h_shape, skip4_h_shape
        self.skip1_o_shape, self.skip2_o_shape, self.skip3_o_shape, self.skip4_o_shape = skip1_o_shape, skip2_o_shape, \
                                                                                         skip3_o_shape, skip4_o_shape

        self.H2O_decoder = UnetDecoder(h_output_shape, skip1_h_shape, skip2_h_shape, skip3_h_shape, skip4_h_shape,
                                       num_classes=num_classes).model()
        self.O2H_decoder = UnetDecoder(o_output_shape, skip1_o_shape, skip2_o_shape, skip3_o_shape, skip4_o_shape,
                                       num_classes=num_classes).model()

    def model(self):
        h_output, o_output = tf.keras.Input(self.h_output_shape), tf.keras.Input(self.o_output_shape)
        skip1_h, skip2_h, skip3_h, skip4_h = tf.keras.Input(self.skip1_h_shape), tf.keras.Input(self.skip2_h_shape), \
                                             tf.keras.Input(self.skip3_h_shape), tf.keras.Input(self.skip4_h_shape)

        skip1_o, skip2_o, skip3_o, skip4_o = tf.keras.Input(self.skip1_o_shape), tf.keras.Input(self.skip2_o_shape), \
                                             tf.keras.Input(self.skip3_o_shape), tf.keras.Input(self.skip4_o_shape)

        o_pred = self.H2O_decoder([h_output, skip1_h, skip2_h, skip3_h, skip4_h])
        h_pred = self.O2H_decoder([o_output, skip1_o, skip2_o, skip3_o, skip4_o])

        HO_decoder_model = tf.keras.Model(inputs=[h_output, o_output,
                                                  skip1_h, skip2_h, skip3_h, skip4_h,
                                                  skip1_o, skip2_o, skip3_o, skip4_o],
                                          outputs=[o_pred, h_pred])
        return HO_decoder_model


class PostProcess_HO_UnetEncoder:
    def __init__(self, H2O_encoder_output_shape=(24, 24, 1024), O2H_encoder_output_shape=(24, 24, 1024)):
        self.H2O_encoder_output_shape = H2O_encoder_output_shape
        self.O2H_encoder_output_shape = O2H_encoder_output_shape

    def model(self):
        H2O_encoder_output = tf.keras.Input(self.H2O_encoder_output_shape)
        O2H_encoder_output = tf.keras.Input(self.O2H_encoder_output_shape)

        output = tf.keras.layers.concatenate([H2O_encoder_output, O2H_encoder_output], axis=getmergeaxis())

        output = tfa.layers.AdaptiveAveragePooling2D(output_size=(1, 1))(output)

        output = tf.keras.layers.Flatten()(output)

        post_process_model = tf.keras.Model(inputs=[H2O_encoder_output, O2H_encoder_output], outputs=output)
        return post_process_model


# MLP class for projector and predictor
class ProjectionHead:
    def __init__(self, input_shape=2048, hidden_size=4096, projection_size=256):
        self.input_shape = input_shape
        self.hidden_size = hidden_size
        self.projection_size = projection_size

    def model(self):
        image_input = tf.keras.Input(shape=self.input_shape)
        outputs = tf.keras.layers.Dense(self.hidden_size)(image_input)
        outputs = tf.keras.layers.BatchNormalization()(outputs)
        outputs = tf.keras.layers.Activation("relu")(outputs)
        outputs = tf.keras.layers.Dense(self.projection_size)(outputs)
        mlp_model = tf.keras.Model(image_input, outputs)
        return mlp_model


# input_shape = (508, 508, 1)
#
# encoder = UnetEncoder(input_shape=input_shape).model()
# print("****************************************** Encoder ******************************************")
# print(encoder.summary())
# decoder = UnetDecoder(input_shape=encoder.output_shape[0][1:], skip_1_shape=encoder.output_shape[1][1:],
#                       skip_2_shape=encoder.output_shape[2][1:], skip_3_shape=encoder.output_shape[3][1:],
#                       skip_4_shape=encoder.output_shape[4][1:], num_classes=input_shape[-1]).model()
# print("****************************************** Decoder ******************************************")
# print(decoder.summary())
#
# combined_encoder = HO_UnetEncoder(input_shape_H=input_shape, input_shape_O=input_shape).model()
# print("****************************************** Combined Encoder ******************************************")
# print(combined_encoder.summary())
# combined_decoder = HO_UnetDecoder(input_shape=UnetEncoder().model().output_shape[0][1:],
#                                   skip_1_shape=UnetEncoder().model().output_shape[1][1:],
#                                   skip_2_shape=UnetEncoder().model().output_shape[2][1:],
#                                   skip_3_shape=UnetEncoder().model().output_shape[3][1:],
#                                   skip_4_shape=UnetEncoder().model().output_shape[4][1:],
#                                   num_classes=input_shape[-1]).model()
# print("****************************************** Combined Decoder ******************************************")
# print(combined_decoder.summary())
#
# pretrained_model_path = "/home/nisar/phd/saved_models/SSL/Nephrectomy/CSCO/unet/02/rep1/cross_stain_prediction/models"
#
# hrcsco = CSCO(input_shape_H=input_shape, input_shape_O=input_shape, pretrained_model_path=pretrained_model_path).model()
# print("****************************************** CSCO ******************************************")
# print(hrcsco.summary())
#
# projection = MLP(input_shape=hrcsco.output_shape[0][1:]).model()
# print("****************************************** Projector ******************************************")
# print(projection.summary())
#
# prediction = MLP(input_shape=projection.output_shape[1:]).model()
# print("****************************************** Predictor ******************************************")
# print(prediction.summary())

# input_shape = (508, 508, 1)
# combined_encoder = HO_UnetEncoder(input_shape_H=input_shape, input_shape_O=input_shape).model()
# print("****************************************** Combined Encoder ******************************************")
# print(combined_encoder.summary())
# combined_decoder = HO_UnetDecoder(input_shape=combined_encoder.output_shape[0][1:],
#                                   skip_1_shape=combined_encoder.output_shape[1][1:],
#                                   skip_2_shape=combined_encoder.output_shape[2][1:],
#                                   skip_3_shape=combined_encoder.output_shape[3][1:],
#                                   skip_4_shape=combined_encoder.output_shape[4][1:],
#                                   num_classes=input_shape[-1]).model()
# print("****************************************** Combined Decoder ******************************************")
# print(combined_decoder.summary())
#
# print(combined_encoder.output_shape[0][1:])
# print(combined_encoder.output_shape[5][1:])
# online_postprocess = PostProcess_HO_UnetEncoder(combined_encoder.output_shape[0][1:],
#                                                 combined_encoder.output_shape[5][1:]).model()
# print(online_postprocess.summary())