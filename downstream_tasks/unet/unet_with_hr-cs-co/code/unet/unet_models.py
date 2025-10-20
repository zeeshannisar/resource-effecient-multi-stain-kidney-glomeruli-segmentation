from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Dropout, Reshape, \
    Activation, Conv2DTranspose, Cropping2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import BatchNormalization
import tensorflow.keras.backend as K
import tensorflow as tf
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


def getoutputsize(inp_shape, depth=5, k_size=3, padding='valid', data_format='channels_last'):
    convolutions_per_layer = 2

    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: ', data_format)

    def calculate_bridge_end_size(dim_size):
        # Calculate what the last feature map size would be with this patch size
        dim_size = float(dim_size)
        for _ in range(depth - 1):
            dim_size = (dim_size - ((k_size - 1) * convolutions_per_layer)) / 2
        if not dim_size.is_integer():
            raise ValueError(
                'This input shape produces a non-integer feature map shape, use getvalidinputsize to calculate a valid input shape.')
        dim_size -= (k_size - 1) * 2

        # Minimum possible size of last feature map
        if dim_size < 4:
            dim_size = 4

        # Round to the next smallest even number
        dim_size = math.floor(dim_size / 2.) * 2

        return int(dim_size)

    def calculate_output_size(dim_size):
        # Calculate what the last feature map size would be with this patch size
        dim_size = float(dim_size)
        for _ in range(depth - 1):
            dim_size = (dim_size * 2) - ((k_size - 1) * convolutions_per_layer)
        if not dim_size.is_integer():
            raise ValueError(
                'This input shape produces non-integer feature map size, use getvalidinputsize to calculate a valid input shape.')

        return int(dim_size)

    if data_format == 'channels_last':
        spatial_dims = range(len(inp_shape))[:-1]
    elif data_format == 'channels_first':
        spatial_dims = range(len(inp_shape))[1:]

    inp_shape = list(inp_shape)
    if padding == 'valid':
        otp_shape = [0] * len(inp_shape)
        for d in spatial_dims:
            inp_shape[d] = calculate_bridge_end_size(inp_shape[d])
            otp_shape[d] = calculate_output_size(inp_shape[d])
    else:
        otp_shape = inp_shape

    return tuple(otp_shape)


def getmergeaxis():
    """
        getmergeaxis: get the correct merge axis depending on the backend (TensorFlow or Theano) used by Keras. It is
        used in the concatenation of features maps

    :return:  (int) the merge axis
    """
    # Feature maps are concatenated along last axis (for tf backend, 0 for theano)
    if K.backend() == 'tensorflow':
        merge_axis = -1
    elif K.backend() == 'theano':
        merge_axis = 0
    else:
        raise Exception('Merge axis for backend %s not defined' % K.backend())

    return merge_axis


def get_crop_shape(target, refer):
    # width, the 3rd dimension
    cw = (target.get_shape()[2] - refer.get_shape()[2]).value
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw / 2), int(cw / 2) + 1
    else:
        cw1, cw2 = int(cw / 2), int(cw / 2)
    # height, the 2nd dimension
    ch = (target.get_shape()[1] - refer.get_shape()[1]).value
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch / 2), int(ch / 2) + 1
    else:
        ch1, ch2 = int(ch / 2), int(ch / 2)

    return (ch1, ch2), (cw1, cw2)


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


def baseline_unet_model(inp_shape, nb_classes, depth=5, filter_factor_offset=0, initialiser='glorot_uniform', padding='valid',
               modifiedarch=False, batchnormalisation=False, k_size=3, dropout=False, learnupscale=False,
               learncolour=False):
    """

    build_UNet: build the U-Net model

    Based on:
        Olaf Ronneberger, Philipp Fischer, Thomas Brox, U-Net: Convolutional Networks for Biomedical Image Segmentation
        https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28

    :param inp_shape: (tuple) the dimension of the inputs given to the network
            for example if the tuple is (x,y,z) the patches given will have 3 dimensions of shape x * y * z
    :param nb_classes: (int) the number of classes
    :param depth: (int) the number of layers both in the contraction and expansion paths of the U-Net. The whole network
    therefore has a size: 2 * depth
    :param filter_factor_offset: (int) the factor by which to reduce the number of filters used in each convolution (relative to
    the published number)
    :param initialiser: (string) the method used to generate the random initialisation values (default = glorot_uniform)
    :param modifiedarch: (boolean) if True, remove the second convolution layer between the contraction and expansion paths
    :param batchnormalisation: (boolean) enable or disable batch normalisation
    :param k_size:(int) the size of the convolution kernels
    :return:(Model) the U-Net generated
    """

    print("\nInitializing the Baseline UNet Network...")
    # get the current merge_axis
    merge_axis = getmergeaxis()

    if padding == 'valid':
        inp_shape = getvalidinputsize(inp_shape, depth, k_size)
    otp_shape = getoutputsize(inp_shape, depth, k_size, padding)

    data = Input(shape=inp_shape, dtype=K.floatx())

    base_filter_power = 6

    # if isinstance(batchnormalisation, str) and batchnormalisation.lower() == 'false' or batchnormalisation.lower() == 'off' or batchnormalisation.lower() == 'no':
    #     batchnormalisation = False

    ###
    # Contraction path
    ###
    conv_down = [None] * depth
    for i in range(0, depth - 1):
        filters = 2 ** (base_filter_power + i - filter_factor_offset)
        if i == 0:
            if learncolour:
                data = Conv2D(filters=1, kernel_initializer=initialiser,
                              kernel_size=1, padding='same', use_bias=True)(data)
                data = Activation('sigmoid')(data)
            conv_down[i] = Conv2DLayer(data, filters, initialiser, k_size, padding, batchnormalisation,
                                       'down_' + str(i) + '_1')
        else:
            conv_down[i] = Conv2DLayer(pool, filters, initialiser, k_size, padding, batchnormalisation,
                                       'down_' + str(i) + '_1')

        conv_down[i] = Conv2DLayer(conv_down[i], filters, initialiser, k_size, padding, batchnormalisation,
                                   'down_' + str(i) + '_2')

        if dropout and i == depth - 2:
            conv_down[i] = Dropout(0.5)(conv_down[i])

        pool = MaxPooling2D(pool_size=(2, 2))(conv_down[i])

    ###
    # Bridge
    ###
    filters = 2 ** (base_filter_power + (depth - 1) - filter_factor_offset)

    conv = Conv2DLayer(pool, filters, initialiser, k_size, padding, batchnormalisation, 'bridge_1')
    if not modifiedarch:
        conv = Conv2DLayer(conv, filters, initialiser, k_size, padding, batchnormalisation, 'bridge_2')

    if dropout:
        conv = Dropout(0.5)(conv)

    ###
    # Expansion path
    ###
    base_crop_size = ((
                              k_size - 1) * 2) * 2  # kernel_size -1 reduced on each convolution, two convolutions, 2 x 2 maxpooling
    curr_crop_size = base_crop_size
    for i in range(depth - 2, -1, -1):
        filters = 2 ** (base_filter_power + i - filter_factor_offset)

        if learnupscale:
            up = Conv2DTranspose(filters=filters, kernel_initializer=initialiser,
                                 kernel_size=(2, 2), strides=(2, 2), padding=padding, use_bias=not batchnormalisation,
                                 name='up_trans_' + str(i))(conv)
            if batchnormalisation == 'before':
                up = BatchNormalization()(up)
            up = Activation('relu')(up)
            if batchnormalisation == 'after':
                up = BatchNormalization()(up)
        else:
            up = UpSampling2D(size=(2, 2))(conv)

        # curr_crop_size = conv_down[i].get_shape()[1].value - up.get_shape()[1].value
        # curr_crop_size = get_crop_shape(up, conv_down[i])

        if padding == 'valid':
            conv_down[i] = Cropping2D(
                cropping=((curr_crop_size // 2, curr_crop_size // 2), (curr_crop_size // 2, curr_crop_size // 2)))(
                conv_down[i])
            curr_crop_size = (2 * curr_crop_size) + (
                    2 * base_crop_size)  # 2 x 2 maxpooling mutiplies range of previous crop by 2, plus current crops
        merged = concatenate([conv_down[i], up], axis=merge_axis)

        conv = Conv2DLayer(merged, filters, initialiser, k_size, padding, batchnormalisation,
                           name='up_' + str(i) + '_1')
        conv = Conv2DLayer(conv, filters, initialiser, k_size, padding, batchnormalisation, name='up_' + str(i) + '_2')

    # Classification layer
    out = Conv2D(filters=nb_classes, kernel_initializer=initialiser, kernel_size=1, padding=padding,
                 activation='softmax', name='class_1')(conv)

    model = Model(data, out)

    return model, inp_shape, otp_shape


def build_AE_UNet(inp_shape, nb_classes, depth=5, filter_factor_offset=0, initialiser='glorot_uniform', padding='valid',
                  modifiedarch=False, batchnormalisation=False, k_size=3, dropout=False, learnupscale=False,
                  learncolour=False):
    """

    build_UNet: build the U-Net model

    Based on:
        Olaf Ronneberger, Philipp Fischer, Thomas Brox, U-Net: Convolutional Networks for Biomedical Image Segmentation
        https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28

    :param inp_shape: (tuple) the dimension of the inputs given to the network
            for example if the tuple is (x,y,z) the patches given will have 3 dimensions of shape x * y * z
    :param nb_classes: (int) the number of classes
    :param depth: (int) the number of layers both in the contraction and expansion paths of the U-Net. The whole network
    therefore has a size: 2 * depth
    :param filter_factor_offset: (int) the factor by which to reduce the number of filters used in each convolution (relative to
    the published number)
    :param initialiser: (string) the method used to generate the random initialisation values (default = glorot_uniform)
    :param modifiedarch: (boolean) if True, remove the second convolution layer between the contraction and expansion paths
    :param batchnormalisation: (boolean) enable or disable batch normalisation
    :param k_size:(int) the size of the convolution kernels
    :return:(Model) the U-Net generated
    """

    # get the current merge_axis
    merge_axis = getmergeaxis()

    if padding == 'valid':
        inp_shape = getvalidinputsize(inp_shape, depth, k_size)
    otp_shape = getoutputsize(inp_shape, depth, k_size, padding)

    data = Input(shape=inp_shape, dtype=K.floatx())

    base_filter_power = 6

    ###
    # Contraction path
    ###
    conv_down = [None] * depth
    for i in range(0, depth - 1):
        filters = 2 ** (base_filter_power + i - filter_factor_offset)
        if i == 0:
            if learncolour:
                data = Conv2D(filters=1, kernel_initializer=initialiser,
                              kernel_size=1, padding='same', use_bias=True)(data)
                data = Activation('sigmoid')(data)
            conv_down[i] = Conv2DLayer(data, filters, initialiser, k_size, padding, batchnormalisation,
                                       'down_' + str(i) + '_1')
        else:
            conv_down[i] = Conv2DLayer(pool, filters, initialiser, k_size, padding, batchnormalisation,
                                       'down_' + str(i) + '_1')

        conv_down[i] = Conv2DLayer(conv_down[i], filters, initialiser, k_size, padding, batchnormalisation,
                                   'down_' + str(i) + '_2')

        if dropout and i == depth - 2:
            conv_down[i] = Dropout(0.5)(conv_down[i])

        pool = MaxPooling2D(pool_size=(2, 2))(conv_down[i])

    ###
    # Bridge
    ###
    filters = 2 ** (base_filter_power + (depth - 1) - filter_factor_offset)

    conv = Conv2DLayer(pool, filters, initialiser, k_size, padding, batchnormalisation, 'bridge_1')
    if not modifiedarch:
        conv = Conv2DLayer(conv, filters, initialiser, k_size, padding, batchnormalisation, 'bridge_2')

    if dropout:
        conv = Dropout(0.5)(conv)

    ###
    # Expansion path
    ###
    base_crop_size = ((
                              k_size - 1) * 2) * 2  # kernel_size -1 reduced on each convolution, two convolutions, 2 x 2 maxpooling
    curr_crop_size = base_crop_size
    for i in range(depth - 2, -1, -1):
        filters = 2 ** (base_filter_power + i - filter_factor_offset)

        if learnupscale:
            up = Conv2DTranspose(filters=filters, kernel_initializer=initialiser,
                                 kernel_size=(2, 2), strides=(2, 2), padding=padding, use_bias=not batchnormalisation,
                                 name='up_trans_' + str(i))(conv)
            if batchnormalisation == 'before':
                up = BatchNormalization()(up)
            up = Activation('relu')(up)
            if batchnormalisation == 'after':
                up = BatchNormalization()(up)
        else:
            up = UpSampling2D(size=(2, 2))(conv)

        # curr_crop_size = conv_down[i].get_shape()[1].value - up.get_shape()[1].value
        # curr_crop_size = get_crop_shape(up, conv_down[i])

        if padding == 'valid':
            conv_down[i] = Cropping2D(
                cropping=((curr_crop_size // 2, curr_crop_size // 2), (curr_crop_size // 2, curr_crop_size // 2)))(
                conv_down[i])
            curr_crop_size = (2 * curr_crop_size) + (
                    2 * base_crop_size)  # 2 x 2 maxpooling mutiplies range of previous crop by 2, plus current crops
        merged = concatenate([conv_down[i], up], axis=merge_axis)

        conv = Conv2DLayer(merged, filters, initialiser, k_size, padding, batchnormalisation,
                           name='up_' + str(i) + '_1')
        conv = Conv2DLayer(conv, filters, initialiser, k_size, padding, batchnormalisation, name='up_' + str(i) + '_2')

    # Classification layer
    out = Conv2D(filters=nb_classes, kernel_initializer=initialiser, kernel_size=1, padding='valid',
                 activation='sigmoid', name='class_1')(conv)

    model = Model(data, out)

    return model, inp_shape, otp_shape


def build_UNet_vgg(inp_shape, nb_classes, depth=5, filter_factor_offset=0, initialiser='glorot_uniform', padding='same',
                   batchnormalisation='before', k_size=3, learnupscale=False):
    if not padding == 'same':
        raise ValueError('Padding must be \'same\' when using the VGG UNet')

    trainable = False
    base_filter_power = 5
    conv_down = [None] * depth
    i = 0
    merge_axis = -1
    # VGG arhitektura
    img_input = Input(inp_shape, dtype='float32')
    # Block 1
    vgg = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', trainable=trainable)(img_input)
    vgg = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', trainable=trainable)(vgg)
    conv_down[i] = vgg
    i += 1
    vgg = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', trainable=trainable)(vgg)

    # Block 2
    vgg = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=trainable)(vgg)
    vgg = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', trainable=trainable)(vgg)
    conv_down[i] = vgg
    i += 1

    vgg = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', trainable=trainable)(vgg)

    # Block 3
    vgg = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', trainable=trainable)(vgg)
    vgg = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', trainable=trainable)(vgg)
    vgg = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', trainable=trainable)(vgg)
    conv_down[i] = vgg
    i += 1

    vgg = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', trainable=trainable)(vgg)

    # Block 4
    vgg = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', trainable=trainable)(vgg)
    vgg = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', trainable=trainable)(vgg)
    vgg = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', trainable=trainable)(vgg)
    conv_down[i] = vgg
    i += 1

    vgg = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', trainable=trainable)(vgg)

    # Block 5
    vgg = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', trainable=trainable)(vgg)
    vgg = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', trainable=trainable)(vgg)
    vgg = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', trainable=trainable)(vgg)

    conv = vgg

    base_crop_size = ((
                              k_size - 1) * 2) * 2  # kernel_size -1 reduced on each convolution, two convolutions, 2 x 2 maxpooling
    curr_crop_size = base_crop_size
    for i in range(depth - 2, -1, -1):
        filters = 2 ** (base_filter_power + i - filter_factor_offset)

        if learnupscale:
            up = Conv2DTranspose(filters=filters, kernel_initializer=initialiser,
                                 kernel_size=(2, 2), strides=(2, 2), padding=padding, use_bias=not batchnormalisation)(
                conv)
            if batchnormalisation == 'before':
                up = BatchNormalization()(up)
            up = Activation('relu')(up)
            if batchnormalisation == 'after':
                up = BatchNormalization()(up)
        else:
            up = UpSampling2D(size=(2, 2))(conv)

        # curr_crop_size = conv_down[i].get_shape()[1].value - up.get_shape()[1].value
        # curr_crop_size = get_crop_shape(up, conv_down[i])

        if padding == 'valid':
            conv_down[i] = Cropping2D(
                cropping=((curr_crop_size // 2, curr_crop_size // 2), (curr_crop_size // 2, curr_crop_size // 2)))(
                conv_down[i])
            curr_crop_size = (2 * curr_crop_size) + (
                    2 * base_crop_size)  # 2 x 2 maxpooling mutiplies range of previous crop by 2, plus current crops
        merged = concatenate([conv_down[i], up], axis=merge_axis)

        conv = Conv2DLayer(merged, filters, initialiser, k_size, padding, batchnormalisation)
        conv = Conv2DLayer(conv, filters, initialiser, k_size, padding, batchnormalisation)

    # Classification layer
    out = Conv2D(filters=nb_classes, kernel_initializer=initialiser, kernel_size=1, padding='valid',
                 activation='softmax')(conv)

    model = Model(img_input, out)

    # set VGG weights
    from keras.applications import VGG16

    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=inp_shape)
    vgg_model_layers = vgg_model.layers

    vgg_model_layers = vgg_model_layers[:-1]

    for (i, layer) in enumerate(vgg_model_layers):
        model.layers[i].set_weights(layer.get_weights())

    return model


def build_UNet_from_pretrained(inp_shape, nb_classes, weightfilename, depth=5, filter_factor_offset=0,
                               initialiser='glorot_uniform', padding='same',
                               modifiedarch=False, batchnormalisation=False, k_size=3, dropout=False,
                               learnupscale=False, learncolour=False):
    UNet, _, _ = build_UNet(inp_shape, nb_classes, depth, filter_factor_offset, initialiser, padding, modifiedarch,
                            batchnormalisation, k_size, dropout, learnupscale, learncolour)

    from keras.models import load_model
    pretrained_UNet = load_model(weightfilename)

    upstarted = False
    for layer, pretrained_layer in zip(UNet.layers[:-1], pretrained_UNet.layers[:-1]):
        if layer.name.startswith('up'):
            upstarted = True
        if hasattr(pretrained_layer, 'trainable') and not upstarted:
            layer[i].set_weights(pretrained_layer.get_weights())
            layer[i].trainable = False

    return UNet


def build_UNetAboveCSCO(csco_model_path, csco_model_trainable, inp_shape, nb_classes, depth=5,
                        filter_factor_offset=0, initialiser='glorot_uniform', padding='valid', modifiedarch=False,
                        batchnormalisation=False, k_size=3, dropout=False, learnupscale=False):
    """

    build_UNetAboveCSCO: build the U-Net decoder on top of CSCO UNet Encoder Features

    :param csco_model_path: weight file path for pretrained_model
    :param inp_shape: (tuple) the dimension of the inputs given to the network
            for example if the tuple is (x,y,z) the patches given will have 3 dimensions of shape x * y * z
    :param nb_classes: (int) the number of classes
    :param depth: (int) the number of layers both in the contraction and expansion paths of the U-Net. The whole network
    therefore has a size: 2 * depth
    :param filter_factor_offset: (int) the factor by which to reduce the number of filters used in each convolution (relative to
    the published number)
    :param initialiser: (string) the method used to generate the random initialisation values (default = glorot_uniform)
    :param modifiedarch: (boolean) if True, remove the second convolution layer between the contraction and expansion paths
    :param batchnormalisation: (boolean) enable or disable batch normalisation
    :param k_size:(int) the size of the convolution kernels
    :return:(Model) the U-Net generated
    """

    if csco_model_trainable:
        print("\nInitializing the CSCO based Finetune UNet Network...")
    else:
        print("\nInitializing the CSCO based Fixed-Features UNet Network...")

    # get the current merge_axis
    merge_axis = getmergeaxis()

    if padding == 'valid':
        inp_shape = getvalidinputsize(inp_shape, depth, k_size)

    otp_shape = getoutputsize(inp_shape, depth, k_size, padding)

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
        encoder_model = tf.keras.Model(data, [outputs, skip_0, skip_1, skip_2, skip_3])
        return encoder_model

    def HO_UnetEncoder(h_input_shape=(508, 508, 1), o_input_shape=(508, 508, 1)):
        H2O_encoder = unet_encoder(input_shape=h_input_shape, name="H20")
        O2H_encoder = unet_encoder(input_shape=o_input_shape, name="02H")

        h_input, o_input = Input(h_input_shape, dtype=K.floatx()), Input(o_input_shape, dtype=K.floatx())

        h_output, skip0_h, skip1_h, skip2_h, skip3_h = H2O_encoder(h_input)
        o_output, skip0_o, skip1_o, skip2_o, skip3_o = O2H_encoder(o_input)

        HO_encoder_model = tf.keras.Model(inputs=[h_input, o_input], outputs=[h_output, o_output,
                                                                              skip0_h, skip1_h, skip2_h, skip3_h,
                                                                              skip0_o, skip1_o, skip2_o, skip3_o])
        return HO_encoder_model

    def unet_decoder(pretrained_model):
        h_output, o_output, skip0_h, skip1_h, skip2_h, skip3_h, \
        skip0_o, skip1_o, skip2_o, skip3_o = pretrained_model.output[0], pretrained_model.output[1], \
                                             pretrained_model.output[2], pretrained_model.output[3], \
                                             pretrained_model.output[4], pretrained_model.output[5], \
                                             pretrained_model.output[6], pretrained_model.output[7],\
                                             pretrained_model.output[8], pretrained_model.output[9]

        conv = concatenate([h_output, o_output], axis=merge_axis)
        conv_down[0] = concatenate([skip0_h, skip0_o], axis=merge_axis)
        conv_down[1] = concatenate([skip1_h, skip1_o], axis=merge_axis)
        conv_down[2] = concatenate([skip2_h, skip2_o], axis=merge_axis)
        conv_down[3] = concatenate([skip3_h, skip3_o], axis=merge_axis)

        # kernel_size -1 reduced on each convolution, two convolutions, 2 x 2 maxpooling
        base_crop_size = ((k_size - 1) * 2) * 2
        curr_crop_size = base_crop_size
        for i in range(depth - 2, -1, -1):
            filters = 2 ** (base_filter_power + i - filter_factor_offset)
            if learnupscale:
                up = Conv2DTranspose(filters=filters, kernel_initializer=initialiser,
                                     kernel_size=(2, 2), strides=(2, 2), padding=padding, use_bias=not batchnormalisation,
                                     name='up_trans_' + str(i))(conv)
                if batchnormalisation == 'before':
                    up = BatchNormalization()(up)
                up = Activation('relu')(up)
                if batchnormalisation == 'after':
                    up = BatchNormalization()(up)
            else:
                up = UpSampling2D(size=(2, 2))(conv)

            if padding == 'valid':
                conv_down[i] = Cropping2D(cropping=((curr_crop_size // 2, curr_crop_size // 2),
                                                    (curr_crop_size // 2, curr_crop_size // 2)))(conv_down[i])
                # 2 x 2 max-pooling multiplies range of previous crop by 2, plus current crops
                curr_crop_size = (2 * curr_crop_size) + (2 * base_crop_size)
            merged = concatenate([conv_down[i], up], axis=merge_axis)

            conv = Conv2DLayer(merged, filters, initialiser, k_size, padding, batchnormalisation,
                               name='up_' + str(i) + '_1')
            conv = Conv2DLayer(conv, filters, initialiser, k_size, padding, batchnormalisation, name='up_' + str(i) + '_2')

        # Classification layer
        out = Conv2D(filters=nb_classes, kernel_initializer=initialiser, kernel_size=1, padding=padding,
                     activation='softmax', name='class_1')(conv)

        return Model(pretrained_model.input, out), inp_shape, otp_shape

    pretrained_encoder_model = HO_UnetEncoder()
    pretrained_encoder_model.load_weights(csco_model_path)
    pretrained_encoder_model.trainable = csco_model_trainable
    segmentation_model, inp_shape, otp_shape = unet_decoder(pretrained_model=pretrained_encoder_model)
    return segmentation_model, inp_shape, otp_shape
