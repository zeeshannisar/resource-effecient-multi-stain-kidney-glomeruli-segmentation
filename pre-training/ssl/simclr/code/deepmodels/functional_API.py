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


def validate_input(x):
    if x.shape.as_list()[1:-1] != [512, 512]:
        pad = [(512 - int(x.shape[1])) // 2, (512 - int(x.shape[2])) // 2]
        paddings = tf.constant([[0, 0], pad, pad, [0, 0]])
        return tf.pad(x, paddings=paddings, mode='REFLECT', constant_values=0)
    else:
        return x


def one_side_pad(x):
    x = tf.keras.layers.ZeroPadding2D((1, 1))(x)
    x = tf.keras.layers.Lambda(lambda x: x[:, :-1, :-1, :])(x)
    return x


class ResNet50:
    def __init__(self, config):
        self.hidden1 = config["training.HiddenLayers"][0]
        self.hidden2 = config["training.HiddenLayers"][1]
        self.input_shape = (config["training.CroppedImageSize"][0], config["training.CroppedImageSize"][1],
                            config["training.ImageChannels"])
        # get the current merge_axis
        self.merge_axis = getmergeaxis()
        self.padding = "valid"
        self.initialiser = "glorot_uniform"

    def identity_block(self, tensor, kernel_size, filters, stage, block, paddings='valid'):
        """The identity block is the block that has no conv layer at shortcut.
        # Arguments
            tensor: input tensor
            kernel_size: defualt 3, the kernel size of middle conv layer at
                         main path
            filters: list of integers, the filterss of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
        # Returns
            Output tensor for the block.
        """
        filters1, filters2, filters3 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = tf.keras.layers.Conv2D(filters1, (1, 1), kernel_initializer=self.initialiser, name=conv_name_base + '2a')(tensor)
        x = tf.keras.layers.BatchNormalization(axis=self.merge_axis, name=bn_name_base + '2a')(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.ZeroPadding2D((1, 1))(x)
        x = tf.keras.layers.Conv2D(filters2, kernel_size, padding=paddings, kernel_initializer=self.initialiser,
                                   name=conv_name_base + '2b')(x)
        x = tf.keras.layers.BatchNormalization(axis=self.merge_axis, name=bn_name_base + '2b')(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Conv2D(filters3, (1, 1), kernel_initializer=self.initialiser, name=conv_name_base + '2c')(x)
        x = tf.keras.layers.BatchNormalization(axis=self.merge_axis, name=bn_name_base + '2c')(x)

        x = tf.keras.layers.Add()([x, tensor])
        x = tf.keras.layers.Activation('relu')(x)
        return x

    def conv_block(self, tensor, kernel_size, filters, stage, block, paddings='valid', strides=(2, 2)):
        """conv_block is the block that has a conv layer at shortcut
        # Arguments
            tensor: input tensor
            kernel_size: defualt 3, the kernel size of middle conv layer at
                         main path
            filters: list of integers, the filterss of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
        # Returns
            Output tensor for the block.
        Note that from stage 3, the first conv layer at main path is with
        strides=(2,2) and the shortcut should have strides=(2,2) as well
        """
        filters1, filters2, filters3 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = tf.keras.layers.Conv2D(filters1, (1, 1), strides=strides, kernel_initializer=self.initialiser,
                                   name=conv_name_base + '2a')(tensor)
        x = tf.keras.layers.BatchNormalization(axis=self.merge_axis, name=bn_name_base + '2a')(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.ZeroPadding2D((1, 1))(x)
        x = tf.keras.layers.Conv2D(filters2, kernel_size, padding=paddings, kernel_initializer=self.initialiser,
                                   name=conv_name_base + '2b')(x)
        x = tf.keras.layers.BatchNormalization(axis=self.merge_axis, name=bn_name_base + '2b')(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Conv2D(filters3, (1, 1), kernel_initializer=self.initialiser, name=conv_name_base + '2c')(x)
        x = tf.keras.layers.BatchNormalization(axis=self.merge_axis, name=bn_name_base + '2c')(x)

        shortcut = tf.keras.layers.Conv2D(filters3, (1, 1), strides=strides, kernel_initializer=self.initialiser,
                                          name=conv_name_base + '1')(tensor)
        shortcut = tf.keras.layers.BatchNormalization(axis=self.merge_axis, name=bn_name_base + '1')(shortcut)

        x = tf.keras.layers.Add()([x, shortcut])
        x = tf.keras.layers.Activation('relu')(x)
        return x

    def ResNet50Encoder(self, data):
        img_input = validate_input(data)

        resnet50 = tf.keras.layers.ZeroPadding2D((3, 3))(img_input)
        resnet50 = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2), kernel_initializer=self.initialiser, name='conv1')(resnet50)
        resnet50 = tf.keras.layers.BatchNormalization(axis=self.merge_axis, name='bn_conv1')(resnet50)
        resnet50 = tf.keras.layers.Activation('relu')(resnet50)
        resnet50 = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(resnet50)

        resnet50 = self.conv_block(resnet50, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        resnet50 = self.identity_block(resnet50, 3, [64, 64, 256], stage=2, block='b')
        resnet50 = self.identity_block(resnet50, 3, [64, 64, 256], stage=2, block='c')
        resnet50 = one_side_pad(resnet50)

        resnet50 = self.conv_block(resnet50, 3, [128, 128, 512], stage=3, block='a')
        resnet50 = self.identity_block(resnet50, 3, [128, 128, 512], stage=3, block='b')
        resnet50 = self.identity_block(resnet50, 3, [128, 128, 512], stage=3, block='c')
        resnet50 = self.identity_block(resnet50, 3, [128, 128, 512], stage=3, block='d')

        resnet50 = self.conv_block(resnet50, 3, [256, 256, 1024], stage=4, block='a')
        resnet50 = self.identity_block(resnet50, 3, [256, 256, 1024], stage=4, block='b')
        resnet50 = self.identity_block(resnet50, 3, [256, 256, 1024], stage=4, block='c')
        resnet50 = self.identity_block(resnet50, 3, [256, 256, 1024], stage=4, block='d')
        resnet50 = self.identity_block(resnet50, 3, [256, 256, 1024], stage=4, block='e')
        resnet50 = self.identity_block(resnet50, 3, [256, 256, 1024], stage=4, block='f')

        resnet50 = self.conv_block(resnet50, 3, [512, 512, 2048], stage=5, block='a')
        resnet50 = self.identity_block(resnet50, 3, [512, 512, 2048], stage=5, block='b')
        resnet50 = self.identity_block(resnet50, 3, [512, 512, 2048], stage=5, block='c')

        return resnet50

    def model(self):
        inputs = tf.keras.Input(self.input_shape)
        x = self.ResNet50Encoder(inputs)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(self.hidden1)(x)
        x = tf.keras.layers.Activation("relu")(x)
        outputs = tf.keras.layers.Dense(self.hidden2)(x)

        model = tf.keras.Model(inputs, outputs)
        return model

        return model


def Conv2DLayer(inputs, filters, kernel_initialiser, kernel_size, padding, batch_norm, name=None):
    use_bias = not batch_norm

    outputs = tf.keras.layers.Conv2D(filters=filters, kernel_initializer=kernel_initialiser,
                                     kernel_size=kernel_size, padding=padding, use_bias=use_bias, name=name)(inputs)
    if batch_norm:
        outputs = tf.keras.layers.BatchNormalization()(outputs)

    outputs = tf.keras.layers.Activation('relu')(outputs)

    return outputs


class UNet:
    def __init__(self, config):
        self.hidden1 = config["training.HiddenLayers"][0]
        self.hidden2 = config["training.HiddenLayers"][1]
        self.input_shape = (config["training.CroppedImageSize"][0], config["training.CroppedImageSize"][1],
                            config["training.ImageChannels"])

        # UNet Parameters
        self.depth = 5
        self.base_filter_power = 6
        self.filter_factor_offset = 0
        self.padding = "valid"
        self.initialiser = "glorot_uniform"
        self.k_size = 3
        self.dropout = False
        self.conv_down = [None] * self.depth
        self.modifiedarch = False

    def Encoder(self, inputs, batch_norm):
        ###
        # Contraction path
        ###

        for i in range(0, self.depth - 1):
            filters = 2 ** (self.base_filter_power + i - self.filter_factor_offset)
            if i == 0:
                self.conv_down[i] = Conv2DLayer(inputs, filters, self.initialiser, self.k_size,
                                                self.padding, batch_norm, 'down_' + str(i) + '_1')
            else:

                self.conv_down[i] = Conv2DLayer(pool, filters, self.initialiser, self.k_size, self.padding,
                                                batch_norm, 'down_' + str(i) + '_1')

            self.conv_down[i] = Conv2DLayer(self.conv_down[i], filters, self.initialiser, self.k_size, self.padding,
                                            batch_norm, 'down_' + str(i) + '_2')

            if self.dropout and i == self.depth - 2:
                self.conv_down[i] = tf.keras.layers.Dropout(0.5)(self.conv_down[i])

            pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(self.conv_down[i])

        ###
        # Bridge path
        ###

        filters = 2 ** (self.base_filter_power + (self.depth - 1) - self.filter_factor_offset)

        conv = Conv2DLayer(pool, filters, self.initialiser, self.k_size, self.padding, batch_norm, 'bridge_1')

        if not self.modifiedarch:
            conv = Conv2DLayer(conv, filters, self.initialiser, self.k_size, self.padding, batch_norm, 'bridge_2')

        if self.dropout:
            conv = tf.keras.layers.Dropout(0.5)(conv)

        return conv

    def model(self):
        inputs = tf.keras.layers.Input(self.input_shape)
        x = self.Encoder(inputs, batch_norm=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(self.hidden1)(x)
        x = tf.keras.layers.Activation("relu")(x)
        outputs = tf.keras.layers.Dense(self.hidden2)(x)

        model = tf.keras.Model(inputs, outputs)
        return model


def get_identity_block_repetition(x):
    switcher = {
        0: 2,
        1: 3,
        2: 5,
        3: 2
    }
    return switcher.get(x, lambda: 'Invalid')


def get_identity_block_name(x):
    switcher = {
        0: "a",
        1: "b",
        2: "c",
        3: "d",
        4: "e",
    }
    return switcher.get(x, lambda: 'Invalid')


def identity_block(tensor, kernel_size, filters, name, stage, block, initialiser='glorot_uniform', paddings='valid'):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    merge_axis = getmergeaxis()
    filters1, filters2, filters3 = filters
    conv_name_base = name + '_res' + str(stage) + block + '_branch'
    bn_name_base = name + '_bn' + str(stage) + block + '_branch'
    activation_name_base = name + '_relu' + str(stage) + block + '_branch'

    x = tf.keras.layers.Conv2D(filters1, (1, 1), kernel_initializer=initialiser, name=conv_name_base + '2a')(tensor)
    x = tf.keras.layers.BatchNormalization(axis=merge_axis, name=bn_name_base + '2a')(x)
    x = tf.keras.layers.Activation('relu', name=activation_name_base + '2a')(x)

    x = tf.keras.layers.ZeroPadding2D((1, 1))(x)
    x = tf.keras.layers.Conv2D(filters2, kernel_size, padding=paddings, kernel_initializer=initialiser, name=conv_name_base + '2b')(x)
    x = tf.keras.layers.BatchNormalization(axis=merge_axis, name=bn_name_base + '2b')(x)
    x = tf.keras.layers.Activation('relu', name=activation_name_base + '2b')(x)

    x = tf.keras.layers.Conv2D(filters3, (1, 1), kernel_initializer=initialiser, name=conv_name_base + '2c')(x)
    x = tf.keras.layers.BatchNormalization(axis=merge_axis, name=bn_name_base + '2c')(x)

    x = tf.keras.layers.Add()([x, tensor])
    x = tf.keras.layers.Activation('relu', name=activation_name_base + '2c')(x)

    return x


class DeepResidualUNet:
    def __init__(self, config):
        self.hidden1 = config["training.HiddenLayers"][0]
        self.hidden2 = config["training.HiddenLayers"][1]
        self.input_shape = (config["training.CroppedImageSize"][0], config["training.CroppedImageSize"][1],
                            config["training.ImageChannels"])

        # UNet Parameters
        self.depth = 5
        self.base_filter_power = 6
        self.filter_factor_offset = 0
        self.padding = "valid"
        self.initialiser = "glorot_uniform"
        self.k_size = 3
        self.dropout = False
        self.conv_down = [None] * self.depth
        self.modifiedarch = False
        self.modifiedfilters = False

    def Encoder(self, inputs, batch_norm):
        ###
        # Contraction path
        ###
        for i in range(0, self.depth - 1):
            filters = 2 ** (self.base_filter_power + i - self.filter_factor_offset)
            if i == 0:
                self.conv_down[i] = Conv2DLayer(inputs, filters, self.initialiser, self.k_size, self.padding,
                                                batch_norm, 'down_' + str(i) + '_1')
            else:
                self.conv_down[i] = Conv2DLayer(pool, filters, self.initialiser, self.k_size, self.padding,
                                                batch_norm, 'down_' + str(i) + '_1')

            if self.modifiedfilters:
                self.conv_down[i] = Conv2DLayer(self.conv_down[i], filters*4, self.initialiser, self.k_size,
                                                self.padding, batch_norm, 'down_' + str(i) + '_2')
            else:
                self.conv_down[i] = Conv2DLayer(self.conv_down[i], filters, self.initialiser, self.k_size, self.padding,
                                                batch_norm, 'down_' + str(i) + '_2')

            for ibr in range(get_identity_block_repetition(i)):
                block = get_identity_block_name(ibr)
                if self.modifiedfilters:
                    self.conv_down[i] = identity_block(self.conv_down[i], 3, [filters, filters, filters*4], f'down_{i}', i+2, block)
                else:
                    self.conv_down[i] = identity_block(self.conv_down[i], 3, [filters, filters, filters], f'down_{i}', i+2, block)

            if self.dropout and i == self.depth - 2:
                self.conv_down[i] = tf.keras.layers.Dropout(0.5)(self.conv_down[i])

            pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(self.conv_down[i])
        ###
        # Bridge
        ###
        filters = 2 ** (self.base_filter_power + (self.depth - 1) - self.filter_factor_offset)

        conv = Conv2DLayer(pool, filters, self.initialiser, self.k_size, self.padding, batch_norm, 'bridge_1')
        if not self.modifiedarch:
            conv = Conv2DLayer(conv, filters, self.initialiser, self.k_size, self.padding, batch_norm, 'bridge_2')

        if self.dropout:
            conv = tf.keras.layers.Dropout(0.5)(conv)

        return conv

    def model(self):
        inputs = tf.keras.layers.Input(self.input_shape)
        x = self.Encoder(inputs, batch_norm=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(self.hidden1)(x)
        x = tf.keras.layers.Activation("relu")(x)
        outputs = tf.keras.layers.Dense(self.hidden2)(x)

        model = tf.keras.Model(inputs, outputs)
        return model

