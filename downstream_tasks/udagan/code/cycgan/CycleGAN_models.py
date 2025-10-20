import tensorflow_addons as tfa
import tensorflow.keras as keras

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, MaxPooling2D, Cropping2D, BatchNormalization, Activation, ZeroPadding2D, Add
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from cycgan import Custom_Layers

# Discriminator takes half of the loss (practical advice from the paper)
normalization_axis = 3

def INSTANCE_NORM(name=None):
    return tfa.layers.InstanceNormalization(axis=normalization_axis, scale=True, center=True, name=name)

def discriminator_loss(y_true, y_pred):
    return 0.5 * keras.losses.mse(y_true, y_pred)

class CycleGAN:
    def __init__(self, stain):
        self.img_rows = 508
        self.img_cols = 508
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        self.gf = 32
        self.df = 64

        # Loss weights
        self.lambda_adv = 1.0
        self.lambda_cycle = 10.0                    # Cycle-consistency loss
        self.lambda_id = 5.0     # Identity loss

        self.optimizer = Adam(0.0002)
        self.initializer = keras.initializers.random_normal(mean=0,stddev=0.02)

        print(f'*************** Loading CycleGAN Model for stain: {stain}  ***************')

        # Build and compile the discriminators
        norm = 'instance'
        self.d_A = self.build_discriminator(norm=norm)
        self.d_B = self.build_discriminator(norm=norm)
        self.d_A.compile(loss=discriminator_loss, optimizer=self.optimizer, metrics=['accuracy'])
        self.d_B.compile(loss=discriminator_loss, optimizer=self.optimizer, metrics=['accuracy'])

        # Calculate output shape of D (PatchGAN)
        # Take the output from the discriminator and its shape for the dimensions of the discriminator patch
        patch = self.d_A.output_shape[1]  # round(self.img_rows / 2 ** 4)
        self.disc_patch = (patch, patch, 1)
        # print(self.disc_patch)

        #-------------------------
        # Construct Computational
        #   Graph of Generators
        #-------------------------

        # Build the generators
        self.g_AB = self.build_generator(norm=norm)
        self.g_BA = self.build_generator(norm=norm)

        # Input images from both domains
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Translate images to the other domain
        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)
        # Translate images back to original domain
        cyc_A = self.g_BA(fake_B)
        cyc_B = self.g_AB(fake_A)
        # Identity mapping of images
        id_A = self.g_BA(img_A)
        id_B = self.g_AB(img_B)

        # For the combined model we will only train the generators
        self.d_A.trainable = False
        self.d_B.trainable = False

        # Discriminators determines validity of translated images
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[img_A, img_B],
                              outputs=[valid_A, valid_B, cyc_A, cyc_B, id_A, id_B])
        self.combined.compile(loss=['mse', 'mse', 'mae', 'mae', 'mae', 'mae'],
                            loss_weights=[self.lambda_adv, self.lambda_adv, self.lambda_cycle, self.lambda_cycle,
                                          self.lambda_id, self.lambda_id ],
                            optimizer=self.optimizer)

    # # as per implementation on github in torch, they first do normalization and only then activation
    def build_generator(self,norm='instance'):

        def c7s1_k(layer_input, k, norm='instance', padding=(3,3)):
            g = Custom_Layers.ReflectionPadding2D(padding=padding)(layer_input)
            g = Conv2D(kernel_size=7,strides=1,filters=k,kernel_initializer=self.initializer)(g)
            if norm=='batch':
                g = BatchNormalization(momentum=0.9)(g)
            else:
                g = INSTANCE_NORM()(g)
            g = Activation('relu')(g)

            return g
        def dk(layer_imput, k, norm='instance'):
            g = Conv2D(kernel_size=3,strides=2,filters=k,kernel_initializer=self.initializer,padding='same')(layer_imput)
            if norm == 'batch':
                g = BatchNormalization(momentum=0.9)(g)
            else:
                g = INSTANCE_NORM()(g)
            g = Activation('relu')(g)

            return g
        def rk(layer_input, k, norm='instance'):
            g = Custom_Layers.ReflectionPadding2D(padding=(1, 1))(layer_input)
            g = Conv2D(kernel_size=3, strides=1, filters=k,kernel_initializer=self.initializer)(g)
            if norm == 'batch':
                g = BatchNormalization(momentum=0.9)(g)
            else:
                g = INSTANCE_NORM()(g)
            g = Activation('relu')(g)

            g = Custom_Layers.ReflectionPadding2D(padding=(1, 1))(g)
            g = Conv2D(kernel_size=3, strides=1, filters=k)(g)
            if norm == 'batch':
                g = BatchNormalization(momentum=0.9)(g)
            else:
                g = INSTANCE_NORM()(g)

            g = Add()([g,layer_input])
            # activation is applied after connecting to the input
            g = Activation('relu')(g)

            return g

        def uk(layer_input, k, norm='instance'):
            g = Conv2DTranspose(kernel_size=3,filters=k,strides=2,padding='same',kernel_initializer=self.initializer)(layer_input)
            if norm == 'batch':
                g = BatchNormalization(momentum=0.9)(g)
            else:
                g = INSTANCE_NORM()(g)
            g = Activation('relu')(g)
            return g

        g0 = Input(shape=self.img_shape)

        g = c7s1_k(g0,32,norm)
        g = dk(g,64,norm)
        g = dk(g,128,norm)
        for _ in range(9):
            g = rk(g,128,norm)
        g = uk(g,64,norm)
        g = uk(g,32,norm)

        # The last convolutional layer
        g = Custom_Layers.ReflectionPadding2D(padding=(3, 3))(g)
        g = Conv2D(kernel_size=7, strides=1, filters=3, kernel_initializer=self.initializer)(g)
        g = Activation('tanh')(g)

        model = Model(g0,g)

        return model

    def build_discriminator(self, norm='instance'):
        def ck(layer_input, k, norm='instance'):
            g = ZeroPadding2D(padding=(1,1))(layer_input)
            g = Conv2D(kernel_size=4, strides=2, filters=k, kernel_initializer=self.initializer)(g)
            if norm == 'batch':
                g = BatchNormalization(momentum=0.9)(g)
            else:
                if norm == 'instance':
                    g = INSTANCE_NORM()(g)

            g = LeakyReLU(0.2)(g)
            return g

        img = Input(shape=self.img_shape)
        d = ck(img,64,norm=None)
        d = ck(d,128,norm)
        d = ck(d,256,norm)

        # To get a 70x70 patch at the output
        d = ZeroPadding2D(padding=(1, 1))(d)
        d = Conv2D(kernel_size=4, filters=512, strides=1, kernel_initializer=self.initializer)(d)
        if norm == 'batch':
            d = BatchNormalization(momentum=0.9)(d)
        else:
            if norm == 'instance':
                d = INSTANCE_NORM()(d)

        d = LeakyReLU(0.2)(d)
        d = ZeroPadding2D(padding=(1, 1))(d)
        d = Conv2D(kernel_size=4,filters=1,strides=1,kernel_initializer=self.initializer)(d)


        model = Model(img, d)

        return model
