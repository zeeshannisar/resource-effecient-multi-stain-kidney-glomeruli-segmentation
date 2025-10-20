from __future__ import print_function, division

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Lambda
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop, Adam

import keras.backend as K

import matplotlib.pyplot as plt

import sys

import numpy as np
import random
from unet import unet_models
from utils import image_utils
import argparse
import os
import ntpath

class WGAN():
    def __init__(self):
        self.img_rows = 512
        self.img_cols = 512
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 1000
#        self.n_critic = 5
        self.clip_value = 0.01
#        optimizer = RMSprop(lr=0.00005)
#        optimizer = RMSprop(lr=0.0005)
        optimizer = Adam(lr=0.0002, beta_1=0.5)

        # Build and compile the critic
        self.critic = self.build_critic()
        self.critic.compile(loss=self.wasserstein_loss,
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generated imgs
        z = Input(shape=self.img_shape)

        img = self.generator(z)

        # For the combined model we will only train the generator
        self.critic.trainable = False

        # The critic takes generated images as input and determines validity
#        print(img[:,:,:,1][...,None].shape)
        valid = self.critic(img)

        # The combined model  (stacked generator and critic)
        self.combined = Model(z, valid)
        self.combined.compile(loss=self.wasserstein_loss,
            optimizer=optimizer,
            metrics=['accuracy'])

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)


    def build_generator(self):
        model,_,_ = unet_models.build_AE_UNet(self.img_shape, 1, depth=2, filter_factor_offset=0, batchnormalisation=None, learnupscale=True, padding='same')
#        model,_,_ = unet_models.build_AE_softmax_UNet(self.img_shape, 1, depth=3, filter_factor_offset=0, batchnormalisation=None, learnupscale=True, padding='same')
#        model,_,_ = unet_models.build_UNet(self.img_shape, 2, depth=3, filter_factor_offset=0, batchnormalisation=None, learnupscale=True, padding='same')

#        def select_1_class(x):
#            return x[:,:,:,1][...,None]

#        def select_1_class_outputshape(input_shape):
#            shape = list(input_shape)
#            assert len(shape) == 4  # only valid for 2D tensors
#            shape[-1] = 1
#            return tuple(shape)

#        new_model = Sequential()
#        new_model.add(model)
#        new_model.add(Lambda(select_1_class, output_shape=select_1_class_outputshape))
#        new_model.summary()

#        data = Input(shape=(1,512,512,1))
#        model = Model(input=data, output=new_model.output)

        model.summary()
        return model

    def build_generator2(self):

        model = Sequential()

        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((7, 7, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=4, padding="same"))
        model.add(Activation("sigmoid"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_critic(self):

        model = Sequential()

        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def load_images(self, dir, num=None):
        import glob
        image_list = []
        filelist=glob.glob(dir+'/*.png')
        if num:
            filelist=filelist[0:num]
        for filename in filelist:
            im=image_utils.read_image(filename)
            im[im != 2] = 0
            im = im.astype(bool)
            image_list.append(im)
        return np.array(image_list)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
#        (X_train, _), (_, _) = mnist.load_data()
        print('loading glomeruli')
        X_train = self.load_images('/home/lampert/data/Nephrectomies/25_106_107_108_p4/downsampledpatches/train/gts/glomeruli')
        print('loaded')
        print('loading negative')
        X_train = np.concatenate((X_train, self.load_images('/home/lampert/data/Nephrectomies/25_106_107_108_p4/downsampledpatches/train/gts/negative', X_train.shape[0])), axis=0)
#        X_train = np.concatenate((X_train, self.load_images('/home/lampert/data/Nephrectomies/25_106_107_108_p4/downsampledpatches/train/gts/negative', 100)), axis=0)
        print('loaded')

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 0.5) / 0.5
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))

        import glob
        image_list = []
        dir = '/home/lampert/detection_patches'
        out_dir = '/home/lampert/gan_training'
        filenames = glob.glob(dir+'/*.png')
        ind=0
        ind_max=len(filenames)

#        sample_noise = []
#        for i in range(0,25):
#            sample_noise.append(image_utils.read_binary_image(filenames[i])).astype(np.float32)
#        sample_noise[sample_noise > 0] = 1
#        sample_noise = (sample_noise - 0.5) / 0.5
#        sample_noise = np.array(sample_noise)[:,:,:,np.newaxis]

        with open("/home/lampert/gan_loss.txt","w", buffering=1) as f:

            for epoch in range(epochs):
                random.shuffle(filenames)

                for p in range(self.n_critic):

                    # ---------------------
                    #  Train Discriminator
                    # ---------------------

                    # Select a random batch of images
                    idx = np.random.randint(0, X_train.shape[0], batch_size)
                    imgs = X_train[idx,...]

                    # Sample noise as generator input
#                    noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
#                    noise = filenames[ind:ind+batch_size-1,:,:]
                    noise = []
                    curr_filenames = []
                    for i in range(ind,ind+batch_size):
                        noise.append((image_utils.read_binary_image(filenames[i]).astype(np.float32) - 0.5) / 0.5)
                        curr_filenames.append(filenames[i])

                    noise = np.array(noise)[:,:,:,np.newaxis]

                    ind+=batch_size
                    if ind+batch_size >= ind_max:
                        ind = 0

                    # Generate a batch of new images
#                    print('data: ', np.amin(noise), np.amax(noise))
                    gen_imgs = self.generator.predict(noise)

#                    for i in range(batch_size):
#                        print(gen_imgs[i,:,:,:].shape)
#                        gen_imgs[i,:,:,:] = gen_imgs[i,:,:,1]
#                        print(gen_imgs[i,:,:,:].shape)

                    if epoch % sample_interval == 0:
                        if not os.path.exists(os.path.join(out_dir, str(epoch))):
                            os.makedirs(os.path.join(out_dir, str(epoch)))
                        for i, filename in enumerate(curr_filenames):
                            image_utils.save_image(((gen_imgs[i,:,:,:]+1)/2 * 255).astype(np.uint8), os.path.join(out_dir, str(epoch), ntpath.basename(filename)))

#                    if p == 0:
#                        print('output: ', np.amin(gen_imgs), np.amax(gen_imgs))

#                    gen_imgs = (gen_imgs - 0.5) / 0.5
#                    print('output standardised: ', np.amin(gen_imgs), np.amax(gen_imgs))

                    # Train the critic
                    d_loss_real = self.critic.train_on_batch(imgs, valid)
                    d_loss_fake = self.critic.train_on_batch(gen_imgs, fake)
                    d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                    # Clip critic weights
                    for l in self.critic.layers:
                        weights = l.get_weights()
                        weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                        l.set_weights(weights)

                # ---------------------
                #  Train Generator
                # ---------------------

                g_loss = self.combined.train_on_batch(noise, valid)

                # Plot the progress
                print ("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0]))
                f.write("%d [D loss: %f] [G loss: %f]\n" % (epoch, 1 - d_loss[0], 1 - g_loss[0]))

                # If at save interval => save generated image samples
#                if epoch % sample_interval == 0:
#                    self.sample_images(epoch, sample_noise)


    def sample_images(self, epoch, filenames):
        r, c = 5, 5
#        noise = np.random.normal(0, 1, (r * c, self.latent_dim))

        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("/home/lampert/gan_training/output_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract patches.')
    parser.add_argument('-g', '--gpu', type=str, help='specify which GPU to use')
    args = parser.parse_args()

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    wgan = WGAN()
    wgan.train(epochs=4000, batch_size=16, sample_interval=1)
