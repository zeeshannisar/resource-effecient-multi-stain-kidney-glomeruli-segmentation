import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import h5py
import json
import argparse
import math

import config_utils

import tensorflow as tf
import numpy as np

from tf_dataloader_API import TFDataLoader, CalculateMeanStddev
from deepmodels.subclass_API import UnetEncoder, ProjectionHead
from plot import images_plot
from optimizer.LR_decay import WarmUpAndCosineDecay
from optimizer.LARS_optimizer import LARSOptimizer


def ViewDataSamples(tmp_config):
    dataset = TFDataLoader(tmp_config, custom_augment=True, print_data_specifications=False).LoadDataset()

    for images, aug_view1, aug_view2 in dataset.take(1):
        # save images and augmented views of images to saveDir
        images_plot(config=tmp_config, images=images, augmented=False, name="Images")
        images_plot(config=tmp_config, images=aug_view1, augmented=True, name="AugmentedImagesView1")
        images_plot(config=tmp_config, images=aug_view2, augmented=True, name="AugmentedImagesView2")

    del dataset, images, aug_view1, aug_view2


def GetInputShape(tmp_config):
    image_H, image_W, image_C = tmp_config["training.CroppedImageSize"][0], tmp_config["training.CroppedImageSize"][1], \
                                tmp_config["training.ImageChannels"]
    input_shape = (image_H, image_W, image_C)
    return input_shape


def CreateDir(savePath):
    if not os.path.isdir(savePath):
        os.makedirs(savePath, exist_ok=True)
    return savePath


def LoadDatasetMeanStd(tmp_config, savePath):
    savePath = CreateDir(savePath=os.path.join(tmp_config["output.DataStatsDir"], savePath))
    stats_filename = os.path.join(savePath, "normalisation_stats.hdf5")
    if os.path.exists(stats_filename):
        tmp_config["preprocessing.ComputeMeanStd"] = False

    if tmp_config["preprocessing.ComputeMeanStd"]:
        filenames = TFDataLoader(tmp_config).GetPNGImageFiles(num_examples_mode=False)
        mean, stddev = CalculateMeanStddev(config=tmp_config, filenames=filenames)
        stats = np.array([mean, stddev])
        print(f"Writing these statistics: {stats} to {stats_filename}")
        with h5py.File(stats_filename, "w") as f:
            f.create_dataset("stats", data=stats)
    else:
        print(f"\nReading already calculated statistics from {stats_filename}...")
        with h5py.File(stats_filename, "r") as f:
            mean = f["stats"][0].tolist()
            stddev = f["stats"][1].tolist()
    return mean, stddev


def SaveConfigParserArguments(tmp_config, savePath):
    print(f'\nSaving Configuration file Passed to train the {tmp_config["training.SelfSupervisedModel"]} model')
    savePath = CreateDir(savePath=os.path.join(tmp_config["output.OutputDir"], savePath))
    with open(f'{savePath}/{tmp_config["training.SelfSupervisedModel"]}.json', 'w') as f:
        json.dump(tmp_config, f, indent=2)


def WriteModelArchitecture(model, path, name=None):
    # summary of Encoder model
    print(f"Saving {name} summary...")
    with open(path + f'/{name}_model_summary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))


def SaveModelArchitecture(online_encoder, online_projector, online_predictor, target_encoder, target_projector,
                          input_shape, savePath=None):
    savePath = CreateDir(savePath=savePath)

    encoder_output, skip0, skip1, skip2, skip3 = online_encoder(tf.ones(shape=(1, *input_shape)), training=True)
    WriteModelArchitecture(model=online_encoder.build_graph(input_shape), path=savePath, name="OnlineEncoder")

    projector_output = online_projector(encoder_output)
    WriteModelArchitecture(model=online_projector.build_graph(encoder_output.shape[1:]), path=savePath, name="OnlineProjector")

    predictor_output = online_predictor(projector_output)
    WriteModelArchitecture(model=online_predictor.build_graph(projector_output.shape[1:]), path=savePath, name="OnlinePredictor")

    encoder_output, skip0, skip1, skip2, skip3 = target_encoder(tf.ones(shape=(1, *input_shape)), training=True)
    WriteModelArchitecture(model=target_encoder.build_graph(input_shape), path=savePath, name="TargetEncoder")

    projector_output = target_projector(encoder_output)
    WriteModelArchitecture(model=target_projector.build_graph(encoder_output.shape[1:]), path=savePath, name="TargetProjector")


def SelectOptimizer(tmp_config, LR):
    """Returns the optimizer."""
    if tmp_config["training.Optimizer"].lower() == 'sgd':
        return tf.keras.optimizers.SGD(LR, tmp_config.Momentum, nesterov=True)
    elif tmp_config["training.Optimizer"].lower() == 'adam':
        return tf.keras.optimizers.Adam(LR)
    elif tmp_config["training.Optimizer"].lower() == 'lars':
        return LARSOptimizer(LR, momentum=tmp_config["training.Momentum"],
                             weight_decay=tmp_config["training.WeightDecay"],
                             exclude_from_weight_decay=['batch_normalization', 'bias', 'head_supervised'])
    else:
        raise ValueError('Unknown optimizer {}'.format(tmp_config["training.Optimizer"]))


def TryRestoreExistingCkpt(tmp_config, online_encoder, online_projector, online_predictor,
                           target_encoder, target_projector, optimizer, savePath):
    checkpoint = tf.train.Checkpoint(online_encoder=online_encoder, online_projector=online_projector,
                                     online_predictor=online_predictor, target_encoder=target_encoder,
                                     target_projector=target_projector, optimizer=optimizer)

    manager = tf.train.CheckpointManager(checkpoint, directory=os.path.join(savePath, "TF_checkpoints"),
                                         max_to_keep=tmp_config["training.MaxNumCheckpoints"],
                                         checkpoint_name='latest_ckpt', step_counter=None,
                                         checkpoint_interval=None, init_fn=None)

    checkpoint.restore(manager.latest_checkpoint).expect_partial()
    if manager.latest_checkpoint:
        start_epoch = int(manager.latest_checkpoint.rsplit("/", 1)[1].rsplit("-", 1)[1]) + 1
        print(f"Training is resumed at Epoch: {start_epoch} with checkpoint: {manager.latest_checkpoint}")
    else:
        print("Training is started from scratch...")
        start_epoch = 0
    return checkpoint, manager, start_epoch


def SaveLatestModelWeights(tmp_config, model, epoch, savePath):
    savePath = CreateDir(savePath=os.path.join(tmp_config["output.OutputDir"], savePath))
    model.save_weights(os.path.join(savePath, f"model_epoch{epoch:03d}.h5"))


def EMA(online_network, target_network, beta=0.99):
    # Update target networks (exponential moving average of online networks)
    online_network_weights = online_network.get_weights()
    target_network_weights = target_network.get_weights()

    for i in range(len(online_network_weights)):
        target_network_weights[i] = beta * target_network_weights[i] + (1 - beta) * online_network_weights[i]

    target_network.set_weights(target_network_weights)

    return target_network


class Training:
    def __init__(self, tmp_config):
        tf.keras.backend.clear_session()  # For easy reset of notebook state.

        self.strategy = tf.distribute.MirroredStrategy()

        self.config = tmp_config
        self.dataset = TFDataLoader(self.config, custom_augment=True, print_data_specifications=True).LoadDataset()
        self.input_shape = GetInputShape(self.config)
        self.beta = self.config["training.TargetBetaDecay"]
        self.train_writer = tf.summary.create_file_writer(logdir=os.path.join(self.config["output.OutputDir"],
                                                                              'tensorboard', self.config["data.Mode"]))

        with self.strategy.scope():
            self.learning_rate = WarmUpAndCosineDecay(self.config)
            self.encoder_optimizer = SelectOptimizer(self.config, self.learning_rate)
            self.projector_optimizer = SelectOptimizer(self.config, self.learning_rate)
            self.predictor_optimizer = SelectOptimizer(self.config, self.learning_rate)

            if self.config["training.DeepModel"].lower() == "unet":
                self.online_encoder = UnetEncoder()
                self.target_encoder = UnetEncoder()

            self.online_projector = ProjectionHead()
            self.online_predictor = ProjectionHead()
            self.target_projector = ProjectionHead()

            SaveModelArchitecture(self.online_encoder, self.online_projector, self.online_predictor,
                                  self.target_encoder, self.target_projector, self.input_shape,
                                  savePath=os.path.join(self.config["output.OutputDir"], 'network_summary',
                                                        self.config["training.DeepModel"]))

    def train(self):
        print('\nNumber of devices: {}'.format(self.strategy.num_replicas_in_sync))

        dataset_distributed = self.strategy.experimental_distribute_dataset(self.dataset)

        with self.strategy.scope():
            def BYOL_loss(q_predictor, g_projector):
                q_predictor = tf.math.l2_normalize(q_predictor, axis=1)
                g_projector = tf.math.l2_normalize(g_projector, axis=1)
                return 2 - 2 * tf.reduce_sum(tf.multiply(q_predictor, g_projector), axis=1)

        def train_on_batch(inputs):
            _, x1, x2 = inputs
            # Forward Pass
            f_target_1, _, _, _, _ = self.target_encoder(x1, training=True)
            g_target_1 = self.target_projector(f_target_1, training=True)

            f_target_2, _, _, _, _ = self.target_encoder(x2, training=True)
            g_target_2 = self.target_projector(f_target_2, training=True)

            with tf.GradientTape(persistent=True) as tape:
                f_online_1, _, _, _, _ = self.online_encoder(x1, training=True)
                g_online_1 = self.online_projector(f_online_1, training=True)
                q_online_1 = self.online_predictor(g_online_1, training=True)

                f_online_2, _, _, _, _ = self.online_encoder(x2, training=True)
                g_online_2 = self.online_projector(f_online_2, training=True)
                q_online_2 = self.online_predictor(g_online_2, training=True)

                byol_loss = BYOL_loss(q_predictor=q_online_1, g_projector=g_target_2)
                byol_loss += BYOL_loss(q_predictor=q_online_2, g_projector=g_target_1)

            # Backward pass (update online networks)
            encoder_grads = tape.gradient(byol_loss, self.online_encoder.trainable_variables)
            projector_grads = tape.gradient(byol_loss, self.online_projector.trainable_variables)
            predictor_grads = tape.gradient(byol_loss, self.online_predictor.trainable_variables)
            self.encoder_optimizer.apply_gradients(zip(encoder_grads, self.online_encoder.trainable_variables))
            self.projector_optimizer.apply_gradients(zip(projector_grads, self.online_projector.trainable_variables))
            self.predictor_optimizer.apply_gradients(zip(predictor_grads, self.online_predictor.trainable_variables))

            del tape
            return byol_loss

        # `run` replicates the provided computation and runs it with the distributed input.
        @tf.function
        def distributed_train_on_batch(inputs):
            per_GPU_loss = self.strategy.run(train_on_batch, args=(inputs,))
            all_GPUs_loss = tf.reduce_sum(self.strategy.reduce("SUM", per_GPU_loss, axis=None)) / self.config["training.BatchSize"]
            return all_GPUs_loss

        ckpt, manager, StartEpoch = TryRestoreExistingCkpt(self.config, self.online_encoder, self.online_projector,
                                                           self.online_predictor, self.target_encoder,
                                                           self.target_projector, self.encoder_optimizer,
                                                           os.path.join(self.config["output.OutputDir"], "models"))

        if StartEpoch >= self.config["training.NumEpochs"]:
            print(f'Model has already trained upto {self.config["training.NumEpochs"]} epochs...')
        else:
            for epoch in range(StartEpoch, self.config["training.NumEpochs"]):
                batch_wise_loss = []
                for step, x in enumerate(dataset_distributed):
                    loss = distributed_train_on_batch(x)
                    batch_wise_loss.append(loss)

                    # Beta weight updates ....
                    beta_update = 1 - (1 - self.beta) * ((tf.math.cos((tf.constant(math.pi) * (step + 1)) / (tf.cast(config["training.MaxBatchSteps"], dtype=tf.float32))) + 1) / 2)
                    # print(f"Epoch: {epoch + 1} --- batch: {step + 1}/{config['training.MaxBatchSteps']}", end="\r", flush=True)

                    self.target_encoder = EMA(self.online_encoder, self.target_encoder, beta=beta_update)
                    self.target_projector = EMA(self.online_projector, self.target_projector, beta=beta_update)

                # Calculate mean of loss at each Epoch and save it to Tensorboard.
                batch_wise_loss = round(float(np.mean(batch_wise_loss)), 5)
                with self.train_writer.as_default():
                    tf.summary.scalar("byol_loss", batch_wise_loss, step=epoch)

                print(f"\nEpoch: {epoch} - LR: {self.encoder_optimizer._decayed_lr(tf.float32).numpy()} - Loss: {batch_wise_loss}")

                print(f"saving latest checkpoint at {manager.save(checkpoint_number=epoch)}")

                if epoch % self.config["training.SaveInterval"] == 0:
                    SaveLatestModelWeights(tmp_config=self.config, model=self.online_encoder, epoch=epoch, savePath="models")
                    print(f"saving current Weights at Epoch: {epoch:03d}")

            print(f"saving final Checkpoint at {manager.save(checkpoint_number=epoch)}")
            SaveLatestModelWeights(tmp_config=self.config, model=self.online_encoder, epoch=epoch, savePath="models")


if __name__ == '__main__':
    """  """

    parser = argparse.ArgumentParser(description="Train Self Supervised Learning Methods")
    parser.add_argument('-c', '--configuration_file', type=str, default="configuration_files/byol.cfg")
    parser.add_argument('-m', '--DeepModel', type=str, default="unet", help='network architecture to use')
    parser.add_argument("-r", "--repetition", type=str, default="rep1")

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("\nGpu Growth Restriction Done...")

    args = parser.parse_args()

    if args.configuration_file:
        config = config_utils.readconfig(args.configuration_file)
    else:
        config = config_utils.readconfig()

    print('Command Line Input Arguments:\n', json.dumps(vars(args), indent=4, separators=(',', ':')))
    print("\n")

    if args.DeepModel.lower() == "unet" or args.DeepModel.lower() == "resnet50" or args.DeepModel.lower() == "deepresidualunet":
        config["training.DeepModel"] = args.DeepModel
    else:
        raise ValueError("Please specify one of (\'unet, resnet50, deepresidualunet\') in args.DeepModel.")

    config["data.DataDir"] = os.path.join(config["data.DataDir"], config["data.Mode"])
    config["output.DataStatsDir"] = os.path.join(config["output.OutputDir"], config["training.SelfSupervisedModel"])
    config["output.OutputDir"] = os.path.join(config["output.OutputDir"], config["training.SelfSupervisedModel"],
                                              config["training.DeepModel"], args.repetition)

    # Derived Parameters
    config["training.NumExamples"] = TFDataLoader(config).GetPNGImageFiles(num_examples_mode=True)
    config["training.MaxBatchSteps"] = config["training.NumExamples"] // config["training.BatchSize"]
    config["training.MaxTrainSteps"] = config["training.NumEpochs"] * config["training.MaxBatchSteps"]
    config["training.WarmupSteps"] = config["training.WarmupEpochs"] * config["training.MaxBatchSteps"]
    config["data.mean"], config["data.stddev"] = LoadDatasetMeanStd(tmp_config=config, savePath="data_statistics")

    print("Configuration File Arguments:\n", json.dumps(config, indent=2, separators=(",", ":")))
    SaveConfigParserArguments(tmp_config=config, savePath="config_file")
    ViewDataSamples(tmp_config=config)
    SSL = Training(tmp_config=config)
    SSL.train()
