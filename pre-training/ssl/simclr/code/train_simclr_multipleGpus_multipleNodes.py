import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import h5py
import json
import argparse

import config_utils

import tensorflow as tf
import numpy as np

from tf_dataloader_API import TFDataLoader, Compute_Mean_Std_Dataset
from deepmodels.functional_API import ResNet50, UNet
from tensorboard_API import TensorboardLogs
from tensorflow_similarity.losses.simclr import SimCLRLoss
from optimizer.LR_decay import WarmUpAndCosineDecay
from optimizer.LARS_optimizer import LARSOptimizer

def CreateDir(savePath):
    if not os.path.isdir(savePath):
        os.makedirs(savePath, exist_ok=True)
    return savePath


def LoadDatasetMeanStd(tmp_config, savePath):
    savePath = CreateDir(savePath=os.path.join(tmp_config["output.DataStatsDir"], savePath))
    stats_filename = os.path.join(savePath, "normalisation_stats.hdf5")
    if os.path.exists(stats_filename):
        tmp_config["augmentations.ComputeMeanStd"] = False

    if tmp_config["augmentations.ComputeMeanStd"]:
        filenames = TFDataLoader(tmp_config).GetPNGImageFiles(num_examples_mode=False)
        mean, stddev = Compute_Mean_Std_Dataset(config=tmp_config, filenames=filenames)
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


def SaveModelArchitecture(model, savePath):
    """
    Function to save model architecture and parameters

    :param model: model to be saved
    :param path: folder to save model
    :return:
    """
    savePath = CreateDir(savePath=savePath)

    # summary of model
    with open(savePath + '/model_summary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    # saving architecture, model can be recreated from this file
    json_str = model.to_json()
    with open(savePath + '/model_json.json', "w") as f:
        json.dump(json.loads(json_str), f, indent=4)


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


def TryRestoreExistingCkpt(tmp_config, model, optimizer, savePath):
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
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


class Training:
    def __init__(self, tmp_config):
        tf.keras.backend.clear_session()  # For easy reset of notebook state.
        slurm_resolver = tf.distribute.cluster_resolver.SlurmClusterResolver(port_base=15000)
        communication = tf.distribute.experimental.CommunicationOptions(implementation=tf.distribute.experimental.CommunicationImplementation.AUTO)
        self.strategy = tf.distribute.MultiWorkerMirroredStrategy(communication_options=communication,
                                                                  cluster_resolver=slurm_resolver)
        self.config = tmp_config

    def train(self):
        strategy = self.strategy
        print(print('\nNumber of devices: {}'.format(strategy.num_replicas_in_sync)))

        dataset = TFDataLoader(self.config, custom_augment=True, print_data_specifications=True).LoadDataset()
        dataset_distributed = strategy.experimental_distribute_dataset(dataset)

        GLOBAL_BATCH_SIZE = self.config["training.BatchSize"]

        T = self.config["training.Temperature"]
        save_model_architecture_path = os.path.join(self.config["output.OutputDir"], 'network_summary',
                                                    self.config["training.DeepModel"])
        with strategy.scope():
            def compute_loss(h1, h2):
                return SimCLRLoss(temperature=T, reduction=tf.keras.losses.Reduction.NONE)(h1, h2)

        with strategy.scope():
            learning_rate = WarmUpAndCosineDecay(self.config, self.config["training.LR"],
                                                 self.config["training.NumExamples"])
            optimizer = SelectOptimizer(self.config, learning_rate)

            if self.config["training.DeepModel"].lower() == "resnet50":
                model = ResNet50(config=self.config).model()
            elif self.config["training.DeepModel"].lower() == "unet":
                model = UNet(config=self.config).model()

        SaveModelArchitecture(model, savePath=save_model_architecture_path)

        tensorboard = TensorboardLogs(config=self.config)

        def train_on_batch(inputs):
            _, x_i, x_j = inputs
            with tf.GradientTape() as tape:
                z_i = model(x_i, training=True)
                z_j = model(x_j, training=True)

                simclr_loss = compute_loss(h1=z_i, h2=z_j)

            gradients = tape.gradient(simclr_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return simclr_loss

        # `run` replicates the provided computation and runs it with the distributed input.
        @tf.function
        def distributed_train_on_batch(inputs):
            per_GPU_loss = strategy.run(train_on_batch, args=(inputs,))
            return tf.reduce_sum(strategy.reduce("SUM", per_GPU_loss, axis=None)) / GLOBAL_BATCH_SIZE

        ckpt, manager, StartEpoch = TryRestoreExistingCkpt(tmp_config=self.config, model=model, optimizer=optimizer,
                                                           savePath=os.path.join(self.config["output.OutputDir"],
                                                                                 "models"))

        if StartEpoch >= self.config["training.NumEpochs"]:
            print(f'Model has already trained upto {self.config["training.NumEpochs"]} epochs...')
        else:
            for epoch in range(StartEpoch, self.config["training.NumEpochs"]):
                batch_wise_loss = []
                for x in dataset_distributed:
                    loss = distributed_train_on_batch(x)
                    batch_wise_loss.append(loss)

                # Calculate mean of loss at each Epoch and save it to Tensorboard.
                with tensorboard.train_writer.as_default():
                    tf.summary.scalar("SimCLR Loss", np.mean(batch_wise_loss), step=epoch)
                    # tf.summary.scalar('Learning Rate', optimizer._decayed_lr(tf.float32), step=epoch)

                print(f"\nEpoch: {epoch} --- LR: {optimizer._decayed_lr(tf.float32)} ---Loss: {np.mean(batch_wise_loss):.4f}")

                print(f"saving latest checkpoint at {manager.save(checkpoint_number=epoch)}")

                # # Saving Images (x), and their Augmented Views (x_i, x_j) at a specific save_interval.
                if epoch % self.config["training.SaveInterval"] == 0:
                    tensorboard.save_to_tensorboard(dataset=dataset, epoch=epoch)

                if epoch % self.config["training.SaveInterval"] == 0:
                    SaveLatestModelWeights(tmp_config=self.config, model=model, epoch=epoch, savePath="models")
                    print(f"saving current Weights at Epoch: {epoch:03d}")

            print(f"saving final Checkpoint at {manager.save(checkpoint_number=epoch)}")
            SaveLatestModelWeights(tmp_config=self.config, model=model, epoch=epoch, savePath="models")


if __name__ == '__main__':
    """  """

    parser = argparse.ArgumentParser(description="Train Self Supervised Learning Methods")
    parser.add_argument('-c', '--configuration_file', type=str, help='the configuration file to use')
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

    if args.DeepModel.lower() == "unet" or args.DeepModel.lower() == "resnet50":
        config["training.DeepModel"] = args.DeepModel
    else:
        raise ValueError("Please specify one of (\'UNet, ResNet50\') in parser.parse_args().DeepModel.")

    config["data.DataDir"] = os.path.join(config["data.DataDir"], config["data.Mode"])
    config["output.OutputDir"] = os.path.join(config["output.OutputDir"], config["data.Dataset"],
                                              config["training.SelfSupervisedModel"], config["training.DeepModel"],
                                              args.repetition, config["output.Label"])
    config["output.DataStatsDir"] = os.path.join(config["output.DataStatsDir"])

    # Derived Parameters

    if config["data.Mode"].lower() == "train":
        config["training.Evaluate"] = False
    else:
        config["training.Evaluate"] = True

    config["training.NumExamples"] = TFDataLoader(config).GetPNGImageFiles(num_examples_mode=True)
    config["data.mean"], config["data.stddev"] = LoadDatasetMeanStd(tmp_config=config, savePath="data_statistics")

    print("Configuration File Arguments:\n", json.dumps(config, indent=2, separators=(",", ":")))

    SaveConfigParserArguments(tmp_config=config, savePath="config_file")
    SSL = Training(tmp_config=config)
    SSL.train()
