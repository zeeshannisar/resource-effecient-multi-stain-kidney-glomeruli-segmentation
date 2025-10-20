import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import h5py
import json
import argparse

import config_utils

import tensorflow as tf

import numpy as np
import pandas as pd

from plot import stain_separation_plot, images_with_labels_plot
from deepmodels.subclass_API import UnetEncoder, UnetDecoder, HO_UnetEncoder, HO_UnetDecoder
from tf_dataloader_API import TFDataLoader, Compute_Mean_Std_Dataset
from tensorboard_API import TensorboardLogs
from custom_callbacks.early_stopping import EarlyStoppingAtMinLoss
from custom_callbacks.reduce_learning_rate import ReduceLROnPlateauAtMinLoss


def normalise(image, mean, stddev):
    image -= mean
    image /= stddev

    return image


def ExtractCentralRegion(images, shape=(324, 324)):
    x = shape[0]
    y = shape[1]

    images_x_offset = (images.shape[0] - x) // 2
    images_y_offset = (images.shape[1] - y) // 2
    return images[images_x_offset:images.shape[0] - images_x_offset, images_y_offset:images.shape[1] - images_y_offset]


def ViewStainSeparationSamples(tmp_config):
    print(f'\nSaving Stain Separation Sample.')
    ds_train = TFDataLoader(tmp_config, ssl_model_phase=tmp_config["training.SSLModelPhase"],
                            mode="train", print_data_specifications=False).LoadDataset()
    for step, (image_x, image_H, label_O, image_O, label_H) in enumerate(ds_train.take(1)):
        stain_separation_plot(image_x, image_H, image_O, tmp_config['output.OutputDir'], tmp_config['data.Stain'])
    del ds_train


def ViewDataSamples(tmp_config):
    print(f'\nSaving Data Samples with Labels extracted from tf.Data API.')
    ds_train = TFDataLoader(tmp_config, ssl_model_phase=tmp_config["training.SSLModelPhase"],
                            mode="train", print_data_specifications=False).LoadDataset()
    for step, (image_x, image_H, label_O, image_O, label_H) in enumerate(ds_train.take(1)):
        images_with_labels_plot(image_H, label_O, image_O, label_H, tmp_config['output.OutputDir'],
                                tmp_config['data.Stain'])
    del ds_train


def GetInputShape(tmp_config):
    ds_train = TFDataLoader(tmp_config, ssl_model_phase=tmp_config["training.SSLModelPhase"],
                            mode="train", print_data_specifications=False).LoadDataset()
    _, image_H, _, _, _ = next(iter(ds_train))
    del ds_train
    input_shape = image_H.shape[1:]
    return input_shape


def GetOutputShape(combined_encoder, combined_decoder, input_shape, batch_size=16):
    h_input_tmp = tf.random.normal(shape=(batch_size, *input_shape))
    o_input_tmp = tf.random.normal(shape=(batch_size, *input_shape))

    h_encoder_output, o_encoder_output, h_skip0, h_skip1, h_skip2, h_skip3, \
    o_skip0, o_skip1, o_skip2, o_skip3 = combined_encoder(h_input_tmp, o_input_tmp)

    h2o_output, o2h_output = combined_decoder(h_encoder_output, o_encoder_output,
                                              h_skip0, h_skip1, h_skip2, h_skip3,
                                              o_skip0, o_skip1, o_skip2, o_skip3)

    output_shape = h2o_output.shape[1:]
    return output_shape


def CreateDir(savePath):
    if not os.path.isdir(savePath):
        os.makedirs(savePath, exist_ok=True)
    return savePath


def SaveConfigParserArguments(tmp_config, savePath):
    print(f'\nSaving Configuration file Passed to train the '
          f'{tmp_config["training.SSLModel"]}-{tmp_config["training.SSLModelPhase"]} model')
    savePath = CreateDir(savePath=os.path.join(tmp_config["output.OutputDir"], savePath))
    with open(f'{savePath}/{tmp_config["config.filename"].split("/")[-1]}', 'w') as f:
        json.dump(tmp_config, f, indent=2)


def SelectOptimizer(tmp_config):
    """Returns the optimizer."""
    if tmp_config["training.Optimiser"].lower() == 'sgd':
        return tf.keras.optimizers.SGD(tmp_config["training.LearningRate"], momentum=0.9, nesterov=True)
    elif tmp_config["training.Optimiser"].lower() == 'adam':
        return tf.keras.optimizers.Adam(tmp_config["training.LearningRate"])
    else:
        raise ValueError('Unknown optimizer {}'.format(tmp_config["training.Optimizer"]))


def TryRestoreExistingCkpt(tmp_config, combined_encoder, combined_decoder, optimizer, savePath):
    checkpoint = tf.train.Checkpoint(combined_encoder=combined_encoder, combined_decoder=combined_decoder,
                                     optimizer=optimizer)
    manager = tf.train.CheckpointManager(checkpoint, directory=os.path.join(savePath, "TF_checkpoints"),
                                         max_to_keep=tmp_config["training.MaxNumCheckpoints"],
                                         checkpoint_name='latest_ckpt', step_counter=None,
                                         checkpoint_interval=None, init_fn=None)
    # expect_partial(): Silence warnings about incomplete checkpoint restores. Warnings are otherwise printed for unused
    # parts of the checkpoint file or object when the Checkpoint object is deleted (often at program shutdown).
    checkpoint.restore(manager.latest_checkpoint).expect_partial()
    if manager.latest_checkpoint:
        start_epoch = int(manager.latest_checkpoint.rsplit("/", 1)[1].rsplit("-", 1)[1])
        print(f"\nTraining is resumed at epoch: {start_epoch} with checkpoint: {manager.latest_checkpoint}...!")
    else:
        print("\nTraining is started from scratch...!")
        start_epoch = 0
    return checkpoint, manager, start_epoch


def WriteModelArchitecture(model, path, name=None):
    # summary of Encoder model
    print(f"Saving {name} summary...")
    with open(path + f'/{name}_model_summary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))


def SaveModelArchitecture(encoder, decoder, combined_encoder, combined_decoder, input_shape, savePath):
    savePath = CreateDir(savePath=savePath)

    encoder_output, skip0, skip1, skip2, skip3 = encoder(tf.ones(shape=(1, *input_shape)), training=True)
    WriteModelArchitecture(model=encoder.build_graph(input_shape), path=savePath, name="UnetEncoder")

    decoder_output = decoder(encoder_output, skip0, skip1, skip2, skip3, training=True)
    WriteModelArchitecture(model=decoder.build_graph(encoder_output.shape[1:], skip0.shape[1:], skip1.shape[1:],
                                                     skip2.shape[1:], skip3.shape[1:]),
                           path=savePath, name="UnetDecoder")

    h_encoder_output, o_encoder_output, h_skip0, h_skip1, h_skip2, h_skip3, \
    o_skip0, o_skip1, o_skip2, o_skip3 = combined_encoder(tf.ones(shape=(1, *input_shape)),
                                                          tf.ones(shape=(0, *input_shape)), training=True)
    WriteModelArchitecture(model=combined_encoder.build_graph(h_input_shape=input_shape, o_input_shape=input_shape),
                           path=savePath, name="HO_UnetEncoder")

    h2o_output, o2h_output = combined_decoder(h_encoder_output, o_encoder_output,
                                              h_skip0, h_skip1, h_skip2, h_skip3,
                                              o_skip0, o_skip1, o_skip2, o_skip3, training=True)
    WriteModelArchitecture(model=combined_decoder.build_graph(h_encoder_output.shape[1:], o_encoder_output.shape[1:],
                                                              h_skip0.shape[1:], h_skip1.shape[1:], h_skip2.shape[1:],
                                                              h_skip3.shape[1:],
                                                              o_skip0.shape[1:], o_skip1.shape[1:], o_skip2.shape[1:],
                                                              o_skip3.shape[1:]),
                           path=savePath, name="HO_UnetDecoder")


def LoadDatasetMeanStd(tmp_config, savePath):
    savePath = CreateDir(savePath=os.path.join(tmp_config["output.DataStatsDir"], savePath))
    stats_filename = os.path.join(savePath, "normalisation_stats.hdf5")
    if os.path.exists(stats_filename):
        tmp_config["preprocessing.ComputeMeanStd"] = False

    if tmp_config["preprocessing.ComputeMeanStd"]:
        filenames = TFDataLoader(tmp_config, ssl_model_phase=tmp_config["training.SSLModelPhase"]).GetPNGImageFiles(
            num_examples_mode=False)
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


def GetDecoderLossType(tmp_config):
    if tmp_config["training.DecoderLossType"] == "L1":
        return "mae_loss"
    elif tmp_config["training.DecoderLossType"] == "L2":
        return "mse_loss"
    else:
        raise ValueError(f"Please specify one of (\'L1, L2\') LossType in {tmp_config['config.filename']} file.")


def loss_function(predictions, labels, loss_type="mse_loss"):
    if loss_type == "mae_loss":
        return tf.reduce_mean(tf.abs(tf.subtract(predictions, labels)))
    if loss_type == "mse_loss":
        return tf.reduce_mean(tf.square(tf.subtract(predictions, labels)))


def SaveModelWeights(tmp_config, combined_encoder, combined_decoder, savePath, saveName="latest"):
    savePath = CreateDir(savePath=os.path.join(tmp_config["output.OutputDir"], savePath))

    combined_encoder.save_weights(os.path.join(savePath, f"HO_encoder_model.{saveName}.hdf5"))
    combined_decoder.save_weights(os.path.join(savePath, f"HO_decoder_model.{saveName}.hdf5"))


class Training:
    def __init__(self, tmp_config):
        self.config = tmp_config

        if self.config["training.DeepModel"].lower() == "unet":
            self.encoder, self.decoder = UnetEncoder(), UnetDecoder()
            self.combined_encoder, self.combined_decoder = HO_UnetEncoder(), HO_UnetDecoder()

        self.input_shape = GetInputShape(self.config)
        print(f"\ninput_shape: {self.input_shape}")

        self.output_shape = GetOutputShape(self.combined_encoder, self.combined_decoder,
                                           self.input_shape, self.config["training.BatchSize"])
        print(f"\noutput_shape: {self.output_shape}")

        self.optimizer = SelectOptimizer(self.config)

        if self.config["training.ReduceLearningRate"]:
            self.reduce_LR = ReduceLROnPlateauAtMinLoss(optimizer_LR=self.optimizer.learning_rate,
                                                        patience=self.config["training.Patience"], factor=0.1)
        if self.config["training.EarlyStopping"]:
            self.early_stopping = EarlyStoppingAtMinLoss(patience=self.config["training.Patience"] * 2)

        self.tensorboard = TensorboardLogs(config=self.config)
        self.loss_type = GetDecoderLossType(tmp_config=self.config)

        self.train_record_file = os.path.join(self.config["output.OutputDir"], "training_record.txt")
        self.logging_loss_filePath = os.path.join(self.config["output.OutputDir"], "training_loss.csv")

        self.ds_train = TFDataLoader(self.config, ssl_model_phase=self.config["training.SSLModelPhase"],
                                     mode="train", print_data_specifications=True).LoadDataset()

        self.ds_valid = TFDataLoader(self.config, ssl_model_phase=self.config["training.SSLModelPhase"],
                                     mode="validation", print_data_specifications=True).LoadDataset()

    def train(self):
        print("\n")
        SaveModelArchitecture(self.encoder, self.decoder, self.combined_encoder, self.combined_decoder,
                              self.input_shape, savePath=f"{self.config['output.OutputDir']}/network_summary")

        @tf.function
        def train_step(images_H, labels_O, images_O, labels_H):
            # persistent is set to True because the tape is used more than once to calculate the gradients.
            with tf.GradientTape(persistent=True) as tape:
                h_encoder_output, o_encoder_output, h_skip0, h_skip1, h_skip2, h_skip3, \
                o_skip0, o_skip1, o_skip2, o_skip3 = self.combined_encoder(images_H, images_O, training=True)

                h2o_decoder_output, o2h_decoder_output = self.combined_decoder(h_encoder_output, o_encoder_output,
                                                                               h_skip0, h_skip1, h_skip2, h_skip3,
                                                                               o_skip0, o_skip1, o_skip2, o_skip3,
                                                                               training=True)

                h2o_train_loss = loss_function(predictions=h2o_decoder_output, labels=labels_O,
                                               loss_type=self.loss_type)
                o2h_train_loss = loss_function(predictions=o2h_decoder_output, labels=labels_H,
                                               loss_type=self.loss_type)
                train_loss = h2o_train_loss + o2h_train_loss

            combined_encoder_gradients = tape.gradient(train_loss, self.combined_encoder.trainable_variables)
            self.optimizer.apply_gradients(zip(combined_encoder_gradients, self.combined_encoder.trainable_variables))

            combined_decoder_gradients = tape.gradient(train_loss, self.combined_decoder.trainable_variables)
            self.optimizer.apply_gradients(zip(combined_decoder_gradients, self.combined_decoder.trainable_variables))

            return h2o_decoder_output, o2h_decoder_output, train_loss, h2o_train_loss, o2h_train_loss

        @tf.function
        def validation_step(images_H, labels_O, images_O, labels_H):
            h_encoder_output, o_encoder_output, h_skip0, h_skip1, h_skip2, h_skip3, \
            o_skip0, o_skip1, o_skip2, o_skip3 = self.combined_encoder(images_H, images_O, training=False)

            h2o_decoder_output, o2h_decoder_output = self.combined_decoder(h_encoder_output, o_encoder_output,
                                                                           h_skip0, h_skip1, h_skip2, h_skip3,
                                                                           o_skip0, o_skip1, o_skip2, o_skip3,
                                                                           training=False)

            h2o_valid_loss = loss_function(predictions=h2o_decoder_output, labels=labels_O, loss_type=self.loss_type)
            o2h_valid_loss = loss_function(predictions=o2h_decoder_output, labels=labels_H, loss_type=self.loss_type)
            valid_loss = h2o_valid_loss + o2h_valid_loss

            return h2o_decoder_output, o2h_decoder_output, valid_loss, h2o_valid_loss, o2h_valid_loss

        ckpt, manager, StartEpoch = TryRestoreExistingCkpt(tmp_config=self.config,
                                                           combined_encoder=self.combined_encoder,
                                                           combined_decoder=self.combined_decoder,
                                                           optimizer=self.optimizer,
                                                           savePath=f"{self.config['output.OutputDir']}/models")

        if StartEpoch == 0:
            if os.path.exists(self.logging_loss_filePath):
                os.remove(self.logging_loss_filePath)
            if os.path.exists(self.train_record_file):
                os.remove(self.train_record_file)
            df_loss = pd.DataFrame(columns=["epoch", "train_loss", "val_loss"])
            previous_best = np.Inf
        else:
            df_loss = pd.read_csv(self.logging_loss_filePath)
            previous_best = round(float(df_loss["val_loss"].min()), 5)
            with open(self.train_record_file, "a") as file:
                print(f"\nbest_val_loss is {previous_best} for model already trained up to {StartEpoch} epochs...!",
                      file=file)

            print(f"\nbest_val_loss is {previous_best} for model already trained up to {StartEpoch} epochs...!\n")

        print(f"Learning Rate: {self.optimizer.learning_rate.numpy()}")

        if self.config["training.ReduceLearningRate"]:
            self.reduce_LR.on_train_begin()
        if self.config["training.EarlyStopping"]:
            self.early_stopping.on_train_begin(previous_best=previous_best)

        if StartEpoch >= self.config["training.NumEpochs"]:
            print(f'Model has already trained up to {self.config["training.NumEpochs"]} epochs...!')
        else:
            for epoch in range(StartEpoch, self.config["training.NumEpochs"]):
                print("\n")
                train_loss_value = []
                train_h2o_loss = []
                train_o2h_loss = []
                valid_loss_value = []
                valid_h2o_loss = []
                valid_o2h_loss = []

                for step, inputs in enumerate(self.ds_train):
                    # print(f"Epoch: {epoch + 1} --- batch-iteration: {step + 1}/{self.ds_train.__len__()}", end="\r",
                    #       flush=True)
                    _, image_H, label_O, image_O, label_H = inputs

                    label_H = np.array([ExtractCentralRegion(x, shape=self.output_shape[:-1]) for x in label_H])
                    label_O = np.array([ExtractCentralRegion(x, shape=self.output_shape[:-1]) for x in label_O])

                    if self.config["preprocessing.NormalizeMeanStd"]:
                        image_H = normalise(image_H, self.config["data.mean"], self.config["data.mean"])
                        image_O = normalise(image_O, self.config["data.mean"], self.config["data.mean"])

                    h2o_output, o2h_output, loss, h2o_loss, o2h_loss = train_step(images_H=image_H, labels_O=label_O,
                                                                                  images_O=image_O, labels_H=label_H)
                    train_loss_value.append(loss)
                    train_h2o_loss.append(h2o_loss)
                    train_o2h_loss.append(o2h_loss)

                    # Saving plots to see the training outputs...
                    if step % 500 == 0:
                        name = "epoch_" + str(epoch+1) + "_step_" + str(step)
                        self.tensorboard.save_to_dir(image_H, image_O, label_H, label_O, h2o_output, o2h_output, name)

                for step, inputs in enumerate(self.ds_valid):
                    _, image_H, label_O, image_O, label_H = inputs

                    label_H = np.array([ExtractCentralRegion(x, shape=self.output_shape[:-1]) for x in label_H])
                    label_O = np.array([ExtractCentralRegion(x, shape=self.output_shape[:-1]) for x in label_O])

                    ###########################################################################################
                    coin = tf.less(tf.random.uniform((), 0., 1.), 0.5)
                    image_H = tf.cond(coin, lambda: tf.image.flip_left_right(image_H), lambda: image_H)
                    label_O = tf.cond(coin, lambda: tf.image.flip_left_right(label_O), lambda: label_O)
                    ###########################################################################################

                    if self.config["preprocessing.NormalizeMeanStd"]:
                        image_H = normalise(image_H, self.config["data.mean"], self.config["data.mean"])
                        image_O = normalise(image_O, self.config["data.mean"], self.config["data.mean"])

                    h2o_output, o2h_output, loss, h2o_loss, o2h_loss = validation_step(images_H=image_H,
                                                                                       labels_O=label_O,
                                                                                       images_O=image_O,
                                                                                       labels_H=label_H)
                    valid_loss_value.append(loss)
                    valid_h2o_loss.append(h2o_loss)
                    valid_o2h_loss.append(o2h_loss)

                    # Saving plots to tensorboard...
                    if step == 0:
                        if epoch % self.config["training.SaveInterval"] == 0:
                            name = "epoch_" + str(epoch+1) + "_valid"
                            self.tensorboard.save_to_dir(image_H, image_O, label_H, label_O,
                                                         h2o_output, o2h_output, name)
                            self.tensorboard.save_to_tensorboard(image_H, image_O, label_H, label_O,
                                                                 h2o_output, o2h_output, epoch+1)

                # Calculate mean of losses at each Epoch and save it to Tensorboard.
                train_loss_mean = round(float(np.mean(train_loss_value)), 5)
                train_h2o_loss_mean = round(float(np.mean(train_h2o_loss)), 5)
                train_o2h_loss_mean = round(float(np.mean(train_o2h_loss)), 5)
                valid_loss_mean = max(round(float(np.mean(valid_loss_value)), 5), 0.0010)
                valid_h2o_loss_mean = round(float(np.mean(valid_h2o_loss)), 5)
                valid_o2h_loss_mean = round(float(np.mean(valid_o2h_loss)), 5)
                with self.tensorboard.train_loss_writer.as_default():
                    tf.summary.scalar(name=self.loss_type, data=train_loss_mean, step=epoch)
                with self.tensorboard.valid_loss_writer.as_default():
                    tf.summary.scalar(name=self.loss_type, data=valid_loss_mean, step=epoch)

                with open(self.train_record_file, "a") as file:
                    print(f"\nEpoch {epoch + 1}/{self.config['training.NumEpochs']}: "
                          f"train_loss: {train_loss_mean} - train_h2o_loss: {train_h2o_loss_mean} - "
                          f"train_o2h_loss: {train_o2h_loss_mean} | val_loss: {valid_loss_mean} - "
                          f"val_h2o_loss: {valid_h2o_loss_mean} - val_o2h_loss: {valid_o2h_loss_mean}", file=file)

                print(f"\nEpoch {epoch + 1}/{self.config['training.NumEpochs']}: "
                      f"train_loss: {train_loss_mean} - train_h2o_loss: {train_h2o_loss_mean} - "
                      f"train_o2h_loss: {train_o2h_loss_mean} | val_loss: {valid_loss_mean} - "
                      f"val_h2o_loss: {valid_h2o_loss_mean} - val_o2h_loss: {valid_o2h_loss_mean}")

                SaveModelWeights(self.config, self.combined_encoder, self.combined_decoder, "models", "latest")
                self.early_stopping.on_epoch_end(epoch=epoch + 1, loss=valid_loss_mean)

                if self.early_stopping.improved_loss:
                    with open(self.train_record_file, "a") as file:
                        print(f"Epoch {epoch + 1:05d}: val_loss improved from {self.early_stopping.previous_best} to "
                              f"{self.early_stopping.best}", file=file)
                    print(f"Epoch {epoch + 1:05d}: val_loss improved from {self.early_stopping.previous_best} to "
                          f"{self.early_stopping.best}")
                    SaveModelWeights(self.config, self.combined_encoder, self.combined_decoder, "models", "best")
                else:
                    with open(self.train_record_file, "a") as file:
                        print(f"Epoch {epoch + 1:05d}: val_loss did not improved from {self.early_stopping.best}",
                              file=file)

                    print(f"Epoch {epoch + 1:05d}: val_loss did not improved from {self.early_stopping.best}")

                if self.config["training.ReduceLearningRate"]:
                    self.reduce_LR.on_epoch_end(epoch + 1, valid_loss_mean, name="cross-stain-prediction")

                df_loss = df_loss.append({"epoch": epoch + 1, "train_loss": train_loss_mean,
                                          "val_loss": valid_loss_mean}, ignore_index=True)
                df_loss.to_csv(self.logging_loss_filePath, sep=',', encoding='utf-8', index=False)

                manager.save(checkpoint_number=epoch + 1)

                if self.config["training.EarlyStopping"]:
                    if self.early_stopping.stop_training:
                        print(f"Training stopped at epoch: {self.early_stopping.stopped_epoch:05d}.")
                        print(f"Best Validation Loss recorded for early stopping: {self.early_stopping.best}")
                        break

            if self.config["training.EarlyStopping"]:
                if not self.early_stopping.stop_training:
                    print("Training completed successfully...")
                    print(f"Best Validation Loss recorded for complete training: {self.early_stopping.best}")
            else:
                print("Training completed successfully...")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Cross Stain Prediction...!")
    parser.add_argument('-c', '--configFile', type=str, default="configuration_files/CS.cfg", help='cfg file to use')
    parser.add_argument('-s', '--stain', type=str, default="02", help='which stain to train')
    parser.add_argument('-m', '--deepModel', type=str, default="UNet", help='network architecture to use')
    parser.add_argument("-r", "--repetition", type=str, default="rep1")
    parser.add_argument('-g', '--gpu', type=str, help='specify which GPU to use')

    args = parser.parse_args()

    # if args.gpu:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # else:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # for gpu in gpus:
    #     tf.config.experimental.set_memory_growth(gpu, True)
    # print("\n")
    # print("Gpu Growth Restriction Done...")
    # print("\n")

    if args.configFile:
        config = config_utils.readconfig(args.configFile)
    else:
        config = config_utils.readconfig()

    print('Command Line Input Arguments:\n', json.dumps(vars(args), indent=4, separators=(',', ':')))
    print("\n")

    if args.stain:
        config["data.Stain"] = args.stain
        config["data.TrainDataDir"] = os.path.join(config["data.TrainDataDir"], config["data.Stain"])
        config["data.ValidationDataDir"] = os.path.join(config["data.ValidationDataDir"], config["data.Stain"])
        config["output.Label"] = config["data.Stain"]
    else:
        raise ValueError("Please specify one of (\'02, 03, 16, 32, 39\') in parser.parse_args().stain.")

    if args.deepModel.lower() == "unet" or args.deepModel.lower() == "resnet50":
        config["training.DeepModel"] = args.deepModel.lower()
    else:
        raise ValueError("Please specify one of (\'UNet, ResNet50\') in parser.parse_args().DeepModel.")

    if "rep" in args.repetition:
        config["training.repetition"] = args.repetition
    else:
        raise ValueError("Please specify proper repetition in parser.parse_args().repetition.")

    config["output.DataStatsDir"] = os.path.join(config["output.OutputDir"], config["training.SSLModel"],
                                                 config["training.DeepModel"], config["data.Stain"])
    config["output.OutputDir"] = os.path.join(config["output.OutputDir"], config["training.SSLModel"],
                                              config["training.DeepModel"], config["data.Stain"],
                                              config["training.repetition"], config["training.SSLModelPhase"])

    config["training.TrainNumExamples"] = TFDataLoader(config, ssl_model_phase=config["training.SSLModelPhase"],
                                                       mode="train").GetPNGImageFiles(num_examples_mode=True)

    config["data.mean"], config["data.stddev"] = LoadDatasetMeanStd(tmp_config=config, savePath="data_statistics")
    print("\nConfiguration File Arguments:\n", json.dumps(config, indent=2, separators=(",", ":")))
    SaveConfigParserArguments(tmp_config=config, savePath="config_file")
    ViewStainSeparationSamples(tmp_config=config)
    ViewDataSamples(tmp_config=config)
    CS = Training(tmp_config=config)
    CS.train()
