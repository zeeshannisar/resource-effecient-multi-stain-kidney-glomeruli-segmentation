import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import h5py
import json
import argparse
import copy

import config_utils

import tensorflow as tf
import tensorflow_addons as tfa

import numpy as np
import pandas as pd

from plot import images_with_labels_plus_augmentations_plot
from tf_dataloader_API import TFDataLoader, Compute_Mean_Std_Dataset
from deepmodels.subclass_API import UnetEncoder, UnetDecoder, HO_UnetEncoder, HO_UnetDecoder, PostProcessHOEncoder, \
    ProjectionHead
from tensorboard_API import TensorboardLogs
from custom_callbacks.early_stopping import EarlyStoppingAtMinLoss


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


def ViewDataSamplesPlusTransforms(tmp_config):
    print(f'Saving Data Samples with Labels Plus Transforms extracted from tf.Data API.')
    ds_train = TFDataLoader(tmp_config, ssl_model_phase=tmp_config["training.SSLModelPhase"],
                            mode="train", print_data_specifications=False).LoadDataset()
    for step, (image, image_perturbed, image_H_T1, label_O_T1, image_O_T1, label_H_T1,
               image_H_T2, label_O_T2, image_O_T2, label_H_T2) in enumerate(ds_train.take(1)):
        images_with_labels_plus_augmentations_plot(image, image_perturbed, image_H_T1, label_O_T1,
                                                   image_O_T1, label_H_T1, image_H_T2, label_O_T2, image_O_T2,
                                                   label_H_T2, config['output.OutputDir'], tmp_config['data.Stain'])
    del ds_train


def GetInputShape(tmp_config):
    ds_train = TFDataLoader(tmp_config, ssl_model_phase=tmp_config["training.SSLModelPhase"],
                            mode="train", print_data_specifications=False).LoadDataset()
    _, _, image_H_T1, _, _, _, _, _, _, _ = next(iter(ds_train))
    input_shape = image_H_T1.shape[1:]
    del ds_train
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
    print("\n")


def SelectOptimizer(tmp_config):
    """Returns the optimizer."""
    if tmp_config["training.Optimiser"].lower() == 'sgd':
        return tf.keras.optimizers.SGD(tmp_config["training.LearningRate"], momentum=0.9, nesterov=True)
    elif tmp_config["training.Optimiser"].lower() == 'adam':
        return tf.keras.optimizers.Adam(tmp_config["training.LearningRate"])
    elif tmp_config["training.Optimiser"].lower() == 'adamw':
        return tfa.optimizers.AdamW(learning_rate=tmp_config["training.LearningRate"], weight_decay=1e-8)
    else:
        raise ValueError('Unknown optimizer {}'.format(tmp_config["config.filename"]))


def TryRestoreExistingCkpt(tmp_config, combined_encoder, target_combined_encoder, post_process, target_post_process,
                           projector, target_projector, predictor, optimizer, savePath):
    checkpoint = tf.train.Checkpoint(combined_encoder=combined_encoder,
                                     target_combined_encoder=target_combined_encoder,
                                     post_process=post_process, target_post_process=target_post_process,
                                     online_projector=projector, target_projector=target_projector,
                                     predictor=predictor, optimizer=optimizer)

    manager = tf.train.CheckpointManager(checkpoint, directory=os.path.join(savePath, "TF_checkpoints"),
                                         max_to_keep=tmp_config["training.MaxNumCheckpoints"],
                                         checkpoint_name='latest_ckpt', step_counter=None,
                                         checkpoint_interval=None, init_fn=None)
    # expect_partial(): Silence warnings about incomplete checkpoint restores. Warnings are otherwise printed for unused
    # parts of the checkpoint file or object when the Checkpoint object is deleted (often at program shutdown).
    checkpoint.restore(manager.latest_checkpoint).expect_partial()
    if manager.latest_checkpoint:
        start_epoch = int(manager.latest_checkpoint.rsplit("/", 1)[1].rsplit("-", 1)[1])
        print(f"\nTraining is resumed from epoch {start_epoch} using checkpoint: {manager.latest_checkpoint}...!")
    else:
        print("\nTraining is started from scratch...!")
        start_epoch = 0
    return checkpoint, manager, start_epoch


def WriteModelArchitecture(model, path, name=None):
    # summary of Encoder model
    print(f"Saving {name} summary...")
    with open(path + f'/{name}_model_summary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))


def SaveModelArchitecture(encoder, decoder, combined_encoder, combined_decoder, post_process,
                          projector, predictor, input_shape, savePath):
    savePath = CreateDir(savePath=savePath)

    encoder_output, skip0, skip1, skip2, skip3 = encoder(tf.ones(shape=(1, *input_shape)), training=True)
    WriteModelArchitecture(model=encoder.build_graph(input_shape), path=savePath, name="UnetEncoder")

    decoder_output = decoder(encoder_output, skip0, skip1, skip2, skip3, training=True)
    WriteModelArchitecture(model=decoder.build_graph(encoder_output.shape[1:], skip0.shape[1:], skip1.shape[1:],
                                                     skip2.shape[1:], skip3.shape[1:]),
                           path=savePath, name="UnetDecoder")

    h_encoder_output, o_encoder_output, h_skip0, h_skip1, h_skip2, h_skip3, \
    o_skip0, o_skip1, o_skip2, o_skip3 = combined_encoder(tf.ones(shape=(1, *input_shape)),
                                                          tf.ones(shape=(1, *input_shape)), training=True)
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

    post_process_output = post_process(h_encoder_output, o_encoder_output)
    WriteModelArchitecture(model=post_process.build_graph(h_encoder_output.shape[1:], o_encoder_output.shape[1:]),
                           path=savePath, name="PostProcess_HOEncoder")

    projector_output = projector(post_process_output)
    WriteModelArchitecture(model=projector.build_graph(post_process_output.shape[1:]), path=savePath, name="Projector")

    predictor_output = predictor(projector_output)
    WriteModelArchitecture(model=predictor.build_graph(projector_output.shape[1:]), path=savePath, name="Predictor")


def LoadDatasetMeanStd(tmp_config, savePath):
    savePath = CreateDir(savePath=os.path.join(tmp_config["output.DataStatsDir"], savePath))
    stats_filename = os.path.join(savePath, "normalisation_stats.hdf5")
    if os.path.exists(stats_filename):
        tmp_config["preprocessing.ComputeMeanStd"] = False

    if tmp_config["preprocessing.ComputeMeanStd"]:
        filenames = TFDataLoader(tmp_config, ssl_model_phase=tmp_config["training.SSLModelPhase"]).GetPNGImageFiles(num_examples_mode=False)
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


def DecoderLossFunction(predictions, labels, loss_type="mse_loss"):
    if loss_type == "mae_loss":
        return tf.reduce_mean(tf.abs(tf.subtract(predictions, labels)))
    if loss_type == "mse_loss":
        return tf.reduce_mean(tf.square(tf.subtract(predictions, labels)))


def ByolLossFunction(opred1, opred2, tproj1, tproj2):
    opred1 = tf.math.l2_normalize(opred1, axis=1)
    opred2 = tf.math.l2_normalize(opred2, axis=1)
    tproj1 = tf.math.l2_normalize(tproj1, axis=1)
    tproj2 = tf.math.l2_normalize(tproj2, axis=1)

    similarities1 = 2 - 2 * tf.reduce_mean((tf.reduce_sum(tf.multiply(opred1, tproj2), axis=1)))
    similarities2 = 2 - 2 * tf.reduce_mean((tf.reduce_sum(tf.multiply(opred2, tproj1), axis=1)))
    return 0.5 * similarities1 + 0.5 * similarities2


def CsCoLoss(online_pred_T1, online_pred_T2, target_proj_T1, target_proj_T2,
             H2O_decoder_T1, O2H_decoder_T1, labels_O_T1, labels_H_T1, decoder_loss_type, decoder_loss_w, byol_loss_w):
    H2O_loss = DecoderLossFunction(H2O_decoder_T1, labels_O_T1, decoder_loss_type)
    O2H_loss = DecoderLossFunction(O2H_decoder_T1, labels_H_T1, decoder_loss_type)
    decoder_loss = H2O_loss + O2H_loss
    byol_loss = ByolLossFunction(opred1=online_pred_T1, opred2=online_pred_T2,
                                 tproj1=target_proj_T1, tproj2=target_proj_T2)
    csco_loss = (byol_loss_w * byol_loss) + (decoder_loss_w * decoder_loss)
    return csco_loss, byol_loss, decoder_loss


def SaveModelWeights(tmp_config, combined_encoder, savePath, saveName="best"):
    savePath = CreateDir(savePath=os.path.join(tmp_config["output.OutputDir"], savePath))

    combined_encoder.save_weights(os.path.join(savePath, f"HO_encoder_model.{saveName}.hdf5"))


def UpdateTargetNetworks(online_combined_encoder, target_combined_encoder,
                         online_projector, target_projector, beta=0.99):
    # Update target networks (exponential moving average of online networks)
    online_combined_encoder_weights = online_combined_encoder.get_weights()
    target_combined_encoder_weights = target_combined_encoder.get_weights()
    for i in range(len(online_combined_encoder_weights)):
        target_combined_encoder_weights[i] = beta * target_combined_encoder_weights[i] + (1 - beta) * \
                                             online_combined_encoder_weights[i]
    target_combined_encoder.set_weights(target_combined_encoder_weights)

    online_projector_weights = online_projector.get_weights()
    target_projector_weights = target_projector.get_weights()
    for i in range(len(online_projector_weights)):
        target_projector_weights[i] = beta * target_projector_weights[i] + (1 - beta) * online_projector_weights[i]
    target_projector.set_weights(target_projector_weights)

    return target_combined_encoder, target_projector


class Training:
    def __init__(self, tmp_config, pretrained_model_path=None):
        self.config = tmp_config

        if self.config["training.DeepModel"].lower() == "unet":
            # Online Networks
            self.encoder, self.decoder = UnetEncoder(), UnetDecoder()
            self.combined_encoder, self.combined_decoder = HO_UnetEncoder(), HO_UnetDecoder()

        self.input_shape = GetInputShape(self.config)
        print(f"\ninput_shape: {self.input_shape}")

        self.output_shape = GetOutputShape(self.combined_encoder, self.combined_decoder,
                                           self.input_shape, self.config["training.BatchSize"])
        print(f"\noutput_shape: {self.output_shape}")

        if pretrained_model_path is not None:
            try:
                print("Loading Pretrained Encoder Model Weights trained for cros-stain-prediction...")
                self.combined_encoder.load_weights(os.path.join(pretrained_model_path, f"HO_encoder_model.best.hdf5"))
                self.combined_encoder.trainable = True
                print("Loading Pretrained Decoder Model Weights trained for cros-stain-prediction ...")
                self.combined_decoder.load_weights(os.path.join(pretrained_model_path, f"HO_decoder_model.best.hdf5"))
                self.combined_decoder.trainable = False
            except:
                raise ValueError("Error in loading cross-stain-prediction models. Please confirm if it has already been trained otherwise double check the path.")

        self.post_process = PostProcessHOEncoder()
        self.projector = ProjectionHead()
        self.predictor = ProjectionHead()

        # Target Networks
        self.target_combined_encoder = copy.deepcopy(self.combined_encoder)
        self.target_post_process = copy.deepcopy(self.post_process)
        self.target_projector = copy.deepcopy(self.projector)

        if self.config["training.EarlyStopping"]:
            self.early_stopping = EarlyStoppingAtMinLoss(patience=self.config["training.Patience"])

        self.optimizer = SelectOptimizer(self.config)
        self.tensorboard = TensorboardLogs(config=self.config)

        self.train_record_file = os.path.join(self.config["output.OutputDir"], "training_record.txt")
        self.logging_loss_filePath = os.path.join(self.config["output.OutputDir"], "training_loss.csv")

        self.ds_train = TFDataLoader(self.config, ssl_model_phase=self.config["training.SSLModelPhase"],
                                     mode="train", print_data_specifications=True).LoadDataset()
        self.ds_valid = TFDataLoader(self.config, ssl_model_phase=self.config["training.SSLModelPhase"],
                                     mode="validation", print_data_specifications=True).LoadDataset()

    def train(self):
        print("\n")
        SaveModelArchitecture(self.encoder, self.decoder, self.combined_encoder, self.combined_decoder,
                              self.post_process, self.projector, self.predictor, self.input_shape,
                              savePath=f"{self.config['output.OutputDir']}/network_summary")

        @tf.function
        def train_step(images_H_T1, images_H_T2, images_O_T1, images_O_T2, labels_O_T1, labels_H_T1):
            # For Augmented Transform 1
            target_h_encoder_output_T1, target_o_encoder_output_T1, _, _, _, _, \
            _, _, _, _ = self.target_combined_encoder(images_H_T1, images_O_T1, training=True)
            target_post_process_output_T1 = self.target_post_process(target_h_encoder_output_T1,
                                                                     target_o_encoder_output_T1, training=True)
            target_projector_T1 = self.target_projector(target_post_process_output_T1, training=True)

            # For Augmented Transform 2
            target_h_encoder_output_T2, target_o_encoder_output_T2, _, _, _, _, \
            _, _, _, _ = self.target_combined_encoder(images_H_T2, images_O_T2, training=True)
            target_post_process_output_T2 = self.target_post_process(target_h_encoder_output_T2,
                                                                     target_o_encoder_output_T2, training=True)
            target_projector_T2 = self.target_projector(target_post_process_output_T2, training=True)

            # persistent is set to True because the tape is used more than once to calculate the gradients.
            with tf.GradientTape(persistent=True) as tape:
                # For Augmented Transform 1
                online_h_encoder_output_T1, online_o_encoder_output_T1, online_h_skip0_T1, online_h_skip1_T1, \
                online_h_skip2_T1, online_h_skip3_T1, online_o_skip0_T1, online_o_skip1_T1, online_o_skip2_T1, \
                online_o_skip3_T1 = self.combined_encoder(images_H_T1, images_O_T1, training=True)

                online_o_decoder_output_T1, \
                online_h_decoder_output_T1 = self.combined_decoder(online_h_encoder_output_T1,
                                                                   online_o_encoder_output_T1,
                                                                   online_h_skip0_T1,
                                                                   online_h_skip1_T1,
                                                                   online_h_skip2_T1,
                                                                   online_h_skip3_T1,
                                                                   online_o_skip0_T1,
                                                                   online_o_skip1_T1,
                                                                   online_o_skip2_T1,
                                                                   online_o_skip3_T1, training=True)

                online_post_process_output_T1 = self.post_process(online_h_encoder_output_T1,
                                                                  online_o_encoder_output_T1, training=True)

                online_projector_T1 = self.projector(online_post_process_output_T1, training=True)
                online_predictor_T1 = self.predictor(online_projector_T1, training=True)

                # For Augmented Transform 2
                online_h_encoder_output_T2, online_o_encoder_output_T2, _, _, _, _, \
                _, _, _, _ = self.combined_encoder(images_H_T2, images_O_T2, training=True)

                online_post_process_output_T2 = self.post_process(online_h_encoder_output_T2,
                                                                  online_o_encoder_output_T2, training=True)
                online_projector_T2 = self.projector(online_post_process_output_T2, training=True)
                online_predictor_T2 = self.predictor(online_projector_T2, training=True)

                csco_loss, byol_loss, decoder_loss = CsCoLoss(online_pred_T1=online_predictor_T1,
                                                              online_pred_T2=online_predictor_T2,
                                                              target_proj_T1=target_projector_T1,
                                                              target_proj_T2=target_projector_T2,
                                                              H2O_decoder_T1=online_o_decoder_output_T1,
                                                              O2H_decoder_T1=online_h_decoder_output_T1,
                                                              labels_O_T1=labels_O_T1,
                                                              labels_H_T1=labels_H_T1,
                                                              decoder_loss_type=GetDecoderLossType(self.config),
                                                              decoder_loss_w=self.config["training.DecoderLossWeights"],
                                                              byol_loss_w=self.config["training.ByolLossWeights"])

            # Backward pass (update online networks)
            combined_encoder_grads = tape.gradient(csco_loss, self.combined_encoder.trainable_variables)
            self.optimizer.apply_gradients(zip(combined_encoder_grads, self.combined_encoder.trainable_variables))
            post_process_grads = tape.gradient(csco_loss, self.post_process.trainable_variables)
            self.optimizer.apply_gradients(zip(post_process_grads, self.post_process.trainable_variables))
            projector_grads = tape.gradient(csco_loss, self.projector.trainable_variables)
            self.optimizer.apply_gradients(zip(projector_grads, self.projector.trainable_variables))
            predictor_grads = tape.gradient(csco_loss, self.predictor.trainable_variables)
            self.optimizer.apply_gradients(zip(predictor_grads, self.predictor.trainable_variables))
            return online_o_decoder_output_T1, online_h_decoder_output_T1, csco_loss, byol_loss, decoder_loss

        @tf.function
        def validation_step(images_H_T1, images_H_T2, images_O_T1, images_O_T2, labels_O_T1, labels_H_T1):
            # For Augmented Transform 1
            target_h_encoder_output_T1, target_o_encoder_output_T1, _, _, _, _, \
            _, _, _, _ = self.target_combined_encoder(images_H_T1, images_O_T1, training=False)
            target_post_process_output_T1 = self.target_post_process(target_h_encoder_output_T1,
                                                                     target_o_encoder_output_T1, training=False)
            target_projector_T1 = self.target_projector(target_post_process_output_T1, training=False)

            # For Augmented Transform 2
            target_h_encoder_output_T2, target_o_encoder_output_T2, _, _, _, _, \
            _, _, _, _ = self.target_combined_encoder(images_H_T2, images_O_T2, training=False)
            target_postprocess_output_T2 = self.target_post_process(target_h_encoder_output_T2,
                                                                    target_o_encoder_output_T2,
                                                                    training=False)
            target_projector_T2 = self.target_projector(target_postprocess_output_T2, training=False)

            # For Augmented Transform 1
            online_h_encoder_output_T1, online_o_encoder_output_T1, online_h_skip0_T1, online_h_skip1_T1, \
            online_h_skip2_T1, online_h_skip3_T1, online_o_skip0_T1, online_o_skip1_T1, online_o_skip2_T1, \
            online_o_skip3_T1 = self.combined_encoder(images_H_T1, images_O_T1, training=False)

            online_h2o_decoder_output_T1, \
            online_o2h_decoder_output_T1 = self.combined_decoder(online_h_encoder_output_T1,
                                                                 online_o_encoder_output_T1,
                                                                 online_h_skip0_T1,
                                                                 online_h_skip1_T1,
                                                                 online_h_skip2_T1,
                                                                 online_h_skip3_T1,
                                                                 online_o_skip0_T1,
                                                                 online_o_skip1_T1,
                                                                 online_o_skip2_T1,
                                                                 online_o_skip3_T1, training=False)

            online_post_process_output_T1 = self.post_process(online_h_encoder_output_T1,
                                                              online_o_encoder_output_T1, training=False)

            online_projector_T1 = self.projector(online_post_process_output_T1, training=False)
            online_predictor_T1 = self.predictor(online_projector_T1, training=False)

            # For Augmented Transform 2
            online_h_encoder_output_T2, online_o_encoder_output_T2, _, _, _, _, \
            _, _, _, _ = self.combined_encoder(images_H_T2, images_O_T2, training=False)

            online_post_process_output_T2 = self.post_process(online_h_encoder_output_T2,
                                                              online_o_encoder_output_T2, training=False)
            online_projector_T2 = self.projector(online_post_process_output_T2, training=False)
            online_predictor_T2 = self.predictor(online_projector_T2, training=False)

            csco_loss, byol_loss, decoder_loss = CsCoLoss(online_pred_T1=online_predictor_T1,
                                                          online_pred_T2=online_predictor_T2,
                                                          target_proj_T1=target_projector_T1,
                                                          target_proj_T2=target_projector_T2,
                                                          H2O_decoder_T1=online_h2o_decoder_output_T1,
                                                          O2H_decoder_T1=online_o2h_decoder_output_T1,
                                                          labels_O_T1=labels_O_T1,
                                                          labels_H_T1=labels_H_T1,
                                                          decoder_loss_type=GetDecoderLossType(self.config),
                                                          decoder_loss_w=self.config["training.DecoderLossWeights"],
                                                          byol_loss_w=self.config["training.ByolLossWeights"])
            return online_h2o_decoder_output_T1, online_o2h_decoder_output_T1, csco_loss, byol_loss, decoder_loss

        ckpt, manager, StartEpoch = TryRestoreExistingCkpt(tmp_config=self.config,
                                                           combined_encoder=self.combined_encoder,
                                                           target_combined_encoder=self.target_combined_encoder,
                                                           post_process=self.post_process,
                                                           target_post_process=self.target_post_process,
                                                           projector=self.projector,
                                                           target_projector=self.target_projector,
                                                           predictor=self.predictor,
                                                           optimizer=self.optimizer,
                                                           savePath=f"{self.config['output.OutputDir']}/models")

        if StartEpoch == 0:
            if os.path.exists(self.logging_loss_filePath):
                os.remove(self.logging_loss_filePath)
            if os.path.exists(self.train_record_file):
                os.remove(self.train_record_file)
            df_loss = pd.DataFrame(columns=["epoch", "train_csco_loss", "train_byol_loss", "train_decoder_loss",
                                            "val_csco_loss", "val_byol_loss", "val_decoder_loss"])
            previous_best = np.Inf
        else:
            df_loss = pd.read_csv(self.logging_loss_filePath)
            previous_best = round(float(df_loss["val_csco_loss"].min()), 5)
            with open(self.train_record_file, "a") as file:
                print(f"\nbest_val_loss is {previous_best} for model already trained up to {StartEpoch} epochs...!",
                      file=file)

            print(f"\nbest_val_loss is {previous_best} for model already trained up to {StartEpoch} epochs...!\n")

        if self.config["training.EarlyStopping"]:
            self.early_stopping.on_train_begin(previous_best=previous_best)

        if StartEpoch >= self.config["training.NumEpochs"]:
            print(f'Model has already trained up to {self.config["training.NumEpochs"]} epochs...!')
        else:
            for epoch in range(StartEpoch, self.config["training.NumEpochs"]):
                print("\n")

                train_csco_loss = []
                train_byol_loss = []
                train_decoder_loss = []
                valid_csco_loss = []
                valid_byol_loss = []
                valid_decoder_loss = []

                for step, inputs in enumerate(self.ds_train):
                    # print(f"Epoch: {epoch + 1} --- batch-iteration: {step + 1}/{self.ds_train.__len__()}", end="\r",
                    #       flush=True)

                    _, _, image_H_T1, label_O_T1, image_O_T1, label_H_T1, \
                    image_H_T2, label_O_T2, image_O_T2, label_H_T2 = inputs

                    label_H_T1 = np.array([ExtractCentralRegion(x, shape=self.output_shape[:-1]) for x in label_H_T1])
                    label_O_T1 = np.array([ExtractCentralRegion(x, shape=self.output_shape[:-1]) for x in label_O_T1])

                    if self.config["preprocessing.NormalizeMeanStd"]:
                        image_H_T1 = normalise(image_H_T1, self.config["data.mean"], self.config["data.mean"])
                        image_H_T2 = normalise(image_H_T2, self.config["data.mean"], self.config["data.mean"])
                        image_O_T1 = normalise(image_O_T1, self.config["data.mean"], self.config["data.mean"])
                        image_O_T2 = normalise(image_O_T2, self.config["data.mean"], self.config["data.mean"])

                    h2o_output, o2h_output, csco_loss, byol_loss, decoder_loss = train_step(images_H_T1=image_H_T1,
                                                                                            images_H_T2=image_H_T2,
                                                                                            images_O_T1=image_O_T1,
                                                                                            images_O_T2=image_O_T2,
                                                                                            labels_O_T1=label_O_T1,
                                                                                            labels_H_T1=label_H_T1)
                    train_csco_loss.append(csco_loss)
                    train_byol_loss.append(byol_loss)
                    train_decoder_loss.append(decoder_loss)

                    self.target_combined_encoder, self.target_projector = UpdateTargetNetworks(self.combined_encoder,
                                                                                               self.target_combined_encoder,
                                                                                               self.projector,
                                                                                               self.target_projector)

                    # Saving plots to see the training outputs...
                    if step % 500 == 0:
                        name = "epoch_" + str(epoch + 1) + "_step_" + str(step)
                        self.tensorboard.save_to_dir(image_H_T1, image_O_T1,
                                                     label_H_T1, label_O_T1, h2o_output, o2h_output, name)

                for step, inputs in enumerate(self.ds_valid):
                    _, _, image_H_T1, label_O_T1, image_O_T1, label_H_T1, \
                    image_H_T2, label_O_T2, image_O_T2, label_H_T2 = inputs

                    label_H_T1 = np.array([ExtractCentralRegion(x, shape=self.output_shape[:-1]) for x in label_H_T1])
                    label_O_T1 = np.array([ExtractCentralRegion(x, shape=self.output_shape[:-1]) for x in label_O_T1])

                    if self.config["preprocessing.NormalizeMeanStd"]:
                        image_H_T1 = normalise(image_H_T1, self.config["data.mean"], self.config["data.mean"])
                        image_H_T2 = normalise(image_H_T2, self.config["data.mean"], self.config["data.mean"])
                        image_O_T1 = normalise(image_O_T1, self.config["data.mean"], self.config["data.mean"])
                        image_O_T2 = normalise(image_O_T2, self.config["data.mean"], self.config["data.mean"])

                    h2o_output, o2h_output, csco_loss, byol_loss, decoder_loss = validation_step(images_H_T1=image_H_T1,
                                                                                                 images_H_T2=image_H_T2,
                                                                                                 images_O_T1=image_O_T1,
                                                                                                 images_O_T2=image_O_T2,
                                                                                                 labels_O_T1=label_O_T1,
                                                                                                 labels_H_T1=label_H_T1)
                    valid_csco_loss.append(csco_loss)
                    valid_byol_loss.append(byol_loss)
                    valid_decoder_loss.append(decoder_loss)
                    # Saving plots to tensorboard...
                    if step == 0:
                        if epoch % self.config["training.SaveInterval"] == 0:
                            name = "epoch_" + str(epoch + 1) + "_valid"
                            self.tensorboard.save_to_dir(image_H_T1, image_O_T1,
                                                         label_H_T1, label_O_T1, h2o_output, o2h_output, name)
                            self.tensorboard.save_to_tensorboard(image_H_T1, image_O_T1, label_H_T1, label_O_T1,
                                                                 h2o_output, o2h_output, epoch + 1)

                # Calculate mean of losses at each Epoch and save it to Tensorboard.
                train_csco_loss_mean = round(float(np.mean(train_csco_loss)), 5)
                train_byol_loss_mean = round(float(np.mean(train_byol_loss)), 5)
                train_decoder_loss_mean = round(float(np.mean(train_decoder_loss)), 5)

                valid_csco_loss_mean = round(float(np.mean(valid_csco_loss)), 5)
                valid_byol_loss_mean = round(float(np.mean(valid_byol_loss)), 5)
                valid_decoder_loss_mean = round(float(np.mean(valid_decoder_loss)), 5)

                with self.tensorboard.train_loss_writer.as_default():
                    tf.summary.scalar(name="csco_loss", data=train_csco_loss_mean, step=epoch)
                with self.tensorboard.valid_loss_writer.as_default():
                    tf.summary.scalar(name="csco_loss", data=valid_csco_loss_mean, step=epoch)

                with open(self.train_record_file, "a") as file:
                    print(f"Epoch {epoch + 1}/{self.config['training.NumEpochs']}: "
                          f"train_csco_loss: {train_csco_loss_mean} - train_byol_loss: {train_byol_loss_mean} - "
                          f"train_decoder_loss: {train_decoder_loss_mean} | val_csco_loss: {valid_csco_loss_mean} - "
                          f"val_byol_loss: {valid_byol_loss_mean} - val_decoder_loss: {valid_decoder_loss_mean}",
                          file=file)

                print(f"Epoch {epoch + 1}/{self.config['training.NumEpochs']}: "
                      f"train_csco_loss: {train_csco_loss_mean} - train_byol_loss: {train_byol_loss_mean} - "
                      f"train_decoder_loss: {train_decoder_loss_mean} | val_csco_loss: {valid_csco_loss_mean} - "
                      f"val_byol_loss: {valid_byol_loss_mean} - val_decoder_loss: {valid_decoder_loss_mean}")

                SaveModelWeights(self.config, self.combined_encoder, "models", saveName="latest")
                self.early_stopping.on_epoch_end(epoch=epoch + 1, loss=valid_csco_loss_mean)

                if self.early_stopping.improved_loss:
                    with open(self.train_record_file, "a") as file:
                        print(f"Epoch {epoch + 1:05d}: val_loss improved from {self.early_stopping.previous_best} to "
                              f"{self.early_stopping.best}", file=file)
                    print(f"Epoch {epoch + 1:05d}: val_loss improved from {self.early_stopping.previous_best} to "
                          f"{self.early_stopping.best}")
                    SaveModelWeights(self.config, self.combined_encoder, "models", saveName="best")
                else:
                    with open(self.train_record_file, "a") as file:
                        print(f"Epoch {epoch + 1:05d}: val_loss did not improved from {self.early_stopping.best}",
                              file=file)

                    print(f"Epoch {epoch + 1:05d}: val_loss did not improved from {self.early_stopping.best}")

                df_loss = df_loss.append({"epoch": epoch + 1, "train_csco_loss": train_csco_loss_mean,
                                          "train_byol_loss": train_byol_loss_mean,
                                          "train_decoder_loss": train_decoder_loss_mean,
                                          "val_csco_loss": valid_csco_loss_mean, "val_byol_loss": valid_byol_loss_mean,
                                          "val_decoder_loss": valid_decoder_loss_mean}, ignore_index=True)

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
    parser = argparse.ArgumentParser(description="Train Complete CS_CO model...!")
    parser.add_argument('-c', '--configFile', type=str, default="configuration_files/CSCO.cfg", help='cfg file to use')
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
    ViewDataSamplesPlusTransforms(tmp_config=config)

    pretrained_cs_model_path = f"{config['output.OutputDir'].rsplit('/', 1)[0]}/cross_stain_prediction/models"

    csco = Training(tmp_config=config, pretrained_model_path=pretrained_cs_model_path)
    csco.train()
