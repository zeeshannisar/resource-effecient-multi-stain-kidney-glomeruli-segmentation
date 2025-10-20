# -*- coding: utf-8 -*-
# Odyssee Merveille 25/10/17
"""
pixel_based_evaluations: convert the segmentation results into detection results based on the comparison of different
thresholds and their best F1 score
"""
import numpy
from utils import image_utils, config_utils, filepath_utils
from utils.evaluation_metrics import threshold_evaluation, evaluate_detection, precision_recall_f1_accuracy
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import argparse
import csv
import glob
import warnings
import json
import tensorflow as tf


def evaluate_thresholds(filePath, segmentationpath, label, class_definitions, detectionpath, patientlist, lod,
                        stain, outputprefix=""):
    """

       evaluate_threshold:  Evaluate detector segmentations

       :param modelpath: (string) the path to the output directory
       :param segmentationpath: (string) the path to the output where the segmentation result will be saved
       :param label: (string) the label of the trained network
       :param class_definitions: dictionary of (string, tuple) that contains the classlabel (integer), extraction method
       (random or centred), and number of samples to extract for each class, a value of -1 means extract all possible
       patches
       :param detectionpath: (string) the path where the thresholded output will be saved
       :param patientlist: (list of string) group of patients to use
       :param lod: (int) level of detail
       :param stain: (string) the type of stain used
       :param outputprefix: (string) the output prefix used
    """

    min_threshold = 0
    max_threshold = 255
    num_thresholds = 256

    nb_classes = len(class_definitions)

    detectionpath_indiv = os.path.join(detectionpath, label, "individualthreshold")
    if not os.path.exists(detectionpath_indiv):
        os.makedirs(detectionpath_indiv)

    detectionpath_global = os.path.join(detectionpath, label, "globalthreshold")
    if not os.path.exists(detectionpath_global):
        os.makedirs(detectionpath_global)

    if not os.path.exists(filePath.get_result_path(label)):
        os.makedirs(filePath.get_result_path(label))

    tps = numpy.zeros((num_thresholds, nb_classes), numpy.uint)
    tns = numpy.zeros((num_thresholds, nb_classes), numpy.uint)
    fps = numpy.zeros((num_thresholds, nb_classes), numpy.uint)
    fns = numpy.zeros((num_thresholds, nb_classes), numpy.uint)

    predictionorder = []
    for className in class_definitions.keys():
        predictionorder.append(class_definitions[className][0])
    indexes = numpy.argsort(predictionorder)
    classNames = [list(class_definitions)[i] for i in indexes]

    modelsegmentationpath = os.path.join(segmentationpath, label)

    #cm = plt.get_cmap('jet')
    with open(os.path.join(filePath.get_result_path(label), outputprefix + "results_threshold_%s.%s.txt" % (stain, label)), 'w') as f:

        writer = csv.writer(f, delimiter='\t')

        writer.writerow(numpy.array(('patientNb', 'bestSeuil', 'F1', 'precision', 'recall', 'accuracy')))
        print("Image segmentation: %s " % os.path.join(modelsegmentationpath, list(class_definitions)[0]))
        print("Patients (%d): %s" % (len(patientlist), ', '.join(patientlist)))
        print("Stain: %s" % stain)

        for imagePath, imageName in filePath.get_images_with_list_patients(os.path.join(modelsegmentationpath, list(class_definitions)[0]), staincode=stain, patients=patientlist):

            print("File: %s" % imagePath)

            prediction = image_utils.read_segmentations(modelsegmentationpath, os.path.basename(imagePath), classNames)

            gtfilename = filePath.get_groundtruth(imageName, lod)
            gt = image_utils.read_image(gtfilename)

            labelsPath = filePath.generate_groundtruthlabelspath(imageName, lod)
            with open(labelsPath, 'r') as fp:
                classLabels = json.load(fp)

            gt = image_utils.relabel_gt(gt, classLabels, class_definitions)

            mask = numpy.any(prediction != 255, axis=-1)
            if not mask.any():
                mask = None

            cl_tps, cl_tns, cl_fps, cl_fns, bestP, bestR, bestF1, bestAcc, bestclps, bestclrs, bestclf1s, bestclaccs, seuil = threshold_evaluation(
                prediction,
                gt,
                min_threshold,
                max_threshold,
                num_thresholds,
                nb_classes,
                mask=mask,
                verbose=1)

            tps = numpy.add(cl_tps, tps)
            tns = numpy.add(cl_tns, tns)
            fps = numpy.add(cl_fps, fps)
            fns = numpy.add(cl_fns, fns)

            writer.writerow(['%s' % imageName, '%.3f' % seuil, '%.3f' % bestF1, '%.3f' % bestP, '%.3f' % bestR, '%.3f' % bestAcc])
            for className, cl_f1, cl_p, cl_r, cl_acc in zip(classNames, bestclf1s, bestclps, bestclrs, bestclaccs):
                writer.writerow(['\t', '%s' % className, '%.3f' % cl_f1, '%.3f' % cl_p, '%.3f' % cl_r, '%.3f' % cl_acc])

            prediction[prediction != prediction.max(axis=-1, keepdims=1)] = 0
            predictionSeuil = numpy.argmax(prediction >= seuil, axis=-1)
            image_utils.save_image(predictionSeuil.astype('uint8'), os.path.join(detectionpath_indiv, imageName + '.png'))

            # Causes memory error with large images
            # Save plot
            #colored_result = cm(predictionSeuil.astype(numpy.uint8) + (mask.astype(numpy.uint8) * 2))[:, :, :3] * 255

            #displayOutputPath = os.path.join(detectionpath, imageName + "_display.png")
            #image_utils.save_image(colored_result.astype(numpy.uint8), displayOutputPath)

        # Calculate global statistics
        ps, rs, f1s, accs = precision_recall_f1_accuracy(tps, tns, fps, fns)

        p = numpy.mean(ps[:, 1:], axis=1)
        r = numpy.mean(rs[:, 1:], axis=1)
        f1 = numpy.mean(f1s[:, 1:], axis=1)
        acc = numpy.mean(accs[:, 1:], axis=1)

        bestF1 = max(f1)
        seuils = numpy.linspace(min_threshold, max_threshold, num=num_thresholds)
        bestseuil_idx = numpy.where(f1 == bestF1)[0]
        bestseuil_idx = numpy.random.choice(bestseuil_idx, 1) # In case more than one max F1 exists

        seuil = seuils[bestseuil_idx]

        # if the best seuil isn't unique take the first
        if len(seuil) != 1:
            seuil = seuil[-1]

        bestP = p[bestseuil_idx]
        bestR = r[bestseuil_idx]
        bestAcc = acc[bestseuil_idx]

        writer.writerow(['overall', '%.3f' % seuil, '%.3f' % bestF1, '%.3f' % bestP, '%.3f' % bestR, '%.3f' % bestAcc])
        for className, cl_f1, cl_p, cl_r, cl_acc in zip(classNames, f1s[bestseuil_idx,:][0].tolist(), ps[bestseuil_idx,:][0].tolist(), rs[bestseuil_idx,:][0].tolist(), accs[bestseuil_idx,:][0].tolist()):
            writer.writerow(['\t', '%s' % className, '%.3f' % cl_f1, '%.3f' % cl_p, '%.3f' % cl_r, '%.3f' % cl_acc])

    #cm = plt.get_cmap('jet')
    for imagePath, imageName in filePath.get_images_with_list_patients(
            os.path.join(modelsegmentationpath, list(class_definitions)[0]), staincode=stain, patients=patientlist):

        prediction = image_utils.read_segmentations(modelsegmentationpath, os.path.basename(imagePath), classNames)

        prediction[prediction != prediction.max(axis=-1, keepdims=1)] = 0
        predictionSeuil = numpy.argmax(prediction >= seuil, axis=-1)

        image_utils.save_image(predictionSeuil.astype('uint8'), os.path.join(detectionpath_global, imageName + '.png'))

        # Causes memory error with large images
        # maskfilename = os.path.join(maskpath, imageName + "_mask_" + labelName + "_lod" + str(lod) + ".png")
        # mask = image_utils.read_binary_image(maskfilename)

        # colored_result = cm(predictionSeuil.astype(numpy.uint8) + (mask.astype(numpy.uint8) * 2))[:, :, :3] * 255

        # displayOutputPath = os.path.join(detectionpath, imageName + "_display.png")
        # image_utils.save_image(colored_result.astype(numpy.uint8), displayOutputPath)

    return seuil


'''
Evaluate detector segmentations using a specified threshold.
'''
def evaluate_threshold(filePath, segmentationpath, label, class_definitions, detectionpath, patientlist, lod,
                       stain, threshold, outputprefix=""):
    """

    evaluate_threshold: Evaluate detector segmentations using a specified threshold.


    :param modelpath: (string) the path to the output directory
    :param segmentationpath: (string) the path to where the segmentation results will be saved
    :param label: (string) the label of the trained network
    :param class_definitions: dictionary of (string, tuple) that contains the classlabel (integer), extraction method
    (random or centred), and number of samples to extract for each class, a value of -1 means extract all possible
    patches
    :param detectionpath: (string) the path where the thresholded output will be saved
    :param patientlist: (list of string) group of patients to use
    :param lod: (int) level of detail
    :param stain: (string) the type of stain used
    :param threshold: (float) the threshold to be used
    :param outputprefix: (string) the output prefix used
    :return (int) the same value than threshold
    """

    min_threshold = threshold
    max_threshold = threshold
    num_thresholds = 1

    nb_classes = len(class_definitions)

    detectionpath = os.path.join(detectionpath, label, "trainingthreshold")
    if not os.path.exists(detectionpath):
        os.makedirs(detectionpath)

    if not os.path.exists(filePath.get_result_path(label)):
        os.makedirs(filePath.get_result_path(label))

    tps = numpy.zeros((num_thresholds, nb_classes), numpy.uint)
    tns = numpy.zeros((num_thresholds, nb_classes), numpy.uint)
    fps = numpy.zeros((num_thresholds, nb_classes), numpy.uint)
    fns = numpy.zeros((num_thresholds, nb_classes), numpy.uint)

    predictionorder = []
    for className in class_definitions.keys():
        predictionorder.append(class_definitions[className][0])
    indexes = numpy.argsort(predictionorder)
    classNames = [list(class_definitions)[i] for i in indexes]

    #cm = plt.get_cmap('jet')
    outputfile = os.path.join(filePath.get_result_path(label), outputprefix + "results_trainingthreshold_%s.%s.txt" % (stain, label))

    modelsegmentationpath = os.path.join(segmentationpath, label)

    with open(outputfile, 'w') as f:
        writer = csv.writer(f, delimiter='\t')

        writer.writerow(numpy.array(('patientNb', 'thrshold', 'F1', 'precision', 'recall', 'accuracy')))
        print("Image segmentation: %s " % os.path.join(modelsegmentationpath, list(class_definitions)[0]))
        print("Patients (%d): %s" % (len(patientlist), ', '.join(patientlist)))
        print("Stain: %s" % stain)

        for imagePath, imageName in filePath.get_images_with_list_patients(
                os.path.join(modelsegmentationpath, list(class_definitions)[0]), staincode=stain, patients=patientlist):

            print("File: %s" % imagePath)

            prediction = image_utils.read_segmentations(modelsegmentationpath, os.path.basename(imagePath), classNames)

            gtfilename = filePath.get_groundtruth(imageName, lod)
            gt = image_utils.read_image(gtfilename)

            labelsOutputPath = filePath.generate_groundtruthlabelspath(imageName, lod)
            with open(labelsOutputPath, 'r') as fp:
                classLabels = json.load(fp)

            gt = image_utils.relabel_gt(gt, classLabels, class_definitions)

            mask = numpy.any(prediction != 255, axis=-1)
            if not mask.any():
                mask = None

            cl_tps, cl_tns, cl_fps, cl_fns, bestP, bestR, bestF1, bestAcc, bestclps, bestclrs, bestclf1s, bestclaccs, seuil = threshold_evaluation(
                prediction,
                gt,
                min_threshold,
                max_threshold,
                num_thresholds,
                nb_classes,
                mask=mask,
                verbose=1)

            tps = numpy.add(cl_tps, tps)
            tns = numpy.add(cl_tns, tns)
            fps = numpy.add(cl_fps, fps)
            fns = numpy.add(cl_fns, fns)

            writer.writerow(['%s' % imageName, '%.3f' % seuil, '%.3f' % bestF1, '%.3f' % bestP, '%.3f' % bestR, '%.3f' % bestAcc])
            for className, cl_f1, cl_p, cl_r, cl_acc in zip(classNames, bestclf1s, bestclps, bestclrs, bestclaccs):
                writer.writerow(['\t', '%s' % className, '%.3f' % cl_f1, '%.3f' % cl_p, '%.3f' % cl_r, '%.3f' % cl_acc])

            prediction[prediction != prediction.max(axis=-1, keepdims=1)] = 0
            predictionSeuil = numpy.argmax(prediction >= seuil, axis=-1)
            image_utils.save_image(predictionSeuil.astype('uint8'),
                                   os.path.join(detectionpath, imageName + '.png'))

            #displayOutputPath = os.path.join(detectionpath, imageName + "_display.png")
            #image_utils.save_image(colored_result.astype(numpy.uint8), displayOutputPath)

        # Calculate global statistics
        ps, rs, f1s, accs = precision_recall_f1_accuracy(tps, tns, fps, fns)

        p = numpy.mean(ps[:, 1:], axis=1)
        r = numpy.mean(rs[:, 1:], axis=1)
        f1 = numpy.mean(f1s[:, 1:], axis=1)
        acc = numpy.mean(accs[:, 1:], axis=1)

        writer.writerow(['overall', '%.3f' % threshold, '%.3f' % f1, '%.3f' % p, '%.3f' % r, '%.3f' % acc])
        for className, cl_f1, cl_p, cl_r, cl_acc in zip(classNames, f1s[0].tolist(), ps[0].tolist(), rs[0].tolist(), accs[0].tolist()):
            writer.writerow(['\t', '%s' % className, '%.3f' % cl_f1, '%.3f' % cl_p, '%.3f' % cl_r, '%.3f' % cl_acc])

    return seuil


'''
Evaluate detector segmentations by taking the class with the maximum probability in the detector's output
'''
def evaluate_max_output(filePath, segmentationpath, label, class_definitions, detectionpath, patientlist, lod,
                        stain, outputprefix=""):

    modeldetectionpath = os.path.join(detectionpath, label, "maxoutput")
    if not os.path.exists(modeldetectionpath):
        os.makedirs(modeldetectionpath)

    if not os.path.exists(filePath.get_result_path(label)):
        os.makedirs(filePath.get_result_path(label))

    nb_classes = len(class_definitions)

    predictionorder = []
    for className in class_definitions.keys():
        predictionorder.append(class_definitions[className][0])
    indexes = numpy.argsort(predictionorder)
    classNames = [list(class_definitions)[i] for i in indexes]

    tps = numpy.zeros((nb_classes), numpy.uint)
    tns = numpy.zeros((nb_classes), numpy.uint)
    fps = numpy.zeros((nb_classes), numpy.uint)
    fns = numpy.zeros((nb_classes), numpy.uint)

    modelsegmentationpath = os.path.join(segmentationpath, label)

    #cm = plt.get_cmap('jet')
    with open(os.path.join(filePath.get_result_path(label), outputprefix + "results_maxoutput_%s.%s.txt" % (stain, label)), 'w') as f:
        writer = csv.writer(f, delimiter='\t')

        writer.writerow(numpy.array(('patientNb', 'F1', 'precision', 'recall', 'accuracy')))
        print("Image segmentation: %s " % os.path.join(modelsegmentationpath, list(class_definitions)[0]))
        print("Patients (%d): %s" % (len(patientlist), ', '.join(patientlist)))
        print("Stain: %s" % stain)

        for imagePath, imageName in filePath.get_images_with_list_patients(
                os.path.join(modelsegmentationpath, list(class_definitions)[0]), staincode=stain, patients=patientlist):
            print("File: %s" % imagePath)

            # If max output doesn't exist, calculate it
            if not os.path.isfile(os.path.join(modeldetectionpath, imageName + '.png')):
                from apply_unet import calculatemaxoutputsegmentations
                calculatemaxoutputsegmentations([imageName], segmentationpath, detectionpath, label, class_definitions)

            detection = image_utils.read_image(os.path.join(modeldetectionpath, imageName + '.png'))

            gtfilename = filePath.get_groundtruth(imageName, lod)
            gt = image_utils.read_image(gtfilename)

            labelsOutputPath = filePath.generate_groundtruthlabelspath(imageName, lod)
            with open(labelsOutputPath, 'r') as fp:
                classLabels = json.load(fp)

            gt = image_utils.relabel_gt(gt, classLabels, class_definitions)

            #mask = image_utils.read_image(filePath.generate_maskpath(imageName, 'tissue', config['general.lod']))
            #mask = numpy.logical_and(mask, detection != 255)
            mask = detection != 255
            if not mask.any():
                mask = None

            cl_tps, cl_tns, cl_fps, cl_fns, cl_ps, cl_rs, cl_f1s, cl_accs, p, r, f1, acc = evaluate_detection(gt,
                                                                                                              detection,
                                                                                                              nb_classes,
                                                                                                              mask=mask)
            tps = numpy.add(cl_tps, tps)
            tns = numpy.add(cl_tns, tns)
            fps = numpy.add(cl_fps, fps)
            fns = numpy.add(cl_fns, fns)

            writer.writerow(['%s' % imageName, '%.3f' % f1, '%.3f' % p, '%.3f' % r, '%.3f' % acc])
            for className, cl_f1, cl_p, cl_r, cl_acc in zip(classNames, cl_f1s, cl_ps, cl_rs, cl_accs):
                writer.writerow(['\t', '%s' % className, '%.3f' % cl_f1, '%.3f' % cl_p, '%.3f' % cl_r, '%.3f' % cl_acc])

        # Calculate global statistics
        ps, rs, f1s, accs = precision_recall_f1_accuracy(tps, tns, fps, fns)

        p = numpy.mean(ps[1:])
        r = numpy.mean(rs[1:])
        f1 = numpy.mean(f1s[1:])
        acc = numpy.mean(accs[1:])

        writer.writerow(['overall', '%.3f' % f1, '%.3f' % p, '%.3f' % r, '%.3f' % acc])
        for className, f1, p, r, acc in zip(classNames, f1s, ps, rs, accs):
            writer.writerow(['\t', '%s' % className, '%.3f' % f1, '%.3f' % p, '%.3f' % r, '%.3f' % acc])


'''
Evaluate detector segmentations using thresholds in the range 0 to 255, find the best overall threshold (which is 
returned).
'''
def evaluate_globalthreshold(filePath, segmentationpath, label, class_definitions, detectionpath, patientlist, lod, stain):

    detectionpath = os.path.join(detectionpath, label, "globalthreshold")
    if not os.path.exists(detectionpath):
        os.makedirs(detectionpath)

    if not os.path.exists(filePath.get_result_path(label)):
        os.makedirs(filePath.get_result_path(label))

    min_threshold = 0
    max_threshold = 255
    num_thresholds = 256

    nb_classes = len(class_definitions)

    tps = numpy.zeros((num_thresholds, nb_classes), numpy.uint)
    tns = numpy.zeros((num_thresholds, nb_classes), numpy.uint)
    fps = numpy.zeros((num_thresholds, nb_classes), numpy.uint)
    fns = numpy.zeros((num_thresholds, nb_classes), numpy.uint)

    predictionorder = []
    for className in class_definitions.keys():
        predictionorder.append(class_definitions[className][0])
    indexes = numpy.argsort(predictionorder)
    classNames = [list(class_definitions)[i] for i in indexes]

    modelsegmentationpath = os.path.join(segmentationpath, label)

    with open(os.path.join(filePath.get_result_path(label), "results_threshold_%s.%s.txt" % (stain, label)), 'w') as f:
        writer = csv.writer(f, delimiter='\t')

        writer.writerow(numpy.array(('bestSeuil', 'F1', 'precision', 'recall')))

        for imagePath, imageName in filePath.get_images_with_list_patients(
                os.path.join(modelsegmentationpath, list(class_definitions)[0]),
                staincode=stain, patients=patientlist):
            print("File: %s" % imagePath)

            prediction = image_utils.read_segmentations(modelsegmentationpath, os.path.basename(imagePath), classNames)

            gtfilename = filePath.get_groundtruth(imageName, lod)
            gt = image_utils.read_image(gtfilename)

            labelsOutputPath = filePath.generate_groundtruthlabelspath(imageName, lod)
            with open(labelsOutputPath, 'r') as fp:
                classLabels = json.load(fp)

            gt = image_utils.relabel_gt(gt, classLabels, class_definitions)

            mask = numpy.any(prediction != 255, axis=-1)
            if not mask.any():
                mask = None

            #  Compute performance statistics
            cl_tps, cl_tns, cl_fps, cl_fns, bestp, bestr, bestf1, bestacc, bestclps, bestclrs, bestclf1s, bestclaccs, seuil = threshold_evaluation(
                prediction,
                gt,
                min_threshold,
                max_threshold,
                num_thresholds,
                nb_classes,
                mask=mask,
                verbose=1)

            tps = numpy.add(cl_tps, tps)
            tns = numpy.add(cl_tps, tns)
            fps = numpy.add(cl_fps, fps)
            fns = numpy.add(cl_fns, fns)

        # Calculate global statistics
        ps, rs, f1s, accs = precision_recall_f1_accuracy(tps, tns, fps, fns)

        p = numpy.mean(ps[:, 1:], axis=1)
        r = numpy.mean(rs[:, 1:], axis=1)
        f1 = numpy.mean(f1s[:, 1:], axis=1)
        acc = numpy.mean(accs[:, 1:], axis=1)

        bestF1 = max(f1)
        seuils = numpy.linspace(min_threshold, max_threshold, num=num_thresholds)
        bestseuil_idx = numpy.where(f1s == bestF1)

        seuil = seuils[bestseuil_idx]

        bestP = p[bestseuil_idx]
        bestR = r[bestseuil_idx]
        bestAcc = acc[bestseuil_idx]

        writer.writerow(['overall', '%.3f' % seuil, '%.3f' % bestF1, '%.3f' % bestP, '%.3f' % bestR, '%.3f' % bestAcc])
        for className, cl_f1, cl_p, cl_r, cl_acc in zip(classNames, f1s[bestseuil_idx,:][0].tolist(), ps[bestseuil_idx, :][0].tolist(), rs[bestseuil_idx, :][0].tolist(), accs[bestseuil_idx, :][0].tolist()):
            writer.writerow(['\t', '%s' % className, '%.3f' % cl_f1, '%.3f' % cl_p, '%.3f' % cl_r, '%.3f' % cl_acc])

    # cm = plt.get_cmap('jet')
    for imagePath, imageName in filePath.get_images_with_list_patients(os.path.join(modelsegmentationpath, list(class_definitions)[0]),
                                                                       staincode=stain, patients=patientlist):
        prediction = image_utils.read_segmentations(modelsegmentationpath, os.path.basename(imagePath), class_definitions.keys(), predictionorder)

        prediction[prediction != prediction.max(axis=-1, keepdims=1)] = 0
        predictionSeuil = numpy.argmax(prediction >= seuil, axis=-1)

        image_utils.save_image(predictionSeuil.astype('uint8'), os.path.join(detectionpath, imageName + '.png'))

        # Causes memory error with large images
        #maskfilename = os.path.join(maskpath, imageName + "_mask_" + labelName + "_lod" + str(lod) + ".png")
        #mask = image_utils.read_binary_image(maskfilename)

        #colored_result = cm(predictionSeuil.astype(numpy.uint8) + (mask.astype(numpy.uint8) * 2))[:, :, :3] * 255

        #displayOutputPath = os.path.join(detectionpath, imageName + "_display.png")
        #image_utils.save_image(colored_result.astype(numpy.uint8), displayOutputPath)

    return seuil


def pretrained_csco_model_path(conf):
    return os.path.join(conf['general.homepath'], 'saved_models/SSL/Nephrectomy',
                        conf['transferlearning.pretrained_ssl_model'], conf['detector.segmentationmodel'],
                        conf['general.staincode'], "rep1/contrastive_learning/models/HO_encoder_model.best.hdf5")

def derived_parameters(conf, arguments):
    conf['detector.patchstrategy'] = arguments.patch_strategy
    conf['detector.percentN'] = arguments.percent_N

    if arguments.validation_data.lower() == "none":
        config['detector.validation_data'] = None
    else:
        config['detector.validation_data'] = arguments.validation_data
        
    if conf['detector.percentN'] == "percent_100":
        conf['detector.traininputpath'] = os.path.join(conf['detector.traininputpath'], 'patches')
    else:
        conf['detector.traininputpath'] = os.path.join(conf['detector.traininputpath'], 'separated_patches',
                                                       conf['detector.patchstrategy'], conf['detector.percentN'])

    if config['detector.validation_data'] is None:
        conf['detector.validationinputpath'] = None 
    elif conf['detector.validation_data'].lower() == "respective_splits":
        if conf['detector.percentN'] == "percent_100":
            conf['detector.validationinputpath'] = os.path.join(conf['detector.validationinputpath'], 'patches')
        else:
            conf['detector.validationinputpath'] = os.path.join(conf['detector.validationinputpath'], 'separated_patches',
                                                                conf['detector.patchstrategy'], conf['detector.percentN']) 
    elif conf['detector.validation_data'].lower() == "full":
        conf['detector.validationinputpath'] = os.path.join(conf['detector.validationinputpath'], 'patches')
    
     
    if conf['detector.validation_data'] is None:
        conf['detector.modelpath'] = os.path.join(conf['detector.modelpath'], conf['transferlearning.finetunemode'],
                                              conf['detector.segmentationmodel'], conf['detector.patchstrategy'],
                                              conf['detector.percentN'], conf['general.staincode'], 'without_validation_data')
    elif conf['detector.validation_data'].lower() == "respective_splits":
        conf['detector.modelpath'] = os.path.join(conf['detector.modelpath'], conf['transferlearning.finetunemode'],
                                              conf['detector.segmentationmodel'], conf['detector.patchstrategy'],
                                              conf['detector.percentN'], conf['general.staincode'], 'respective_splits_validation_data')
    elif conf['detector.validation_data'].lower() == "full":
        conf['detector.modelpath'] = os.path.join(conf['detector.modelpath'], conf['transferlearning.finetunemode'],
                                              conf['detector.segmentationmodel'], conf['detector.patchstrategy'],
                                              conf['detector.percentN'], conf['general.staincode'], 'full_validation_data')
    
    if conf['transferlearning.finetune']:
        conf['transferlearning.stain_separate'] = True
        conf['transferlearning.pretrained_ssl_model'] = arguments.pretrained_ssl_model.lower()
        conf['transferlearning.pretrained_model_trainable'] = arguments.pretrained_model_trainable
        
        if 'hrcsco' in conf['transferlearning.pretrained_ssl_model']:
            conf['transferlearning.csco_model_path'] = pretrained_csco_model_path(conf)
        else:
            raise ValueError("Self-supervised learning based pretrained-models should be one of ['hrcsco']")

        conf['detector.outputpath'] = os.path.join(conf['detector.modelpath'], conf['detector.modelname'],
                                                   conf['transferlearning.pretrained_ssl_model'])
    else:
        conf['detector.outputpath'] = os.path.join(conf['detector.modelpath'], conf['detector.modelname'])

    conf['segmentation.segmentationpath'] = os.path.join(conf['detector.outputpath'],
                                                         conf['segmentation.segmentationpath'])
    conf['segmentation.detectionpath'] = os.path.join(conf['detector.outputpath'],
                                                      conf['segmentation.detectionpath'])
    return conf


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test a UNet model.')

    parser.add_argument('-c', '--configfile', type=str, help='the configuration file to use')
    parser.add_argument('-l', '--label', type=str, help='the label of the model to be tested')
    parser.add_argument('-s', '--teststain', type=str, help='the stain to test upon')
    parser.add_argument('-m', '--maxoutput', action='store_const', default=False, const=True,
                        help='test maxoutput')
    parser.add_argument('-t', '--threshold', action='store_const', default=False, const=True,
                        help='test threshold levels')
    parser.add_argument('-g', '--gpu', type=str, help='specify which GPU to use')
   
   # Adding parameters to finetune the UNet with pretrained Self Supervised Learning Models (SimCLR, Byol, CSCO, etc)
    parser.add_argument('-pm', '--pretrained_ssl_model', type=str, default='CSCO')
    parser.add_argument('-pmt', '--pretrained_model_trainable', default=True, 
                        type=lambda x: str(x).lower() in ['true', '1', 'yes'])
    # Adding parameters to specify the data strategy
    parser.add_argument('-ps', '--patch_strategy', type=str, default='percentN_equally_randomly')
    parser.add_argument('-pn', '--percent_N', type=str, default='percent_1')
   
    parser.add_argument('-lr', '--LR', type=str, default="None")
    parser.add_argument('-lrd', '--LR_weightdecay', type=str, default="None")
    parser.add_argument('-rlrp', '--reduceLR_percentile', type=str, default="None")
    
    parser.add_argument('-vd', '--validation_data', type=str, default="None", help="none | respective_splits | full")
    
    args = parser.parse_args()

    args = parser.parse_args()

    # if args.gpu:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # else:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = str(pick_gpu_lowest_memory())
    #
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # for gpu in gpus:
    #     tf.config.experimental.set_memory_growth(gpu, True)
    # print("\nGpu Growth Restriction Done...")

    if not args.maxoutput and not args.threshold:
        args.maxoutput = True
        args.threshold = True

    if args.configfile:
        config = config_utils.readconfig(args.configfile)
    else:
        config = config_utils.readconfig()

    config = derived_parameters(config, arguments=args)

    if args.LR != 'None':
        config['detector.learn_rate'] = float(args.LR)

    if args.LR_weightdecay != 'None':
        config['detector.LR_weightdecay'] = float(args.LR_weightdecay)
    else:
        config['detector.LR_weightdecay'] = None
    
    if args.reduceLR_percentile != 'None' and config['transferlearning.finetune']:
        if config['detector.reducelr'] and config['transferlearning.pretrained_model_trainable']:
            config['detector.reduceLR_percentile'] = int(args.reduceLR_percentile)
            config['detector.reduceLR_patience'] = int((config['detector.reduceLR_percentile'] / 100) * config['detector.epochs'])

    if args.label:
        label = args.label

    filePath = filepath_utils.FilepathGenerator(config)

    if args.teststain:
        teststain = args.teststain
    else:
        teststain = '*'

    if args.maxoutput:
        # Evaluate maximum classifier output
        evaluate_max_output(filePath, config["segmentation.segmentationpath"], label,
                            config['extraction.class_definitions'], config["segmentation.detectionpath"],
                            config['general.testPatients'], config["detector.lod"], teststain)

    if args.threshold:
        # Evaluate range of thresholds
        evaluate_thresholds(filePath, config["segmentation.segmentationpath"], label,
                            config['extraction.class_definitions'], config["segmentation.detectionpath"],
                            config['general.testPatients'], config["detector.lod"], teststain)

    train_threshold_filename = os.path.join(config['detector.outputpath'], "models",
                                            "threshold_%s_%s.txt" % (config["general.staincode"], label))
    if os.path.isfile(train_threshold_filename):
        # Evaluate threshold found during training
        with open(train_threshold_filename, 'r') as f:
            trainingthreshold = float(f.read())

        evaluate_threshold(filePath, config["segmentation.segmentationpath"], label,
                           config['extraction.class_definitions'], config["segmentation.detectionpath"],
                           config['general.testPatients'], config["detector.lod"], teststain, trainingthreshold)
    else:
        warnings.warn("Training threshold filename not found, skipping evaluation")
