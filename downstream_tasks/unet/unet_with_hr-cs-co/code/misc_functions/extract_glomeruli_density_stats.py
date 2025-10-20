import os
import csv
import argparse
import numpy
from utils import image_utils, config_utils
import skimage.draw
import skimage.segmentation
import shutil

# python3 extract_glomeruli_density_stats.py 1 <network_time_stamp> -c sysmifta_Nx_02_03_16_32_39_new.cfg

# GLOBAL VARIABLES
file_prefix = 'IFTA_EXC_' # before patient number
file1_postfix = '_NX_III' # after patient number

glomeruli_fibrosis_staining = '_10_105'
fibrosis_staining = '_10_105'

glomeruli_macrophage_staining = '_10_106'
macrophage_staining = '_10_106'
tcell_staining = '_10_106'

glomerulus_diameter = 150   # +/-10%
roi_rad = 275.5             # in pixels, must end in .5

number_of_segments = 20                 # used for slic
segment_compactness = 0.0001            # used for slic
enforce_segment_connectivity = False    # used for slic


def extract_glomeruli_bb(patient_number, glomeruliDetectionPath, glomeruli_class_index, staining):

    if (roi_rad * 2) % 2 == 0:
        raise ValueError("ROI radius x 2 should be odd, i.e.\ roi_rad = 10 -> roi_rad = 10.5" % labelName)

    patient_filename = file_prefix + patient_number.zfill(3) + file1_postfix + staining + '.png'

    glomeruli_detection = image_utils.read_image(os.path.join(glomeruliDetectionPath, patient_filename))
    glomeruli_detection = glomeruli_detection.copy()
    glomeruli_detection[glomeruli_detection != glomeruli_class_index] = 0

    labelmask, nbLabels = skimage.measure.label(glomeruli_detection, background=0, return_num=True)

    glomeruli = {}
    glomeruli['coords'] = []
    glomeruli['areas'] = []
    glomeruli['centroid'] = []
    for region in skimage.measure.regionprops(labelmask):
        # take regions with large enough areas
        if region.bbox[2] - region.bbox[0] < int(roi_rad * 2) \
                and region.bbox[3] - region.bbox[1] < int(roi_rad * 2) \
                and glomerulus_diameter * 0.9 <= region.equivalent_diameter <= glomerulus_diameter * 1.1:

            #and region. ...:
            glomeruli['centroid'].append([int(c) for c in region.centroid])
            glomeruli['coords'].append(region.coords)
            glomeruli['areas'].append(region.area)


    return glomeruli


def _rescale(image):
    image = image.astype(numpy.float)
    return (image * (255 / (numpy.amax(image)+0.00001))).astype(numpy.uint8)


def extract_glomeruli_macrophage_tcell_stats(patient_number, glomeruli, densityPath, lod):

    if (roi_rad * 2) % 2 == 0:
        raise ValueError("ROI radius x 2 should be odd, i.e.\ roi_rad = 10 -> roi_rad = 10.5" % labelName)

    patient_filename = file_prefix + patient_number.zfill(3) + file1_postfix

    macrophage_filename = patient_filename + macrophage_staining + '_macroDetection_lod' + str(lod) + '_densityMap' + '.npy'
    tcell_filename = patient_filename + tcell_staining + '_tCellsDetection_lod' + str(lod) + '_densityMap' + '.npy'

    macrophage_density = numpy.load(os.path.join(densityPath, 'macrophage', macrophage_filename))
    tcell_density = numpy.load(os.path.join(densityPath, 'tcell', tcell_filename))

    rr, cc = skimage.draw.circle(int(roi_rad), int(roi_rad), int(roi_rad))
    circle_coords = numpy.concatenate((rr[..., None], cc[..., None]), axis=1)
    circle_mask = numpy.zeros((int(roi_rad * 2), int(roi_rad * 2)), dtype=numpy.uint8)
    circle_mask[circle_coords[:, 0]-1, circle_coords[:, 1]-1] = 1

    idx = 0
    with open(os.path.join(outputpath, patient_filename + "_glomeruli_macrophage_tcell_stats.txt"), 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(numpy.array(('index', 'centre', 'area', 'macrophage_out', 'tcell_out', 'macrophage_in', 'tcell_in')))

        for centroid, glomeruli_coords, area in zip(glomeruli['centroid'], glomeruli['coords'], glomeruli['areas']):
            if int(centroid[0]) >= int(roi_rad) \
                    and int(centroid[1]) >= int(roi_rad) \
                    and int(centroid[0]) + int(roi_rad) < macrophage_density.shape[0] \
                    and int(centroid[1]) + int(roi_rad) < macrophage_density.shape[1]:

                start_y = int(centroid[0]) - int(roi_rad)
                end_y = int(centroid[0]) + int(roi_rad)+1
                start_x = int(centroid[1]) - int(roi_rad)
                end_x = int(centroid[1]) + int(roi_rad)+1

                patch_glomeruli_coords = glomeruli_coords
                patch_glomeruli_coords[:, 0] -= start_y
                patch_glomeruli_coords[:, 1] -= start_x

                glomerulus_mask = numpy.zeros((int(roi_rad * 2), int(roi_rad * 2)), dtype=numpy.uint8)
                glomerulus_mask[patch_glomeruli_coords[:, 0], patch_glomeruli_coords[:, 1]] = 1

                m_patch = macrophage_density[start_y:end_y, start_x:end_x].copy()
                t_patch = tcell_density[start_y:end_y, start_x:end_x].copy()

                circle_not_glomerulus_mask = numpy.logical_and(circle_mask, numpy.logical_not(glomerulus_mask))

                glomeruli_macrophage_density_outside = numpy.mean(m_patch[circle_not_glomerulus_mask])
                glomeruli_tcell_density_outside = numpy.mean(t_patch[circle_not_glomerulus_mask])

                glomeruli_macrophage_density_inside = numpy.mean(m_patch[glomerulus_mask])
                glomeruli_tcell_density_inside = numpy.mean(t_patch[glomerulus_mask])

                # Write stats to CSV
                writer.writerow(['%i' % idx,
                                 ','.join(map(str, centroid[::-1])),
                                 '%i' % area,
                                 '%.3f' % glomeruli_macrophage_density_outside,
                                 '%.3f' % glomeruli_tcell_density_outside,
                                 '%.3f' % glomeruli_macrophage_density_inside,
                                 '%.3f' % glomeruli_tcell_density_inside])

                # Write patch images
                image_utils.save_image(_rescale(circle_not_glomerulus_mask), os.path.join(outputpath, patient_filename + '_mask_out_' + str(idx) + '.png'))
                image_utils.save_image(_rescale(glomerulus_mask), os.path.join(outputpath, patient_filename + '_mask_in_' + str(idx) + '.png'))

                # Outside Glomeruli Images
                m_patch_outside = m_patch.copy()
                m_patch_outside[numpy.logical_not(circle_not_glomerulus_mask)] = 0
                image_utils.save_image(_rescale(m_patch_outside), os.path.join(outputpath, patient_filename + '_macrophage_out_' + str(idx) + '.png'))

                m_patch_outside = m_patch.copy()
                m_patch_outside[numpy.logical_not(circle_not_glomerulus_mask)] = 0
                m_patch_segment = skimage.segmentation.slic(m_patch_outside, n_segments=number_of_segments, compactness=segment_compactness, enforce_connectivity=enforce_segment_connectivity) + 1
                m_patch_segment[numpy.logical_not(circle_not_glomerulus_mask)] = 0
                m_patch_segment = _relabel_segments_with_means_values(m_patch_segment, m_patch, backgroundvalue=0)
                m_patch_segment[numpy.logical_not(circle_not_glomerulus_mask)] = 0
                image_utils.save_image(_rescale(m_patch_segment), os.path.join(outputpath, patient_filename + '_macrophage_out_seg_' + str(idx) + '.png'))

                t_patch_outside = t_patch.copy()
                t_patch_outside[numpy.logical_not(circle_not_glomerulus_mask)] = 0
                image_utils.save_image(_rescale(t_patch_outside), os.path.join(outputpath, patient_filename + '_tcell_out_' + str(idx) + '.png'))

                t_patch_outside = t_patch.copy()
                t_patch_outside[numpy.logical_not(circle_not_glomerulus_mask)] = 0
                t_patch_segment = skimage.segmentation.slic(t_patch_outside, n_segments=number_of_segments, compactness=segment_compactness, enforce_connectivity=enforce_segment_connectivity) + 1
                t_patch_segment[numpy.logical_not(circle_not_glomerulus_mask)] = 0
                t_patch_segment = _relabel_segments_with_means_values(t_patch_segment, t_patch, backgroundvalue=0)
                t_patch_segment[numpy.logical_not(circle_not_glomerulus_mask)] = 0
                image_utils.save_image(_rescale(t_patch_segment).astype(numpy.uint8), os.path.join(outputpath, patient_filename + '_tcell_out_seg_' + str(idx) + '.png'))

                # Inside Glomeruli Images
                m_patch_inside = m_patch.copy()
                m_patch_inside[numpy.logical_not(glomerulus_mask)] = 0
                image_utils.save_image(_rescale(m_patch_inside), os.path.join(outputpath, patient_filename + '_macrophage_in_' + str(idx) + '.png'))

                m_patch_inside = m_patch.copy()
                m_patch_inside[numpy.logical_not(glomerulus_mask)] = 0
                m_patch_segment = skimage.segmentation.slic(m_patch_inside, n_segments=number_of_segments, compactness=segment_compactness, enforce_connectivity=enforce_segment_connectivity)+1
                m_patch_segment[numpy.logical_not(glomerulus_mask)] = 0
                m_patch_segment = _relabel_segments_with_means_values(m_patch_segment, m_patch, backgroundvalue=0)
                m_patch_segment[numpy.logical_not(glomerulus_mask)] = 0
                image_utils.save_image(_rescale(m_patch_segment), os.path.join(outputpath, patient_filename + '_macrophage_in_seg_' + str(idx) + '.png'))

                t_patch_inside = t_patch.copy()
                t_patch_inside[numpy.logical_not(glomerulus_mask)] = 0
                image_utils.save_image(_rescale(t_patch_inside), os.path.join(outputpath, patient_filename + '_tcell_in_' + str(idx) + '.png'))

                t_patch_inside = t_patch.copy()
                t_patch_inside[numpy.logical_not(glomerulus_mask)] = 0
                t_patch_segment = skimage.segmentation.slic(t_patch_inside, n_segments=number_of_segments, compactness=segment_compactness, enforce_connectivity=enforce_segment_connectivity) + 1
                t_patch_segment[numpy.logical_not(glomerulus_mask)] = 0
                t_patch_segment = _relabel_segments_with_means_values(t_patch_segment, t_patch, backgroundvalue=0)
                t_patch_segment[numpy.logical_not(glomerulus_mask)] = 0
                image_utils.save_image(_rescale(t_patch_segment), os.path.join(outputpath, patient_filename + '_tcell_in_seg_' + str(idx) + '.png'))

                idx += 1


def _relabel_segments_with_means_values(segmentation, image, backgroundvalue=None):

    inds = numpy.unique(segmentation)
    if backgroundvalue:
        inds = list(set(inds) - set([backgroundvalue]))

    mean_segmentation = numpy.zeros_like(segmentation, dtype=numpy.float)
    for ind in inds:
        mean_segmentation[segmentation == ind] = numpy.mean(image[segmentation == ind])

    return mean_segmentation


def extract_glomeruli_fibrosis_stats(patient_number, glomeruli, densityPath, lod):

    if (roi_rad * 2) % 2 == 0:
        raise ValueError("ROI radius x 2 should be odd, i.e.\ roi_rad = 10 -> roi_rad = 10.5" % labelName)

    patient_filename = file_prefix + patient_number.zfill(3) + file1_postfix

    fibrosis_filename = patient_filename + fibrosis_staining + '_fapDetection_lod' + str(lod) + '_densityMap' + '.npy'

    fibrosis_density = numpy.load(os.path.join(densityPath, 'fibrosis', fibrosis_filename))

    rr, cc = skimage.draw.circle(roi_rad, roi_rad, roi_rad)
    circle_coords = numpy.concatenate((rr[..., None], cc[..., None]), axis=1)
    circle_mask = numpy.zeros((int(roi_rad * 2), int(roi_rad * 2)), dtype=numpy.uint8)
    circle_mask[circle_coords[:, 0], circle_coords[:, 1]] = 1

    idx = 0
    with open(os.path.join(outputpath, patient_filename + "_glomeruli_fibrosis_stats.txt"), 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(numpy.array(('index', 'centre', 'area', 'fibrosis_out', 'fibrosis_in')))

        for centroid, glomeruli_coords, area in zip(glomeruli['centroid'], glomeruli['coords'], glomeruli['areas']):
            if int(centroid[0]) >= int(roi_rad) \
                    and int(centroid[1]) >= int(roi_rad) \
                    and int(centroid[0]) + int(roi_rad) < fibrosis_density.shape[0] \
                    and int(centroid[1]) + int(roi_rad) < fibrosis_density.shape[1]:

                start_y = int(centroid[0]) - int(roi_rad)
                end_y = int(centroid[0]) + int(roi_rad) + 1
                start_x = int(centroid[1]) - int(roi_rad)
                end_x = int(centroid[1]) + int(roi_rad) + 1

                patch_glomeruli_coords = glomeruli_coords
                patch_glomeruli_coords[:, 0] -= start_y
                patch_glomeruli_coords[:, 1] -= start_x

                glomerulus_mask = numpy.zeros((int(roi_rad * 2), int(roi_rad * 2)), dtype=numpy.uint8)
                glomerulus_mask[patch_glomeruli_coords[:, 0], patch_glomeruli_coords[:, 1]] = 1

                f_patch = fibrosis_density[start_y:end_y, start_x:end_x].copy()

                circle_not_glomerulus_mask = numpy.logical_and(circle_mask, numpy.logical_not(glomerulus_mask))

                glomeruli_fibrosis_density_outside = numpy.mean(f_patch[circle_not_glomerulus_mask])

                glomeruli_fibrosis_density_inside = numpy.mean(f_patch[glomerulus_mask])

                # Write stats to CSV
                writer.writerow(['%i' % idx,
                                 ','.join(map(str, centroid[::-1])),
                                 '%i' % area,
                                 '%.3f' % glomeruli_fibrosis_density_outside,
                                 '%.3f' % glomeruli_fibrosis_density_inside])

                # Write patch images

                #image_utils.save_image(circle_not_glomerulus_mask.astype(numpy.uint8) * 255, os.path.join(outputpath,
                #                                                                         patient_filename + '_mask_out_' + str(
                #                                                                             idx) + '.png'))

                # Outside Glomeruli Images
                f_patch_outside = f_patch.copy()
                f_patch_outside[numpy.logical_not(circle_not_glomerulus_mask)] = 0
                image_utils.save_image(_rescale(f_patch_outside), os.path.join(outputpath, patient_filename + '_fibrosis_out_' + str(idx) + '.png'))

                f_patch_outside = f_patch.copy()
                f_patch_outside[numpy.logical_not(circle_not_glomerulus_mask)] = 0
                f_patch_segment = skimage.segmentation.slic(f_patch_outside, n_segments=number_of_segments, compactness=segment_compactness, enforce_connectivity=enforce_segment_connectivity) + 1
                f_patch_segment[numpy.logical_not(circle_not_glomerulus_mask)] = 0
                f_patch_segment = _relabel_segments_with_means_values(f_patch_segment, m_patch, backgroundvalue=0)
                f_patch_segment[numpy.logical_not(circle_not_glomerulus_mask)] = 0
                image_utils.save_image(_rescale(f_patch_segment), os.path.join(outputpath, patient_filename + '_fibrosis_out_seg_' + str(idx) + '.png'))

                # Inside Glomeruli Images
                f_patch_inside = f_patch.copy()
                f_patch_inside[numpy.logical_not(glomerulus_mask)] = 0
                image_utils.save_image(_rescale(f_patch_inside), os.path.join(outputpath, patient_filename + '_fibrosis_in_' + str(idx) + '.png'))

                f_patch_inside = f_patch.copy()
                f_patch_inside[numpy.logical_not(glomerulus_mask)] = 0
                f_patch_segment = skimage.segmentation.slic(f_patch_inside, n_segments=number_of_segments, compactness=segment_compactness, enforce_connectivity=enforce_segment_connectivity) + 1
                f_patch_segment[numpy.logical_not(glomerulus_mask)] = 0
                f_patch_segment = _relabel_segments_with_means_values(f_patch_segment, f_patch, backgroundvalue=0)
                f_patch_segment[numpy.logical_not(glomerulus_mask)] = 0
                image_utils.save_image(_rescale(f_patch_segment), os.path.join(outputpath, patient_filename + '_fibrosis_in_seg_' + str(idx) + '.png'))

                idx += 1


if __name__ == '__main__':

    ''' Generate the background masks for all images of staining _{staincode} in a given directory'''

    parser = argparse.ArgumentParser(description='Extract glomeruli-cell density statistics.')

    parser.add_argument('patient', type=str, help='the patient to be analysed')
    parser.add_argument('label', type=str, help='the label of the model\'s output to be used')
    parser.add_argument('-c', '--configfile', type=str, help='the configuration file to use')
    args = parser.parse_args()

    if args.configfile:
        config = config_utils.readconfig(args.configfile)
    else:
        config = config_utils.readconfig()
    config = config_utils.readconfig(os.path.join(config['detector.outputpath'], 'sysmifta.' + args.label + '.cfg'))

    densityPath = os.path.join(config['general.datapath'], 'densities')

    glomeruliDetectionPath = os.path.join(config['segmentation.detectionpath'], args.label, 'maxoutput')

    glomeruli_class_index = config['extraction.class_definitions']['glomeruli'][0]

    outputpath = os.path.join(config['detector.outputpath'], 'object_stats', 'glomeruli')
    print('Storing results in %s' % outputpath)
    shutil.rmtree(outputpath, ignore_errors=True)
    if not (os.path.exists(outputpath)):
        os.makedirs(outputpath)

    glomeruli = extract_glomeruli_bb(args.patient, glomeruliDetectionPath, glomeruli_class_index, glomeruli_macrophage_staining)
    extract_glomeruli_macrophage_tcell_stats(args.patient, glomeruli, densityPath, config['detector.lod'])

    glomeruli = extract_glomeruli_bb(args.patient, glomeruliDetectionPath, glomeruli_class_index, glomeruli_fibrosis_staining)
    extract_glomeruli_fibrosis_stats(args.patient, glomeruli, densityPath, config['detector.lod'])
