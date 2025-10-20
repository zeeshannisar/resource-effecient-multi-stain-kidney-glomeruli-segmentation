from utils import image_utils, filepath_utils, config_utils
from skimage import morphology, measure
import numpy
import scipy
from os import path
import math
import os, shutil
from keras.models import load_model
import argparse

stains = ['25', '106', '107', '108']
#stains = ['25', '107', '108']

filtereccentricity = False

model='base_valid_nostain_1'
#model='base_valid_nostain_nolearn_mixed_p4_2'
#model='base_valid_nostain_nolearn_mixed_p4_new_3'
model='base_valid_nostain_nolearn_mixed_p2_1'
#model='base_valid_nostain_nolearn_nostd_mixed_p4_new_1'


parser = argparse.ArgumentParser(description='Test a UNet model (default behaviour is to read the SVS images defined by the test patients in the configuration file \'sysmifta.cfg\').')

parser.add_argument('-g', '--gpu', type=str, help='specify which GPU to use')
args = parser.parse_args()

if args.gpu:
	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
else:
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"

patient = 1
patient = 4
patient=2

#sffix = 'NX_NE4'
suffix = 'NX_III'
suffix='NX_NE3'
suffix='NX_NE4'

windows_size = 512
cut_width=math.floor(windows_size/2)

config = config_utils.readconfig(os.path.join('/home/lampert/model/Nephrectomies_25_106_107_108_p' + str(patient) + '_rgb', 'sysmifta.' + model + '.cfg'))
filePath = filepath_utils.FilepathGenerator(config)

gan_cleaner = load_model('../gan_clean_models/gan_model_ep6.h5')

for stain in stains:
	x = image_utils.read_image('/home/lampert/model/Nephrectomies_25_106_107_108_p' + str(patient) + '_rgb/detections/' + model + '/maxoutput/IFTA_EXC_00' + str(patient) + '_' + suffix + '_' + stain + '.png')
	m = image_utils.read_image('/home/lampert/data/Nephrectomies/' + stain + '/masks/tissue/IFTA_EXC_00' + str(patient) + '_' + suffix + '_' + stain + '_mask_tissue_lod1.png')

	imageName = 'IFTA_EXC_00' + str(patient) + '_' + suffix + '_' + stain
	gtfilename = filePath.get_groundtruth(imageName, 1)
	gt = image_utils.read_image(gtfilename)

	x[m == 0] = 0
	x[x != 2] = 0
	x = x.astype(bool)

	gt[gt != 2] = 0
	gt = gt.astype(bool)

	s = scipy.ndimage.generate_binary_structure(2, 2)
	labelmask, _ = scipy.ndimage.label(x, structure=s)
	regions = measure.regionprops(labelmask, coordinates='xy')

	cleaned = numpy.zeros(x.shape)
	for region in regions:

		x0 = int(round(region.centroid[0]))
		y0 = int(round(region.centroid[1]))

		x0 = int(round((region.bbox[0] + region.bbox[2]) / 2))
		y0 = int(round((region.bbox[1] + region.bbox[3]) / 2))

		filename = str(x0) + '_' + str(y0) #+ '.png'

		patch = x[x0 - cut_width:x0 + cut_width, y0 - cut_width:y0 + cut_width]

		fake_A = numpy.squeeze(gan_cleaner.layers[1].predict_on_batch(patch[None, :, :, None]))

		cleaned[x0 - cut_width:x0 + cut_width, y0 - cut_width:y0 + cut_width] += numpy.squeeze(fake_A)

	image_utils.save_image(cleaned, '/home/lampert/model/Nephrectomies_25_106_107_108_p' + str(patient) + '_rgb/detections/' + model + '/maxoutput/IFTA_EXC_00' + str(patient) + '_' + suffix + '_' + stain + '-gan_cleaned.png')

	gt = image_utils.read_image('/home/lampert/data/Nephrectomies/' + stain + '/masks/glomeruli/IFTA_EXC_00' + str(patient) + '_' + suffix + '_' + stain + '_mask_glomeruli_lod1.png')
#               print('sclerotic')
#               gt = numpy.logical_or(gt, image_utils.read_image('/home/lampert/data/Nephrectomies/' + stain + '/masks/sclerotic/IFTA_EXC_00' + str(patient) + '_' + suffix + '_' + stain + '_mask_scleroti$
	from utils.evaluation_metrics import evaluate_detection, tp_tn_fp_fn
	#tp, tn, fp, fn = tp_tn_fp_fn(m, gt, 1, mask=m)
	#cl_tp, cl_tn, cl_fp, cl_fn, cl_p, cl_r, cl_f1, cl_acc, p, r, f1, acc = evaluate_detection(gt, x.astype('uint8'), 1, mask=m)

	s = scipy.ndimage.generate_binary_structure(2, 2)
	labelmask, num_regions = scipy.ndimage.label(gt, structure=s)
	print('stain:\t' + stain)
	print('glomeruli:\t' + str(num_regions))

	tp = numpy.sum(numpy.logical_and(cleaned > 0, gt > 0))
	tn = numpy.sum(numpy.logical_and(cleaned == 0, gt == 0))
	fp = numpy.sum(numpy.logical_and(cleaned > 0, gt == 0))
	fn = numpy.sum(numpy.logical_and(cleaned == 0, gt > 0))
	tp = tp.astype(numpy.float)
	tn = tn.astype(numpy.float)
	fp = fp.astype(numpy.float)
	fn = fn.astype(numpy.float)
	precision_after = tp / ((tp + fp) + 0.0000001)
	recall_after = tp / ((tp + fn) + 0.0000001)
	f1_after = (2 * (precision_after * recall_after)) / ((recall_after + precision_after) + 0.0000001)
	accuracy_after = (tp + tn) / (tp + tn + fp + fn)

	tp = numpy.sum(numpy.logical_and(x > 0, gt > 0))
	tn = numpy.sum(numpy.logical_and(x == 0, gt == 0))
	fp = numpy.sum(numpy.logical_and(x > 0, gt == 0))
	fn = numpy.sum(numpy.logical_and(x == 0, gt > 0))
	tp = tp.astype(numpy.float)
	tn = tn.astype(numpy.float)
	fp = fp.astype(numpy.float)
	fn = fn.astype(numpy.float)
	precision_before = tp / ((tp + fp) + 0.0000001)
	recall_before = tp / ((tp + fn) + 0.0000001)
	f1_before = (2 * (precision_before * recall_before)) / ((recall_before + precision_before) + 0.0000001)
	accuracy_before = (tp + tn) / (tp + tn + fp + fn)

	print('before:\t' + str(f1_before) + ', ' + str(precision_before) + ', ' + str(recall_before))
	print('after:\t' + str(f1_after) + ', ' + str(precision_after) + ', ' + str(recall_after))
