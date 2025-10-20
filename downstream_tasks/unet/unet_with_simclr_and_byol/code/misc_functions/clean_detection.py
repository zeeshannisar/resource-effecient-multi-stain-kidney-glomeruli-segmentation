from utils import image_utils
from skimage import morphology, measure
import numpy
import scipy
from os import path
import sys
from utils.evaluation_metrics import threshold_evaluation, evaluate_detection, precision_recall_f1_accuracy

stains = ['25', '106', '107', '108']

filtereccentricity = False

patient=int(sys.argv[1])
model=sys.argv[2]

print(patient)
print(model)

#model='base_valid_nostain_nolearn_mixed_p1_redo_3'
#patient = 1

#model='base_valid_nostain_nolearn_mixed_p2_redo_2'
#patient=2

if patient == 1:
	suffix = 'NX_III'
else:
	suffix='NX_NE4'


area_lim=30000
eccentricity_lim=0.8

nb_classes=2

with open('/home/lampert/model/Nephrectomies_25_106_107_108_p' + str(patient) + '_rgb/detections/' + model + '/maxoutput/clean_results.txt', "w+") as f:

	tps = numpy.zeros((nb_classes), numpy.uint)
	tns = numpy.zeros((nb_classes), numpy.uint)
	fps = numpy.zeros((nb_classes), numpy.uint)
	fns = numpy.zeros((nb_classes), numpy.uint)
	tps_orig = numpy.zeros((nb_classes), numpy.uint)
	tns_orig = numpy.zeros((nb_classes), numpy.uint)
	fps_orig = numpy.zeros((nb_classes), numpy.uint)
	fns_orig = numpy.zeros((nb_classes), numpy.uint)

	for stain in stains:
		x = image_utils.read_image('/home/lampert/model/Nephrectomies_25_106_107_108_p' + str(patient) + '_rgb/detections/' + model + '/maxoutput/IFTA_EXC_00' + str(patient) + '_' + suffix + '_' + stain + '.png')
		if patient == 2:
			m = image_utils.read_image('/home/lampert/data/Nephrectomies/' + stain + '/masks/roi2/IFTA_EXC_00' + str(patient) + '_' + suffix + '_' + stain + '_mask_roi2_lod1.png')
		else:
			m = image_utils.read_image('/home/lampert/data/Nephrectomies/' + stain + '/masks/tissue/IFTA_EXC_00' + str(patient) + '_' + suffix + '_' + stain + '_mask_tissue_lod1.png')

		x[m == 0] = 0
		x[x != 2] = 0
		x = x.astype(bool)

		x_orig = numpy.copy(x)

		morphology.binary_opening(x, out=x)

		morphology.remove_small_objects(x, min_size=area_lim, in_place=True)


		if filtereccentricity:
			s = scipy.ndimage.generate_binary_structure(2, 2)
			labelmask, _ = scipy.ndimage.label(x, structure=s)
			regions = measure.regionprops(labelmask, coordinates='xy')
			for region in regions:
				if region.eccentricity > eccentricity_lim: # or region.areaint < area_lim:
					x[labelmask == region.label] = False


		image_utils.save_binary_image(x.astype(bool), '/home/lampert/model/Nephrectomies_25_106_107_108_p' + str(patient) + '_rgb/detections/' + model + '/maxoutput/IFTA_EXC_00' + str(patient) + '_' + suffix + '_' + stain + '_cleaned.png')

		gt = image_utils.read_image('/home/lampert/data/Nephrectomies/' + stain + '/masks/glomeruli/IFTA_EXC_00' + str(patient) + '_' + suffix + '_' + stain + '_mask_glomeruli_lod1.png')
		if patient == 2:
			gt[m == 0] = 0

		image_utils.save_binary_image(gt.astype(bool), '/home/lampert/IFTA_EXC_00' + str(patient) + '_' + suffix + '_' + stain + '_maskedglomeruli_lod1.png')

#		gt = image_utils.read_image('/home/lampert/data/Nephrectomies/' + stain + '/masks/healthy/IFTA_EXC_00' + str(patient) + '_' + suffix + '_' + stain + '_mask_healthy_lod1.png').astype(bool)
#		if path.exists('/home/lampert/data/Nephrectomies/' + stain + '/masks/sclerotic/IFTA_EXC_00' + str(patient) + '_' + suffix + '_' + stain + '_mask_sclerotic_lod1.png'):
#		print('sclerotic')
#		gt = numpy.logical_or(gt, image_utils.read_image('/home/lampert/data/Nephrectomies/' + stain + '/masks/sclerotic/IFTA_EXC_00' + str(patient) + '_' + suffix + '_' + stain + '_mask_sclerotic_lod1.png').astype(bool))
		from utils.evaluation_metrics import evaluate_detection, tp_tn_fp_fn
		#tp, tn, fp, fn = tp_tn_fp_fn(m, gt, 1, mask=m)
		#cl_tp, cl_tn, cl_fp, cl_fn, cl_p, cl_r, cl_f1, cl_acc, p, r, f1, acc = evaluate_detection(gt, x.astype('uint8'), 1, mask=m)

		s = scipy.ndimage.generate_binary_structure(2, 2)
		labelmask, num_regions = scipy.ndimage.label(gt, structure=s)

		print('stain:\t' + stain)
		print('glomeruli:\t' + str(num_regions))

		tp = numpy.sum(numpy.logical_and(x > 0, gt > 0))
		tn = numpy.sum(numpy.logical_and(x == 0, gt == 0))
		fp = numpy.sum(numpy.logical_and(x > 0, gt == 0))
		fn = numpy.sum(numpy.logical_and(x == 0, gt > 0))
		tp = tp.astype(numpy.float)
		tn = tn.astype(numpy.float)
		fp = fp.astype(numpy.float)
		fn = fn.astype(numpy.float)
		precision_after = tp / ((tp + fp) + 0.0000001)
		recall_after = tp / ((tp + fn) + 0.0000001)
		f1_after = (2 * (precision_after * recall_after)) / ((recall_after + precision_after) + 0.0000001)
		accuracy_after = (tp + tn) / (tp + tn + fp + fn)

		tp = numpy.sum(numpy.logical_and(x_orig > 0, gt > 0))
		tn = numpy.sum(numpy.logical_and(x_orig == 0, gt == 0))
		fp = numpy.sum(numpy.logical_and(x_orig > 0, gt == 0))
		fn = numpy.sum(numpy.logical_and(x_orig == 0, gt > 0))
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

		f.write('IFTA_EXC_00%s_%s_%s\n' % (str(patient), suffix, stain))
		f.write('glomeruli:\t%s\n' % str(num_regions))
		f.write('before:\t%s, %s, %s\n' % (str(f1_before), str(precision_before), str(recall_before)))
		f.write('after:\t%s, %s, %s\n\n' % (str(f1_after), str(precision_after), str(recall_after)))

		#####
		cl_tps, cl_tns, cl_fps, cl_fns, cl_ps, cl_rs, cl_f1s, cl_accs, p, r, f1, acc = evaluate_detection((gt>0).astype('uint8'),
                                                                                                              (x>0).astype('uint8'),
                                                                                                              nb_classes,
                                                                                                              mask=m>0)
		tps = numpy.add(cl_tps, tps)
		tns = numpy.add(cl_tns, tns)
		fps = numpy.add(cl_fps, fps)
		fns = numpy.add(cl_fns, fns)

		cl_tps_orig, cl_tns_orig, cl_fps_orig, cl_fns_orig, cl_ps_orig, cl_rs_orig, cl_f1s_orig, cl_accs_orig, p_orig, r_orig, f1_orig, acc_orig = evaluate_detection((gt>0).astype('uint8'),
                                                                                                              (x_orig>0).astype('uint8'),
                                                                                                              nb_classes,
                                                                                                              mask=m>0)
		tps_orig = numpy.add(cl_tps_orig, tps_orig)
		tns_orig = numpy.add(cl_tns_orig, tns_orig)
		fps_orig = numpy.add(cl_fps_orig, fps_orig)
		fns_orig = numpy.add(cl_fns_orig, fns_orig)
		#####

	ps, rs, f1s, accs = precision_recall_f1_accuracy(tps, tns, fps, fns)
	p = numpy.mean(ps[1:])
	r = numpy.mean(rs[1:])
	f1 = numpy.mean(f1s[1:])
	acc = numpy.mean(accs[1:])
	f.write('overall %.3f %.3f %.3f %.3f\n' % (f1, p, r, acc))
	classNames=['negative','glomeruli']
	for className, f1, p, r, acc in zip(classNames, f1s, ps, rs, accs):
		f.write('\t %s %.3f %.3f %.3f %.3f\n' % (className, f1, p, r, acc))

	ps_orig, rs_orig, f1s_orig, accs_orig = precision_recall_f1_accuracy(tps_orig, tns_orig, fps_orig, fns_orig)
	p_orig = numpy.mean(ps_orig[1:])
	r_orig = numpy.mean(rs_orig[1:])
	f1_orig = numpy.mean(f1s_orig[1:])
	acc_orig = numpy.mean(accs_orig[1:])
	f.write('overall %.3f %.3f %.3f %.3f\n' % (f1_orig, p_orig, r_orig, acc_orig))
	classNames=['negative','glomeruli']
	for className, f1, p, r, acc in zip(classNames, f1s_orig, ps_orig, rs_orig, accs_orig):
		f.write('\t %s %.3f %.3f %.3f %.3f\n' % (className, f1_orig, p_orig, r_orig, acc_orig))

	print('before:\t' + str(f1s_orig[1]) + ', ' + str(ps_orig[1]) + ', ' + str(rs_orig[1]))
	print('after:\t' + str(f1s[1]) + ', ' + str(ps[1]) + ', ' + str(rs[1]))
