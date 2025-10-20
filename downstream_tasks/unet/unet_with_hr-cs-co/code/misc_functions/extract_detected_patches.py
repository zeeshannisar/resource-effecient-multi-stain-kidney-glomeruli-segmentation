from utils import image_utils, filepath_utils, config_utils
from skimage import morphology, measure
import numpy
import scipy
from os import path
import math
import os, shutil

stains = ['25', '106', '107', '108']
#stains = ['25', '107', '108']
stains=['25']

filtereccentricity = False

model='base_valid_nostain_1'
model='base_valid_nostain_nolearn_mixed_p4_2'
model='base_valid_nostain_nolearn_mixed_p1_1epoch_1'
#model='base_valid_nostain_nolearn_mixed_p4_new_3'
#model='base_valid_nostain_nolearn_mixed_p2_1'
#model='base_valid_nostain_nolearn_nostd_mixed_p4_new_1'

patient = 1
#patient = 4
#patient=2

#sffix = 'NX_NE4'
#suffix = 'NX_III'
suffix='NX_NE3'
suffix='NX_NE4'

windows_size = 512
cut_width=math.floor(windows_size/2)

config = config_utils.readconfig(os.path.join('/home/lampert/model/Nephrectomies_25_106_107_108_p' + str(patient) + '_rgb', 'sysmifta.' + model + '.cfg'))
filePath = filepath_utils.FilepathGenerator(config)

patient = 2

output_base = '/home/lampert/disk1/detection_patches/'

if os.path.isdir(output_base):
	shutil.rmtree(output_base)
os.mkdir(output_base)
os.mkdir(os.path.join(output_base, 'gt'))
os.mkdir(os.path.join(output_base, 'image'))
os.mkdir(os.path.join(output_base, 'probabilities'))

for stain in stains:
	svsImage = image_utils.open_svs_image_forced('/home/lampert/data/Nephrectomies/' + stain + '/images/IFTA_EXC_00' + str(patient) + '_' + suffix + '_' + stain + '.svs')

	x = image_utils.read_image('/home/lampert/model/Nephrectomies_25_106_107_108_p' + '1' + '_rgb/detections/' + model + '/maxoutput/IFTA_EXC_00' + str(patient) + '_' + suffix + '_' + stain + '.png')
	x2 = image_utils.read_image('/home/lampert/model/Nephrectomies_25_106_107_108_p' + '1' + '_rgb/segmentations/'+model+'/glomeruli/IFTA_EXC_00' + str(patient) + '_' + suffix + '_' + stain + '.png')
	m = image_utils.read_image('/home/lampert/data/Nephrectomies/' + stain + '/masks/tissue/IFTA_EXC_00' + str(patient) + '_' + suffix + '_' + stain + '_mask_tissue_lod1.png')

	print(numpy.amax(x2))

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
	print(len(regions))
	for region in regions:

		x0 = int(round(region.centroid[0]))
		y0 = int(round(region.centroid[1]))

		x0 = int(round((region.bbox[0] + region.bbox[2]) / 2))
		y0 = int(round((region.bbox[1] + region.bbox[3]) / 2))

		filename = str(x0) + '_' + str(y0) #+ '.png'
		print(os.path.join(output_base, filename))

#		print(numpy.amax(x[x0 - cut_width:x0 + cut_width, y0 - cut_width:y0 + cut_width]))
#		print(numpy.amax(x[x0 - cut_width:x0 + cut_width, y0 - cut_width:y0 + cut_width].astype('uint8')))
#		print(numpy.amax(x[x0 - cut_width:x0 + cut_width, y0 - cut_width:y0 + cut_width].astype('uint8')*255))

		image_utils.save_image(x[x0 - cut_width:x0 + cut_width, y0 - cut_width:y0 + cut_width].astype(numpy.uint8)*255, os.path.join(output_base, filename + '.png'))
		image_utils.save_image(x2[x0 - cut_width:x0 + cut_width, y0 - cut_width:y0 + cut_width].astype(numpy.uint8), os.path.join(output_base, 'probabilities', filename + '.png'))
		image_utils.save_image(gt[x0 - cut_width:x0 + cut_width, y0 - cut_width:y0 + cut_width].astype(numpy.uint8)*255, os.path.join(output_base, 'gt', filename + '.png'))
		image_utils.save_image(svsImage.read_region((y0 - cut_width, x0 - cut_width), 1, (windows_size, windows_size)), os.path.join(output_base, 'image', filename + '.png'))

#		image_utils.save_image(region.image.astype(numpy.uint8)*255, '/home/lampert/detection_patches/' + filename + '_2.png')

#		image_utils.save_binary_image(x[x0 - cut_width:x0 + cut_width, y0 - cut_width:y0 + cut_width], '/home/lampert/detection_patches/' + filename)
#		image_utils.save_binary_image(gt[x0 - cut_width:x0 + cut_width, y0 - cut_width:y0 + cut_width], '/home/lampert/detection_patches/gt/' + filename)
