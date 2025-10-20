# -*- coding: utf-8 -*-
# Odyssee Merveille 27/10/17

import glob, time
import numpy
from PIL import Image
import utils.image_utils
import matplotlib.pylab as plt


''' Compute the precision and recall at each threshold from TP and FP vectors
	INPUT: 
		- vecTpPath: the TP numpy array saved on the disk (.npy)
		- vecFpPath: the FP numpy array saved on the disk (.npy)
		
	RETURN:
		- precision, recall: 1D numpy arrays
'''
def compute_precision_recall(vecTpPath, vecFpPath, maskPath):
	
	maskPil = Image.open(maskPath)
	positive = numpy.count_nonzero(numpy.array(maskPil))
	
	vecTp = numpy.load(vecTpPath)
	vecFp = numpy.load(vecFpPath)

	precision = vecTp / (vecTp + vecFp).astype(numpy.float32)
	recall = vecTp / float(positive)
	
	return precision, recall


'''
	Compute the mean precision / recall curves for the 4 test patients (nephrectomy) 
	for each glomeruli segmentation method (LeNet, UNet, Hog).
'''

if __name__ == '__main__':
	
	
	patientNb = ['16', '17', '18', '19']
	
	vecPrecisionLenet = []
	vecPrecisionHog = []
	vecPrecisionUnet = []
	vecRecallLenet = []
	vecRecallHog = []
	vecRecallUnet = []
	
	for nb in patientNb:
		
		maskPathLenet = "/data/Images/Sysmifta/nephrectomy/mask/IFTA_" + nb + "_16_lod3.png"
		maskPathHog = "/data/Images/Sysmifta/nephrectomy/mask/IFTA_" + nb + "_16_lod4.png"
		
		vecTpLenetPath = "/data/Results/Sysmifta/glomeruli_detection/Lenet/patches_maja/IFTA_" + nb + "_16_lod3_vecTp.npy"  
		vecFpLenetPath = "/data/Results/Sysmifta/glomeruli_detection/Lenet/patches_maja/IFTA_" + nb + "_16_lod3_vecFp.npy"  
		
		vecTpHogPath = "/data/Results/Sysmifta/glomeruli_detection/Hog/patient_" + nb + "/IFTA_" + nb + "_16_lod4_vecTp.npy"  
		vecFpHogPath = "/data/Results/Sysmifta/glomeruli_detection/Hog/patient_" + nb + "/IFTA_" + nb + "_16_lod4_vecFp.npy"  
		
		vecTpUnetPath = "/data/Results/Sysmifta/glomeruli_detection/UNet/without_augmentation/_16/IFTA_" + nb + "_16_lod4_vecTp.npy"  
		vecFpUnetPath = "/data/Results/Sysmifta/glomeruli_detection/UNet/without_augmentation/_16/IFTA_" + nb + "_16_lod4_vecFp.npy"  

		precisionLenet, recallLenet = compute_precision_recall(vecTpLenetPath, vecFpLenetPath, maskPathLenet)
		vecPrecisionLenet.append(precisionLenet)
		vecRecallLenet.append(recallLenet)
		
		precisionHog, recallHog = compute_precision_recall(vecTpHogPath, vecFpHogPath, maskPathHog)
		vecPrecisionHog.append(precisionHog)
		vecRecallHog.append(recallHog)
		
		precisionUnet, recallUnet = compute_precision_recall(vecTpUnetPath, vecFpUnetPath, maskPathHog)
		vecPrecisionUnet.append(precisionUnet)
		vecRecallUnet.append(recallUnet)
		
		
	
	meanRecallLenet = numpy.nanmean(vecRecallLenet, axis = 0)
	meanRecallHog = numpy.nanmean(vecRecallHog, axis = 0)
	meanRecallUnet = numpy.nanmean(vecRecallUnet, axis = 0)
	meanPrecisionLenet = numpy.nanmean(vecPrecisionLenet, axis = 0)
	meanPrecisionHog = numpy.nanmean(vecPrecisionHog, axis = 0)
	meanPrecisionUnet = numpy.nanmean(vecPrecisionUnet, axis = 0)
	
	stdPrecisionLenet = numpy.nanstd(vecPrecisionLenet, axis = 0)
	stdPrecisionHog = numpy.nanstd(meanPrecisionHog, axis = 0)
	stdPrecisionUnet = numpy.nanstd(meanPrecisionUnet, axis = 0)
	
	print("curves")
	fig = plt.figure()
	fig.patch.set_facecolor('white')
	plt.plot(meanRecallLenet, meanPrecisionLenet, 'b', label='Lenet')
	plt.plot(meanRecallHog, meanPrecisionHog, 'r', label = 'Hog')
	plt.plot(meanRecallUnet, meanPrecisionUnet, 'g', label = 'Unet')
	
	plt.fill_between(meanRecallLenet, meanPrecisionLenet - stdPrecisionLenet, meanPrecisionLenet + stdPrecisionLenet, alpha=0.5, interpolate = True)
	plt.fill_between(meanRecallHog, meanPrecisionHog - stdPrecisionHog, meanPrecisionHog + stdPrecisionHog, facecolor = 'red', alpha=0.5, interpolate = True)
	plt.fill_between(meanRecallUnet, meanPrecisionUnet - stdPrecisionUnet, meanPrecisionUnet + stdPrecisionUnet, facecolor = 'green', alpha=0.5, interpolate = True)
	plt.title("Pixel-based comparison of the detection of glomeruli ")
	plt.xlabel("recall")
	plt.ylabel("precision")
	plt.legend(loc=3)
	plt.tight_layout()
	plt.ylim(0,1)
	
	plt.savefig("/data/Results/Sysmifta/glomeruli_detection/mean_roc_curves.pdf")
	plt.show(block =False)
