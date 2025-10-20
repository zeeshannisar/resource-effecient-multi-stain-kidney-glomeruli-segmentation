dir='/home/lampert/data/Nephrectomies/25/groundtruths/'
filename='IFTA_EXC_001_NX_III_25_gt_lod1.png'
filename='IFTA_EXC_003_NX_NE2_25_gt_lod1.png'
filename='IFTA_EXC_004_NX_NE3_25_gt_lod1.png'
filename='IFTA_EXC_002_NX_NE4_25_gt_lod1.png'

dir='/home/lampert/data/Nephrectomies/02/groundtruths/'
filename='IFTA_Nx_0014_02_gt_lod1.png'

from utils import image_utils
gt = image_utils.read_image(dir+filename)
import skimage
gt[gt != 2] = 0
import scipy
s = scipy.ndimage.generate_binary_structure(2, 2)
labelmask, _ = scipy.ndimage.label(gt, structure=s)
sizes = []
regions = skimage.measure.regionprops(labelmask)
for region in regions:
    sizes.append(region.area)

print(sizes)
