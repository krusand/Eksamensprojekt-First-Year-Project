import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage import morphology

def compactness(mask_image_file):
    mask = plt.imread(mask_image_file)
    area = np.sum(mask)

    struct_el = morphology.disk(3)
    mask_eroded = morphology.binary_erosion(mask, struct_el)
    perimeter = np.sum(mask - mask_eroded)
    
    return perimeter**2/(4*np.pi*area)