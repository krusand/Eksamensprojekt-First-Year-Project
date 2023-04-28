import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import skimage
import os
from skimage import morphology
from skimage.transform import resize
import Segmentation

def loadImageVersions(im_path, mask_path):
    mask = Segmentation.loadImg(mask_path)
    color_img = Segmentation.loadImg(im_path, color=True)
    Byte_size = os.path.getsize(im_path)
    if Byte_size > 1000000: 
        color_img = resize(color_img, (color_img.shape[0] // 4, color_img.shape[1] // 4))
    mask[mask == 0.08322275] = 0
    mask[mask == 0.8690804] = 1
    im2 = color_img.copy()
    im2[mask == 0] = 0
    return color_img, mask, im2

def plotImages(im,mask,color_img):
    fig, axes = plt.subplots(nrows = 1, ncols=3, figsize=(7,7))
    axes[2].imshow(im)
    axes[1].imshow(mask, cmap="gray")
    axes[0].imshow(color_img)

def averageColor(im, mask):
    tot_pixels = np.sum(mask)
    red_avg = np.sum(im[:,:,0]) / tot_pixels
    green_avg = np.sum(im[:,:,1]) / tot_pixels
    blue_avg = np.sum(im[:,:,2]) / tot_pixels
    pixel_color = np.array([red_avg, green_avg, blue_avg])
    return pixel_color
