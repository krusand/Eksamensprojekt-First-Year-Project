import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.draw import polygon
from skimage.transform import resize
from skimage.filters import threshold_otsu

# This file is used to create masks using basic segmentation methods.
# The methods used are active contour, and otsu threshold


def loadImg(path, resize_image=False, color=False):
    """
    Loads image given a path.
        Parameters:
            path : str
                Path to the image
            resize_image : bool
                Option to resize image or not
            color : bool
                Load image with or without colors

        Returns:
            np.array containing pixel values
    """
    if resize_image:
        im = rgb2gray(plt.imread(path)[:, :, :3])
        return resize(im, (im.shape[0] // 4, im.shape[1] // 4), anti_aliasing=True)
    elif color:
        return plt.imread(path)[:, :, :3]
    else:
        return rgb2gray(plt.imread(path)[:, :, :3])


def blurImg(im, strength):
    """Blurs the image by a given strength and returns it"""
    return gaussian(im, strength, preserve_range=False)


def activeContourMask(im, radius, xval, yval):
    """Creates an active contour mask"""
    s = np.linspace(0, 2 * np.pi, 200)
    r = yval + radius * np.sin(s)
    c = xval + radius * np.cos(s)
    init = np.array([r, c]).T
    snake = active_contour(im, init, w_line=0)
    mask = np.zeros_like(im)
    rr, cc = polygon(snake[:, 0], snake[:, 1], im.shape)
    mask[rr, cc] = 1
    return mask


def plotImg(im):
    """Plots a grayscale image"""
    plt.imshow(im, cmap="gray")
    plt.show()


def plotImageWithMask(im, mask):
    """Plots a binary mask on top of an grayscale image"""
    im2 = im.copy()
    im2[mask == 0] = 0
    plt.imshow(im2, cmap="gray")
    plt.show()


def otsuThreshold(img):
    """
    Computes the otsu threshold and returns a binary mask
    where all pixels less than the threshold are white and otherwise black

    Input: Grayscale image
    Outputs: Binary mask
    """
    threshold = threshold_otsu(img)
    return img < threshold
