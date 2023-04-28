
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.filters import threshold_otsu
from skimage.draw import polygon
from skimage.transform import resize


def loadImg(path, resize_image = False, color=False):
    if resize_image:
        im = rgb2gray(plt.imread(path)[:,:,:3])
        return resize(im, (im.shape[0] // 4, im.shape[1] // 4), anti_aliasing=True)
    elif color:
        return plt.imread(path)[:,:,:3]
    else:
        return rgb2gray(plt.imread(path)[:,:,:3])


def blurImg(im,strength):
    return gaussian(im, strength,preserve_range=False)

def activeContourMask(im, radius, xval, yval):
    s = np.linspace(0, 2*np.pi, 200)
    r = yval + radius*np.sin(s)
    c = xval + radius*np.cos(s)
    init = np.array([r,c]).T
    snake = active_contour(im, init, w_line = 0)
    mask = np.zeros_like(im)
    rr, cc = polygon(snake[:,0], snake[:,1], im.shape)
    mask[rr,cc] = 1
    return mask

def plotImg(im):
    plt.imshow(im, cmap="gray")
    plt.show()



def plotImageWithMask(im, mask):
    im2 = im.copy()
    im2[mask == 0] = 0
    plt.imshow(im2, cmap="gray")
    plt.show()




