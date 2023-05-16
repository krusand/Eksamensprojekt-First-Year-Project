import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import cv2
from sklearn.cluster import KMeans


def get_com_col(cluster, centroids, com_col_list):
    # Get the number of different clusters, create histogram, and normalize
    labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
    (hist, _) = np.histogram(cluster.labels_, bins = labels)
    hist = hist.astype("float")
    hist /= hist.sum()

    # Create frequency rect and iterate through each cluster's color and percentage
    rect = np.zeros((50, 300, 3), dtype=np.uint8)
    colors = sorted([(percent, color) for (percent, color) in zip(hist, centroids)])
    start = 0
    for (percent, color) in colors:
        if percent > 0.08:
            com_col_list.append(color)
        end = start + (percent * 300)
        cv2.rectangle(rect, (int(start), 0), (int(end), 50), \
                      color.astype("uint8").tolist(), -1)
        start = end


def is_mc(im, mask, n):
    im2 = im.copy()
    im2[mask == 0] = 0

    columns = im.shape[0]
    rows = im.shape[1]
    col_list = []
    for i in range(columns):
        for j in range(rows):
            if mask[i][j] != 0:
                col_list.append(im2[i][j]*256)
    
    com_col_list = []

    cluster = KMeans(n_clusters=n).fit(col_list)
    get_com_col(cluster, cluster.cluster_centers_, com_col_list)

    dist_list = []
    m = len(com_col_list)

    for i in range(0, m-1):
        j = i + 1
        col_1 = com_col_list[i]
        col_2 = com_col_list[j]
        dist_list.append(np.sqrt((col_1[0] - col_2[0])**2 + (col_1[1] - col_2[1])**2 + (col_1[2] - col_2[2])**2))

    return np.max(dist_list)
