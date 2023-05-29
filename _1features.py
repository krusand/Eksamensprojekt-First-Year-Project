import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from skimage.transform import resize
from skimage import morphology
from skimage import color

from glob import glob

import cv2
from sklearn.cluster import KMeans


class FeatureReader:
    def extractFeatures(self, mask_path, img_path, metadata):
        image_names = sorted(glob(f"{img_path}/*.png"))
        mask_names = sorted(glob(f"{mask_path}/*.png"))
        # df = self.__readMetadata(metadata)
        df = pd.DataFrame()

        compactness = []
        avg_red_channel = []
        avg_green_channel = []
        avg_blue_channel = []
        multicolor_rate = []

        for img, mask in zip(image_names, mask_names):
            mask = plt.imread(mask)
            image = plt.imread(img)[:, :, :3]
            mask = resize(mask, output_shape=image.shape)
            r, g, b = self.__averageColor(image, mask)
            avg_red_channel.append(r)
            avg_green_channel.append(g)
            avg_blue_channel.append(b)

            compactness.append(self.__compactness(mask))
            multicolor_rate.append(self.get_multicolor_rate(image, mask, 3))

        df["compactness"] = compactness
        df["avg_red_channel"] = avg_red_channel
        df["avg_green_channel"] = avg_green_channel
        df["avg_blue_channel"] = avg_blue_channel
        df["multicolor_rate"] = multicolor_rate

        return df

    def __readMetadata(self, path):
        return pd.read_csv(path)

    def compactness(self, mask):
        mask = color.rgb2gray(mask)
        area = np.sum(mask)

        struct_el = morphology.disk(3)
        mask_eroded = morphology.binary_erosion(mask, struct_el)
        perimeter = np.sum(mask - mask_eroded)

        return perimeter**2 / (4 * np.pi * area)

    def averageColor(self, img, mask):
        img[mask == 0] = 0
        tot_pixels = np.sum(mask)
        red_avg = np.sum(img[:, :, 0]) / tot_pixels
        green_avg = np.sum(img[:, :, 1]) / tot_pixels
        blue_avg = np.sum(img[:, :, 2]) / tot_pixels
        pixel_color = np.array([red_avg, green_avg, blue_avg])
        return pixel_color

    def get_com_col(sef, cluster, centroids, com_col_list):
        labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
        (hist, _) = np.histogram(cluster.labels_, bins=labels)
        hist = hist.astype("float")
        hist /= hist.sum()

        rect = np.zeros((50, 300, 3), dtype=np.uint8)
        colors = sorted([(percent, color) for (percent, color) in zip(hist, centroids)])
        start = 0
        for percent, color in colors:
            if percent > 0.08:
                com_col_list.append(color)
            end = start + (percent * 300)
            cv2.rectangle(
                rect,
                (int(start), 0),
                (int(end), 50),
                color.astype("uint8").tolist(),
                -1,
            )
            start = end

    def get_multicolor_rate(self, im, mask, n):
        im = resize(im, (im.shape[0] // 4, im.shape[1] // 4), anti_aliasing=True)
        mask = resize(
            mask, (mask.shape[0] // 4, mask.shape[1] // 4), anti_aliasing=True
        )
        im2 = im.copy()
        im2[mask == 0] = 0

        columns = im.shape[0]
        rows = im.shape[1]
        col_list = []
        for i in range(columns):
            for j in range(rows):
                if mask[i][j] != 0:
                    col_list.append(im2[i][j] * 256)

        com_col_list = []

        cluster = KMeans(n_clusters=n).fit(col_list)
        self.get_com_col(cluster, cluster.cluster_centers_, com_col_list)

        dist_list = []
        m = len(com_col_list)

        for i in range(0, m - 1):
            j = i + 1
            col_1 = com_col_list[i]
            col_2 = com_col_list[j]
            dist_list.append(
                np.sqrt(
                    (col_1[0] - col_2[0]) ** 2
                    + (col_1[1] - col_2[1]) ** 2
                    + (col_1[2] - col_2[2]) ** 2
                )
            )

        return np.max(dist_list)


def main():
    FR = FeatureReader()
    df = FR.extractFeatures(
        mask_path="results", img_path="img_subset", metadata="metadata.csv"
    )
    df.to_csv("Data/features.csv")


if __name__ == "__main__":
    main()
