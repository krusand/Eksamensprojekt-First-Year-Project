import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from skimage.transform import resize
from skimage import morphology
from skimage import color

from glob import glob


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

        for img, mask in zip(image_names, mask_names):
            mask = plt.imread(mask)
            image = plt.imread(img)[:, :, :3]
            mask = resize(mask, output_shape=image.shape)
            r, g, b = self.__averageColor(image, mask)
            avg_red_channel.append(r)
            avg_green_channel.append(g)
            avg_blue_channel.append(b)

            compactness.append(self.__compactness(mask))

        df["compactness"] = compactness
        df["avg_red_channel"] = avg_red_channel
        df["avg_green_channel"] = avg_green_channel
        df["avg_blue_channel"] = avg_blue_channel

        return df

    def __readMetadata(self, path):
        return pd.read_csv(path)

    def __compactness(self, mask):
        mask = color.rgb2gray(mask)
        area = np.sum(mask)

        struct_el = morphology.disk(3)
        mask_eroded = morphology.binary_erosion(mask, struct_el)
        perimeter = np.sum(mask - mask_eroded)

        return perimeter**2 / (4 * np.pi * area)

    def __averageColor(self, img, mask):
        img[mask == 0] = 0
        tot_pixels = np.sum(mask)
        red_avg = np.sum(img[:, :, 0]) / tot_pixels
        green_avg = np.sum(img[:, :, 1]) / tot_pixels
        blue_avg = np.sum(img[:, :, 2]) / tot_pixels
        pixel_color = np.array([red_avg, green_avg, blue_avg])
        return pixel_color


def main():
    FR = FeatureReader()
    df = FR.extractFeatures(
        mask_path="results", img_path="img_subset", metadata="metadata.csv"
    )
    print(df)


if __name__ == "__main__":
    main()
