import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from skimage.transform import resize
from skimage import morphology
from skimage import color




from glob import glob

class FeatureReader:

    def extract_features(self, mask_path, img_path, metadata):
        image_names = sorted(glob(f"{img_path}/*.png"))
        mask_names = sorted(glob(f"{mask_path}/*.png"))
        #df = self.__read_metadata(metadata)
        df = pd.DataFrame()

        compactness = []

        for img, mask in zip(image_names, mask_names):
            mask = color.rgb2gray(plt.imread(mask))
            image = plt.imread(img)[:,:,:3]

            compactness.append(self.__compactness(mask))

        df["compactness"] = compactness

        return df

    def __read_metadata(self, path):
        return pd.read_csv(path)

    def __compactness(self, mask):
        area = np.sum(mask)

        struct_el = morphology.disk(3)
        mask_eroded = morphology.binary_erosion(mask, struct_el)
        perimeter = np.sum(mask - mask_eroded)
        
        return perimeter**2/(4*np.pi*area)

def main():
    FR = FeatureReader()
    df = FR.extract_features(mask_path="results", img_path= "img_subset", metadata="metadata.csv")
    print(df["compactness"].min())

if __name__ == "__main__":
    main()