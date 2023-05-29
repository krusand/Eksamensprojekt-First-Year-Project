import derm_ita
import os
from tqdm import tqdm
from PIL import Image
import pandas as pd
import random

path = "Data/images"
images = os.listdir(path)

# Testing:
# scores = {img_id: random.randint(1, 6) for img_id in img_ids}

scores = {}

for image in tqdm(images):
    if image == ".DS_Store":
        continue
    img = Image.open(path + "/" + image).convert("RGB")
    whole_image_ita = derm_ita.get_ita(image=img)
    scores[image] = derm_ita.get_fitzpatrick_type(whole_image_ita)

raw_df = pd.read_csv("data/metadata.csv")
df = raw_df.loc[:, ["img_id", "fitspatrick"]]
# df = df.dropna()

img_ids = df["img_id"].values
fitspatrick_metadata = df["fitspatrick"].values

metadata_scores = {img_ids[i]: fitspatrick_metadata[i] for i in range(len(img_ids))}


count = 0

for key in scores:
    if metadata_scores[key] == scores[key]:
        count += 1

n = len(df["fitspatrick"].dropna().values)

print(count / n)
