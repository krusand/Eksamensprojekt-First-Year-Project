import os
import derm_ita
from PIL import Image
from tqdm import tqdm
import plotly.express as px
import pandas as pd
import numpy as np
import math


def classify_fitspatrick_score(img_path):
    classifications = []
    images = os.listdir(img_path)
    for image in images:
        whole_image_ita = derm_ita.get_ita(image=Image.open(img_path + "/" + image))
        classifications.append(derm_ita.get_fitzpatrick_type(whole_image_ita))
    return classifications


def create_plot(classifications):
    data = np.unique(list(classifications.values()), return_counts=True)
    n = sum(data[1])

    df = pd.DataFrame(
        dict(
            type=data[0],
            Count=data[1],
            text=[
                str(round((data[1][i] / n) * 100, 2)) + "%"
                for i in range((len(data[1])))
            ],
        )
    )

    df["type"] = df["type"].astype(str)

    fig = px.bar(
        df,
        x="type",
        y="Count",
        color="type",
        color_discrete_map={
            "1": "#d5bfa2",
            "2": "#c4a481",
            "3": "#b18f6f",
            "4": "#9a6947",
            "5": "#885026",
            "6": "#412d28",
        },
        labels={"type": "Fitzpatrick Type"},
        text="text",
    ).update_layout(
        title_text="Fitzpatrick type distribution for dataset",
        title_x=0.5,
        showlegend=False,
    )
    fig.update_traces(
        textfont_size=12, textangle=0, textposition="outside", cliponaxis=False
    )
    return fig


def save_plot(fig):
    fig.write_image(f"image/FitzPatricksScores.png")


def compare_fitspatrick_scores(classifications, comparison_path):
    raw_df = pd.read_csv(f"{comparison_path}/metadata.csv")
    df = raw_df.loc[:, ["img_id", "fitspatrick"]]
    df = df.dropna()

    img_ids = df["img_id"].values
    fitspatrick_metadata = df["fitspatrick"].values

    metadata_scores = {img_ids[i]: fitspatrick_metadata[i] for i in range(len(img_ids))}

    count = 0

    for key in classifications:
        if metadata_scores[key] == classifications[key]:
            count += 1

    return count / len(classifications)


def preparedata(classifications, metadata, convert=False):
    # create metadata dictionary
    img_ids = metadata["img_id"].values
    fitspatrick_metadata = metadata["fitspatrick"].values
    if convert:
        metadata_scores = {
            img_ids[i]: int(fitspatrick_metadata[i]) for i in range(len(img_ids))
        }
    else:
        metadata_scores = {
            img_ids[i]: fitspatrick_metadata[i] for i in range(len(img_ids))
        }

    # create classification dictionary
    img_ids_class = classifications["img_id"].values
    fitspatrick_class = classifications["fitspatrick"].values

    classifications_dict = {
        img_ids_class[i]: int(fitspatrick_class[i]) for i in range(len(img_ids_class))
    }
    return classifications_dict, metadata_scores


def mean_average_difference_fitspatrick(classifications_dict, metadata_scores):
    absolute_difference = 0

    # calculate absolute difference between predicted and true fitspatrick score
    for key in metadata_scores:
        absolute_difference += np.abs(metadata_scores[key] - classifications_dict[key])

    mean_absolute_difference = absolute_difference / len(metadata_scores)
    return mean_absolute_difference


def interpolate_fitspatrick_metadata(classifications_dict, metadata_scores):
    for key in metadata_scores:
        if math.isnan(metadata_scores[key]):
            metadata_scores[key] = classifications_dict[key]
        else:  # fitspatrick score is a float so convert
            metadata_scores[key] = int(metadata_scores[key])
    return metadata_scores


def main():
    raw_df = pd.read_csv("data/metadata.csv")
    df = raw_df.loc[:, ["img_id", "fitspatrick"]]
    classifications = pd.read_csv("data/fits.csv")
    classifications, metadata_scores = preparedata(classifications, df)

    # Calculate mean absolute difference
    # mean_average_difference_fitspatrick(classifications, metadata_scores)

    # interpolate fitspatrick
    fitspatrick_scores = interpolate_fitspatrick_metadata(
        classifications, metadata_scores
    )

    fig = create_plot(fitspatrick_scores)
    save_plot(fig)


if __name__ == "__main__":
    main()
