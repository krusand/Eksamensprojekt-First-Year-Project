import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from glob import glob
from tqdm import tqdm
import os

# To use the script, run the script.
# When an image comes up left click to select an area based on the model. Right click to deselect an area.
# Press q to update an image. Press e,q to save an image. Press d,q to discard an image.
# Press r,q to reset an image. Press m,q to save and manually edit the mask
# To select/deselect click mouse and then q to update.

# Make sure the pictures are in a folder called images.
# Optionally, they could be in data/images but one would have to change the path in throughout the file.


# Helper functions to display images
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.5])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


with open("images.csv", "r") as f:
    f = f.read().strip()
    part = f.split("\n")


# import the model checkpoint

import sys

sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

# large model - download: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
# sam_checkpoint = "sam_vit_h_4b8939.pth"
# model_type = "vit_h"

# medium model - download: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
sam_checkpoint = "sam_vit_l_0b3195.pth"
model_type = "vit_l"

# small model - download: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
# sam_checkpoint = "sam_vit_b_01ec64.pth"
# model_type = "vit_b"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

# Comment out the lines below if neither Metal or Cuda hardware accerelation is available on your system.

# CUDA (NVIDIA)
torch.cuda.empty_cache()
device = "cuda"
sam.to(device=device)

# MACBOOK METAL
# device = "mps"
# sam.to(device=device)

predictor = SamPredictor(sam)
print("Model loaded")
images = glob("images/*.png")

results = glob("results/*.png")

# Windows
results = [s.split("\\")[-1] for s in results]

# Mac
# results = [s.split("/")[-1] for s in results]

with open("discarded.csv", "r") as f:
    results += [n for n in f.read().strip().split("\n")]

np.random.shuffle(images)

print("Loaded data")


class ImagePoints:
    t = {1: 1, 3: 0}

    def __init__(self) -> None:
        self.points = np.array([])
        self.labels = np.array([])
        self.status = None

    def update_coords(self):
        l = self.points
        self.points = np.array([[l[i], l[i + 1]] for i in range(0, len(l), 2)])

    def onclick(self, event):
        self.points = np.append(
            self.points, np.array([[int(event.xdata), int(event.ydata)]])
        )
        self.labels = np.append(self.labels, self.t[event.button])

    def on_key(self, event):
        if event.key == "e":
            self.status = "finished"
        elif event.key == "r":
            self.status = "reset"
        elif event.key == "d":
            self.status = "discard"
        elif event.key == "m":
            self.status = "manual"
        elif event.key != "q":
            print("you pressed", event.key, event.xdata, event.ydata)

    # print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
    #       ('double' if event.dblclick else 'single', event.button,
    #        event.x, event.y, event.xdata, event.ydata))


for image_path in images:
    # for windows
    image_name = image_path[7:]

    # for mac
    # image_name = image_path.split("/")[-1]

    if image_name in results or image_name in part[5:]:
        continue
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    predictor.set_image(image)

    data = ImagePoints()

    while True:
        if len(data.points) > 0:
            masks, scores, logits = predictor.predict(
                point_coords=data.points,
                point_labels=data.labels,
                multimask_output=False,
            )

            # mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask

        fig, ax = plt.subplots()

        cid2 = fig.canvas.mpl_connect("key_press_event", data.on_key)
        cid = fig.canvas.mpl_connect("button_press_event", data.onclick)
        # ax.figure(figsize=(10,10))
        ax.imshow(image)
        if len(data.points) > 0:
            show_mask(masks, ax)
            show_points(data.points, data.labels, ax)
        # plt.axis('off')
        plt.show()

        if data.status == "discard":
            print("Discarded")
            with open("discarded.csv", "a") as f:
                f.writelines(image_name + "\n")
            break
        elif data.status == "manual":
            print("Manual editing")
            plt.imsave("results/" + image_name, masks[0].squeeze(), cmap="gray")
            # os.system(f"open results/{image_name}")
            os.startfile(f"results\{image_name}")
            os.startfile(f"images\{image_name}")

            break
        elif data.status == "finished":
            print("Saving Image")
            plt.imsave("results/" + image_name, masks[0].squeeze(), cmap="gray")
            print(len(glob("results/*.png")))
            break
        elif data.status == "reset":
            data = ImagePoints()
        else:
            data.update_coords()
