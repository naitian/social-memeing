import numpy as np
import torch
from torchvision.utils import make_grid, save_image
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
from torchvision.transforms.functional import resize
import torchvision.transforms.functional as F


import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
plt.rcParams["savefig.bbox"] = 'tight'


def show(imgs, title=None):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    fig.set_dpi(300)
    if title is not None:
        fig.suptitle(title, y=0.7, fontsize=10)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def load_image(path):
    return resize(read_image(path, mode=ImageReadMode.RGB), (300, 300))


def draw_grid(paths, title=None, ncols=8):
    show(make_grid([load_image(path) for path in paths], nrow=ncols), title=title)
