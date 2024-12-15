import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms.v2.functional import to_dtype, resize

op = os.path


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss


class EnhanceDataset(Dataset):

    def __init__(self, imgpath, gtpath=None):
        super(EnhanceDataset, self).__init__()
        image_names = [f for f in os.listdir(imgpath) if f.endswith(".png")]
        ids = [op.splitext(k.split("_")[-1])[0] for k in image_names]
        df = pd.DataFrame({"image": image_names, "id": ids}).set_index(
            "id", verify_integrity=True
        )
        df["image"] = df["image"].apply(lambda x: op.join(imgpath, x))
        self._labeled = gtpath is not None
        if self._labeled:
            paths = [f for f in os.listdir(gtpath) if f.endswith(".png")]
            ids = [op.splitext(k.split("_")[-1])[0] for k in paths]
            ydf = pd.DataFrame({"label": paths, "id": ids}).set_index(
                "id", verify_integrity=True
            )
            df["label"] = ydf["label"].apply(lambda x: op.join(gtpath, x))
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = read_image(self.df.iloc[idx]["image"])
        if self._labeled:
            label = read_image(self.df.iloc[idx]["label"])
            return image, label
        return image

    def show(self, n):
        if self._labeled:
            fig, ax = plt.subplots(nrows=n, ncols=2, figsize=(16, 16))
            for i, (_, row) in enumerate(self.df.sample(n).iterrows()):
                image, label = self[i]
                ax[i, 0].imshow(image.permute(1, 2, 0))
                ax[i, 1].imshow(label.permute(1, 2, 0))
        else:
            fig, ax = plt.subplots(nrows=n, ncols=1, figsize=(16, 16))
            for i, (_, row) in enumerate(self.df.sample(n).iterrows()):
                image = self[i]
                ax[i].imshow(image.permute(1, 2, 0))
        [a.set_axis_off() for a in ax.ravel()]
        plt.tight_layout()
        plt.show()


def collate(batch):
    images, labels = map(torch.stack, zip(*batch))
    images = to_dtype(images, torch.float32, scale=True)
    h, w = labels.shape[-2:]
    h, w = map(int, (h / 4, w / 4))
    labels = resize(labels, (h, w))
    return images, to_dtype(labels, torch.float32, scale=True)


if __name__ == "__main__":
    EnhanceDataset("archive/train/train", "archive/train/gt").show(4)
    EnhanceDataset("archive/val/val", "archive/val/gt").show(4)
    EnhanceDataset("archive/test").show(4)
