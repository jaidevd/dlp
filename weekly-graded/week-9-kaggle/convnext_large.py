import os.path as op
import random

from joblib import Parallel, delayed
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Subset, StackDataset
from torchvision.datasets import ImageFolder
from torchvision.io import decode_image, ImageReadMode
from torchvision.models import convnext
from torchvision.transforms.v2 import functional as trx, ColorJitter
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABELS = {
    "Amphibia": 0,
    "Animalia": 1,  # Confuses with Mollusca (23 times)
    "Arachnida": 2,
    "Aves": 3,
    "Fungi": 4,
    "Insecta": 5,
    "Mammalia": 6,
    "Mollusca": 7,  # Confuses with Animalia (23 times)
    "Plantae": 8,
    "Reptilia": 9,
}

METRICS_CACHE = {}


def _save(model, fname, **metrics):
    METRICS_CACHE.update(metrics)
    torch.save(model, fname)


def save_best(model, minimizing, maximizing, mode="both", fname=None, **metrics):
    if fname is None:
        fname = "best.pth"
    best_min = METRICS_CACHE.get(minimizing, float("inf"))
    best_max = METRICS_CACHE.get(maximizing, 0)
    new_min, new_max = metrics[minimizing], metrics[maximizing]

    if mode == "both" and new_min <= best_min and new_max >= best_max:
        print('Better model found.')  # NOQA: T201
        _save(model, fname, **metrics)
    elif mode == "either" and (new_min <= best_min or new_max >= best_max):
        _save(model, fname, **metrics)


def _preproc(image):
    image = trx.to_dtype(image, torch.float32, scale=True)
    return trx.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def collate(batch):
    images, labels = zip(*batch)
    images = torch.stack(images).to(device)
    images = _preproc(images)
    return images, torch.tensor(labels).to(device)


def _read_image(path):
    image = decode_image(path, mode=ImageReadMode.RGB)
    image = trx.resize(image, [256])
    return trx.center_crop(image, [224])


def read(path, label, augment=True):
    image = _read_image(path)
    if augment and random.choice([True, False]):
        image = trx.horizontal_flip(image)
    if augment and random.choice([True, False]):
        image = ColorJitter(0.2, 0.2, 0.2)(image)

    return image, label


def get_loaders(root):
    folder = ImageFolder("data/train/")
    tensor_labels = Parallel(n_jobs=-1)(delayed(read)(path, label) for path, label in folder.samples)
    tensors, labels = zip(*tensor_labels)

    ds = StackDataset(tensors, labels)
    trix, valix = train_test_split(range(len(ds)), test_size=0.3, stratify=labels)
    train_ds, val_ds = Subset(ds, trix), Subset(ds, valix)

    train_loader = DataLoader(train_ds, batch_size=128, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=64, collate_fn=collate)
    return train_loader, val_loader


def get_model():
    model = convnext.convnext_large(weights=convnext.ConvNeXt_Large_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    clf = model.classifier[2]
    model.classifier[2] = nn.Linear(in_features=clf.in_features, out_features=10)
    model = model.to(device)
    return model


def train(model, train_loader, val_loader, n_epochs=1, fname="models/resnet50.pth"):
    losser = nn.CrossEntropyLoss()
    optimizer = Adam(model.classifier.parameters(), lr=0.005)
    scheduler = MultiStepLR(optimizer, [5])  # NOQA: F841
    for epoch in tqdm(range(n_epochs)):
        train_loss = val_loss = 0
        model.train()
        y_train_pred = []
        y_train_true = []
        for images, labels in train_loader:
            optimizer.zero_grad()
            out = model(images)
            y_train_pred.extend(out.argmax(dim=1).tolist())
            y_train_true.extend(labels.tolist())
            loss = losser(out, labels)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        scheduler.step()
        train_loss = round(train_loss / len(train_loader), 3)
        train_f1 = round(f1_score(y_train_true, y_train_pred, average="macro"), 2)

        model.eval()
        y_val_pred = []
        y_val_true = []
        for images, labels in val_loader:
            with torch.no_grad():
                out = model(images)
            y_val_pred.extend(out.argmax(dim=1).tolist())
            y_val_true.extend(labels.tolist())
            loss = losser(out, labels)
            val_loss += loss.item()
        val_loss = round(val_loss / len(val_loader), 3)
        val_f1 = round(f1_score(y_val_true, y_val_pred, average="macro"), 2)
        # lr = round(scheduler.get_last_lr()[0], 6)
        print(f'Train loss: {train_loss}; Train F1: {train_f1}; ', end="")  # NOQA: T201
        print(f'Val loss: {val_loss}; Val F1: {val_f1}')                    # NOQA: T201
        save_best(model, minimizing="val_loss", maximizing="val_f1", fname=fname,
                  val_loss=val_loss, val_f1=val_f1)


def predict(model, dfpath="data/sample_submission.csv", imgroot="data/test/",
            outpath="submission.csv"):
    df = pd.read_csv(dfpath)
    labels = []
    model.eval()
    for ix, rowdata in df.iterrows():
        image = _read_image(imgroot + rowdata["Image_ID"] + ".jpg")
        image = _preproc(image).unsqueeze(0)
        with torch.no_grad():
            out = model(image.to(device))
        labels.append(out.argmax(dim=1).item())
    df['Label'] = labels
    if op.exists(outpath):
        raise FileExistsError(f"{outpath} already exists")
    df.to_csv(outpath, index=False)


if __name__ == "__main__":
    train_loader, val_loader = get_loaders("data/train/")
    model = get_model()
    train(model, train_loader, val_loader, n_epochs=10, fname="models/convnext-large-last.pth")
    predict(model, dfpath="data/sample_submission.csv", imgroot="data/test/",
            outpath="data/convnext-large-last-augmented.csv")
