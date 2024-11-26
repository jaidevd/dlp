from torchvision.datasets import ImageFolder
from torchvision.io import decode_image, ImageReadMode
from torchvision.models import resnet50
from torchvision.models.resnet import ResNet50_Weights
from torchvision.transforms.v2 import functional as trx
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset, StackDataset
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from joblib import Parallel, delayed
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABELS = {
    "Amphibia": 0,
    "Animalia": 1,
    "Arachnida": 2,
    "Aves": 3,
    "Fungi": 4,
    "Insecta": 5,
    "Mammalia": 6,
    "Mollusca": 7,
    "Plantae": 8,
    "Reptilia": 9,
}


def collate(batch):
    images, labels = zip(*batch)
    images = torch.stack(images).to(device)
    images = trx.to_dtype(images, torch.float32, scale=True)
    images = trx.normalize(images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return images, torch.tensor(labels).to(device)


def read(path, label):
    image = decode_image(path, mode=ImageReadMode.RGB)
    image = trx.resize(image, [256])
    image = trx.center_crop(image, [224])
    return image, label


folder = ImageFolder("data/train/")
tensor_labels = Parallel(n_jobs=-1)(delayed(read)(path, label) for path, label in folder.samples)
tensors, labels = zip(*tensor_labels)

ds = StackDataset(tensors, labels)
trix, valix = train_test_split(range(len(ds)), test_size=0.2, stratify=labels)
train_ds, val_ds = Subset(ds, trix), Subset(ds, valix)


train_loader = DataLoader(train_ds, batch_size=256, collate_fn=collate)
val_loader = DataLoader(val_ds, batch_size=128, collate_fn=collate)

model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, len(LABELS))
model = model.to(device)

losser = nn.CrossEntropyLoss()
optimizer = Adam(model.fc.parameters(), lr=0.0005)


def train(n_epochs=1):
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
        print(f'Train loss: {train_loss}; Train F1: {train_f1}; ', end="")
        print(f'Val loss: {val_loss}; Val F1: {val_f1}')


train(100)
torch.save(model.state_dict(), "models/resnet60-100-epoch.pth")
