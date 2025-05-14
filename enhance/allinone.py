import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from main import EnhanceDataset
from torchvision.models import vgg16, VGG16_Weights
from torchvision.transforms.v2 import Normalize
from torchvision.transforms.v2.functional import to_dtype
from tqdm import tqdm

try:
    from torcheval.metrics.functional import peak_signal_noise_ratio as psnr
except OSError:
    from torcheval.metrics.functional import peak_signal_noise_ratio as psnr
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.normalize = Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.criterion = nn.MSELoss()

    def forward(self, pred, target):
        pred = self.normalize(pred)
        target = self.normalize(target)
        pred_features = self.vgg(pred)
        target_features = self.vgg(target)
        return self.criterion(pred_features, target_features)


class DenoiseSuperResolutionNet(nn.Module):
    def __init__(self):
        super(DenoiseSuperResolutionNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
        )
        self.upsample = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3 * 16, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor=4),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        x = self.upsample(x)
        return x


model = DenoiseSuperResolutionNet().to(device)
mse_loss = nn.MSELoss()
perceptual_loss = PerceptualLoss().to(device)
opt = Adam(model.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.66)


def collate(batch):
    images, labels = map(torch.stack, zip(*batch))
    images = to_dtype(images, torch.float32, scale=True)
    labels = to_dtype(labels, torch.float32, scale=True)
    return images, labels


def train_epoch(model, train_loader, val_loader, opt):
    model.train()
    train_loss = val_loss = train_psnr = val_psnr = 0
    n_batches = len(train_loader) + len(val_loader)
    with tqdm(total=n_batches) as pbar:
        for i, (lr, hr) in enumerate(train_loader):
            lr, hr = lr.to(device), hr.to(device)
            pred = model(lr)
            loss = mse_loss(pred, hr) + 0.1 * perceptual_loss(pred, hr)
            train_psnr += psnr(pred, hr)
            train_loss += loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
            pbar.update(1)
        model.eval()
        with torch.no_grad():
            for i, (lr, hr) in enumerate(val_loader):
                lr, hr = lr.to(device), hr.to(device)
                pred = model(lr)
                loss = mse_loss(pred, hr) + 0.1 * perceptual_loss(pred, hr)
                val_psnr += psnr(pred, hr)
                val_loss += loss.item()
                pbar.update(1)
    sched.step()
    train_psnr /= len(train_loader)
    val_psnr /= len(val_loader)
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    print(f"Train Loss: {train_loss:.4f} PSNR: {train_psnr:.4f}",  # noqa
          f"Val Loss: {val_loss:.4f} PSNR: {val_psnr:.4f}")
    return (train_psnr * len(train_loader) + val_psnr * len(val_loader)) / n_batches


if __name__ == "__main__":
    best_psnr = 0
    train_ds = EnhanceDataset("archive/train/train", "archive/train/gt")
    val_ds = EnhanceDataset("archive/val/val", "archive/val/gt")
    train_loader = DataLoader(
        train_ds, batch_size=6, shuffle=True, pin_memory=True, collate_fn=collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=6, shuffle=True, pin_memory=True, collate_fn=collate
    )
    n_epochs = 40
    for epoch in range(n_epochs):
        current_psnr = train_epoch(model, train_loader, val_loader, opt)
        if current_psnr > best_psnr:
            best_psnr = current_psnr
            torch.save(model.state_dict(), "best-naive.pth")
