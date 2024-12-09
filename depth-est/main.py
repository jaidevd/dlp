import os
import torch
from unet import UNet
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.v2 import functional as trx


op = os.path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DepthDataset(Dataset):
    def __init__(self, root):
        self.image_dir = op.join(root, "images")
        self.depth_dir = op.join(root, "depths")
        self.image_files = [
            op.join(self.image_dir, k) for k in os.listdir(self.image_dir)
        ]
        self.depth_files = [
            op.join(self.depth_dir, k) for k in os.listdir(self.depth_dir)
        ]
        assert len(self.image_files) == len(self.depth_files)
        assert set(map(op.basename, self.image_files)) == set(
            map(op.basename, self.depth_files)
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = read_image(self.image_files[idx], mode=ImageReadMode.RGB)
        depth = read_image(self.depth_files[idx], mode=ImageReadMode.GRAY)
        return image, depth


def collate(batch):
    images, depths = zip(*batch)
    images = torch.stack(images).to(device)
    images = trx.to_dtype(images, torch.float32, scale=True)
    depths = torch.stack(depths).to(device).squeeze(1)
    batch_size = depths.shape[0]
    dmax = depths.amax((1, 2)).reshape(batch_size, 1, 1)
    dmin = depths.amin((1, 2)).reshape(batch_size, 1, 1)
    depths = (depths - dmin) / (dmax - dmin)
    return images, depths


def depth_loss(pred, depths):
    depth = torch.nn.functional.sigmoid(depths).squeeze(1)
    mse = torch.mean((pred - depth) ** 2)
    mae = torch.mean(torch.abs(pred - depth))
    return 0.1 * mse + 0.9 * mae


if __name__ == "__main__":
    from tqdm import tqdm
    model = UNet()
    loader = DataLoader(
        DepthDataset("data/training"), batch_size=8, shuffle=True, collate_fn=collate
    )
    val_loader = DataLoader(
        DepthDataset("data/validation/"), batch_size=8, shuffle=True, collate_fn=collate
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_epochs = 1
    with tqdm(total=n_epochs) as ebar:
        for epoch in range(n_epochs):
            epoch_train_loss = epoch_val_loss = 0
            with tqdm(total=len(loader) + len(val_loader)) as bbar:
                model.train()
                for images, depths in loader:
                    pred = model(images)
                    loss = depth_loss(pred, depths)
                    loss.backward()
                    opt.step()
                    epoch_train_loss += loss.item()
                    bbar.set_postfix_str(f"{loss.item():.3f}")
                    bbar.update(1)
                model.eval()
                for images, depths in val_loader:
                    with torch.no_grad():
                        preds = model(images)
                    loss = depth_loss(pred, depths)
                    epoch_val_loss += loss.item()
                    bbar.update(1)
            epoch_train_loss /= len(loader)
            epoch_val_loss /= len(val_loader)
            ebar.set_postfix_str(f"{epoch_train_loss:.3f} / {epoch_val_loss:.3f}")
            ebar.update(1)
