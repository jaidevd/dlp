import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms.v2.functional import to_dtype

try:
    from torcheval.metrics.functional import peak_signal_noise_ratio
except OSError:
    from torcheval.metrics.functional import peak_signal_noise_ratio
from tqdm import tqdm
import numpy as np
from main import EnhanceDataset


class Generator(nn.Module):
    def __init__(self, scale_factor=4):
        super(Generator, self).__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4), nn.PReLU()
        )
        self.residual_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(16)])
        self.upsampling = nn.Sequential(
            *[UpsampleBlock(64) for _ in range(int(np.log2(scale_factor)))]
        )
        self.final = nn.Conv2d(64, 3, kernel_size=9, padding=4)

    def forward(self, x):
        x = self.initial(x)
        x = self.residual_blocks(x)
        x = self.upsampling(x)
        x = self.final(x)
        return torch.tanh(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return x + residual


class UpsampleBlock(nn.Module):
    def __init__(self, channels):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_channels, out_channels, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride, 1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.blocks = nn.Sequential(
            discriminator_block(3, 64, 2),
            discriminator_block(64, 128, 2),
            discriminator_block(128, 256, 2),
            discriminator_block(256, 512, 2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.blocks(x)
        x = self.classifier(x)
        return x.view(x.size(0), -1)


def train_srgan(generator, discriminator, train_loader, val_loader, n_epochs, device):
    # Loss functions
    criterion_gan = nn.BCELoss()
    criterion_content = nn.MSELoss()

    # Optimizers
    opt_g = optim.Adam(generator.parameters(), lr=0.000005)
    opt_d = optim.Adam(discriminator.parameters(), lr=0.000005)
    steppers = [optim.lr_scheduler.StepLR(opt_g, 5, gamma=0.5),
                optim.lr_scheduler.StepLR(opt_d, 5, gamma=0.5)]

    best_psnr_epoch = best_psnr = 0
    n_batches = len(train_loader) + len(val_loader)

    for epoch in range(n_epochs):
        epoch_train_loss = epoch_val_loss = train_psnr = val_psnr = 0
        with tqdm(total=n_batches) as pbar:
            generator.train()
            discriminator.train()
            for lr_images, hr_images in train_loader:
                batch_size = lr_images.size(0)

                # Move to device
                lr_images = lr_images.to(device)
                hr_images = hr_images.to(device)

                # Ground truths
                real_labels = torch.ones(batch_size, 1).to(device)
                fake_labels = torch.zeros(batch_size, 1).to(device)

                opt_g.zero_grad()

                # Generate SR images
                sr_images = generator(lr_images)

                # Adversarial loss
                gen_validity = discriminator(sr_images)
                loss_gan = criterion_gan(gen_validity, real_labels)

                # Content loss
                loss_content = criterion_content(sr_images, hr_images)

                # Total generator loss
                loss_g = loss_content + 1e-3 * loss_gan
                loss_g.backward()
                opt_g.step()

                opt_d.zero_grad()

                # Loss on real images
                real_validity = discriminator(hr_images)
                loss_real = criterion_gan(real_validity, real_labels)

                # Loss on fake images
                fake_validity = discriminator(sr_images.detach())
                loss_fake = criterion_gan(fake_validity, fake_labels)

                # Total discriminator loss
                loss_d = (loss_real + loss_fake) / 2
                loss_d.backward()
                opt_d.step()
                litem = (loss_d.item() + loss_g.item()) / 2
                epoch_train_loss += litem

                train_psnr += peak_signal_noise_ratio(
                    sr_images, hr_images, data_range=1.0
                )

                pbar.set_postfix_str(f"Train loss: {litem:.4f}")
                pbar.update(1)

            generator.eval()
            discriminator.eval()
            for lr_images, hr_images in val_loader:
                batch_size = lr_images.size(0)

                # Move to device
                lr_images = lr_images.to(device)
                hr_images = hr_images.to(device)

                # Generate SR images
                with torch.no_grad():
                    sr_images = generator(lr_images)

                    gen_validity = discriminator(sr_images)
                real_labels = torch.ones(batch_size, 1).to(device)
                loss_gan = criterion_gan(gen_validity, real_labels)

                # Content loss
                loss_content = criterion_content(sr_images, hr_images)

                # Total generator loss
                loss_g = loss_content + 1e-3 * loss_gan
                litem = loss_g.item()
                epoch_val_loss += litem
                val_psnr += peak_signal_noise_ratio(
                    sr_images, hr_images, data_range=1.0
                )
                pbar.set_postfix_str(f"Val loss: {litem:.4f}")
                pbar.update(1)
            [stepper.step() for stepper in steppers]
        epoch_train_loss /= len(train_loader)
        epoch_val_loss /= len(val_loader)
        train_psnr /= len(train_loader)
        val_psnr /= len(val_loader)
        current_psnr = (val_psnr * len(val_loader) + train_psnr * len(train_loader)) / n_batches
        if current_psnr > best_psnr:
            best_psnr = current_psnr
            best_psnr_epoch = epoch
            print(f'Better model found at epoch {epoch}')  # NOQA: T201
            torch.save(generator.state_dict(), "generator-best.pth")
            torch.save(discriminator.state_dict(), "discriminator-best.pth")
        elif epoch - best_psnr_epoch > 5:
            print(f'Stopping at epoch {epoch} - best epoch was {best_psnr_epoch}')  # NOQA: T201
            break

        print(  # NOQA: T201
            f"Train: {epoch_train_loss:.4f}; Val: {epoch_val_loss:.4f};",
            f"Train PSNR: {train_psnr:.4f}; Val PSNR: {val_psnr:.4f}",
        )


def collate_sr(batch):
    images, labels = map(torch.stack, zip(*batch))
    images = to_dtype(images, torch.float32, scale=True)
    labels = to_dtype(labels, torch.float32, scale=True)
    return images, labels


def main():
    n_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize networks
    generator = Generator().to(device)
    generator.load_state_dict(torch.load("generator-best.pth", weights_only=True))
    discriminator = Discriminator().to(device)
    discriminator.load_state_dict(torch.load("discriminator-best.pth", weights_only=True))

    train_ds = EnhanceDataset("denoised/train/train/", "archive/train/gt/")
    train_loader = DataLoader(
        train_ds, batch_size=4, shuffle=True, pin_memory=True, collate_fn=collate_sr
    )
    val_ds = EnhanceDataset("denoised/val/val/", "archive/val/gt/")
    val_loader = DataLoader(
        val_ds, batch_size=4, shuffle=True, pin_memory=True, collate_fn=collate_sr
    )

    # Train the model
    train_srgan(generator, discriminator, train_loader, val_loader, n_epochs, device)

    # Save the trained model
    torch.save(generator.state_dict(), f"generator-{n_epochs}.pth")
    torch.save(discriminator.state_dict(), f"discriminator-{n_epochs}.pth")


if __name__ == "__main__":
    main()
