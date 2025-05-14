import torch
from tqdm import tqdm
import math
from torch import nn, Tensor
from torch.nn import functional as F_torch  # NOQA: N812
from torch.optim.swa_utils import AveragedModel
from torchvision import models, transforms
from torchvision.models.feature_extraction import create_feature_extractor
from typing import cast
from main import EnhanceDataset
from srgan import collate_sr, Discriminator
import os
try:
    from torcheval.metrics.functional import peak_signal_noise_ratio as psnr
except OSError:
    from torcheval.metrics.functional import peak_signal_noise_ratio as psnr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_layers() -> nn.Sequential:
    net_cfg = [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ]
    layers = nn.Sequential()
    in_channels = 3
    for v in net_cfg:
        if v == "M":
            layers.append(nn.MaxPool2d((2, 2), (2, 2)))
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, (3, 3), (1, 1), (1, 1))
            layers.append(conv2d)
            layers.append(nn.BatchNorm2d(v))
            layers.append(nn.ReLU(True))
            in_channels = v

    return layers


class _FeatureExtractor(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
    ) -> None:
        super(_FeatureExtractor, self).__init__()
        self.features = _make_layers()

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

        # Initialize neural network weights
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


class SRResNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        channels: int = 64,
        num_rcb: int = 16,
        upscale: int = 4,
    ) -> None:
        super(SRResNet, self).__init__()
        # Low frequency information extraction layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, channels, (9, 9), (1, 1), (4, 4)),
            nn.PReLU(),
        )

        # High frequency information extraction block
        trunk = []
        for _ in range(num_rcb):
            trunk.append(_ResidualConvBlock(channels))
        self.trunk = nn.Sequential(*trunk)

        # High-frequency information linear fusion layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
        )

        # zoom block
        upsampling = []
        if upscale == 2 or upscale == 4 or upscale == 8:
            for _ in range(int(math.log(upscale, 2))):
                upsampling.append(_UpsampleBlock(channels, 2))
        else:
            raise NotImplementedError(f"Upscale factor `{upscale}` is not support.")
        self.upsampling = nn.Sequential(*upsampling)

        # reconstruction block
        self.conv3 = nn.Conv2d(channels, out_channels, (9, 9), (1, 1), (4, 4))

        # Initialize neural network weights
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> Tensor:
        conv1 = self.conv1(x)
        x = self.trunk(conv1)
        x = self.conv2(x)
        x = torch.add(x, conv1)
        x = self.upsampling(x)
        x = self.conv3(x)

        x = torch.clamp_(x, 0.0, 1.0)

        return x


class _ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super(_ResidualConvBlock, self).__init__()
        self.rcb = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        x = self.rcb(x)

        x = torch.add(x, identity)

        return x


class _UpsampleBlock(nn.Module):
    def __init__(self, channels: int, upscale_factor: int) -> None:
        super(_UpsampleBlock, self).__init__()
        self.upsample_block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels * upscale_factor * upscale_factor,
                (3, 3),
                (1, 1),
                (1, 1),
            ),
            nn.PixelShuffle(upscale_factor),
            nn.PReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.upsample_block(x)

        return x


class ContentLoss(nn.Module):
    """Constructs a content loss function based on the VGG19 network.
    Using high-level feature mapping layers from the latter layers will focus more on the texture content of the image.

    Paper reference list:
        -`Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network <https://arxiv.org/pdf/1609.04802.pdf>` paper.
        -`ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks                    <https://arxiv.org/pdf/1809.00219.pdf>` paper.
        -`Perceptual Extreme Super Resolution Network with Receptive Field Block               <https://arxiv.org/pdf/2005.12597.pdf>` paper.

    """

    def __init__(
        self,
        num_classes: int,
        model_weights_path: str,
        feature_nodes: list,
        feature_normalize_mean: list,
        feature_normalize_std: list,
    ) -> None:
        super(ContentLoss, self).__init__()
        # Define the feature extraction model
        model = _FeatureExtractor(num_classes)
        # Load the pre-trained model
        if model_weights_path == "":
            model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        elif model_weights_path is not None and os.path.exists(model_weights_path):
            checkpoint = torch.load(
                model_weights_path, map_location=lambda storage, loc: storage
            )
            if "state_dict" in checkpoint.keys():
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)
        else:
            raise FileNotFoundError("Model weight file not found")
        # Extract the output of the feature extraction layer
        self.feature_extractor = create_feature_extractor(model, feature_nodes)
        # Select the specified layers as the feature extraction layer
        self.feature_extractor_nodes = feature_nodes
        # input normalization
        self.normalize = transforms.Normalize(
            feature_normalize_mean, feature_normalize_std
        )
        # Freeze model parameters without derivatives
        for model_parameters in self.feature_extractor.parameters():
            model_parameters.requires_grad = False
        self.feature_extractor.eval()

    def forward(self, sr_tensor: Tensor, gt_tensor: Tensor) -> [Tensor]:
        assert (
            sr_tensor.size() == gt_tensor.size()
        ), "Two tensor must have the same size"
        device = sr_tensor.device

        losses = []
        # input normalization
        sr_tensor = self.normalize(sr_tensor)
        gt_tensor = self.normalize(gt_tensor)

        # Get the output of the feature extraction layer
        sr_feature = self.feature_extractor(sr_tensor)
        gt_feature = self.feature_extractor(gt_tensor)

        # Compute feature loss
        for i in range(len(self.feature_extractor_nodes)):
            losses.append(
                F_torch.mse_loss(
                    sr_feature[self.feature_extractor_nodes[i]],
                    gt_feature[self.feature_extractor_nodes[i]],
                )
            )

        losses = torch.Tensor([losses]).to(device)

        return losses


def ema_avg_fn(avg_model_param, model_param, num_avg, ema_decay=0.999):
    return (1 - ema_decay) * avg_model_param + ema_decay * model_param


# Define models
generator = SRResNet().to(device)
discriminator = Discriminator().to(device)
ema_g_model = AveragedModel(generator, device="cpu", avg_fn=ema_avg_fn)

# Define losses
pixel_criterion = nn.MSELoss().to(device)
content_criterion = ContentLoss(
    1000, "", ["features.35"], [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
).to(device)
adv_criterion = nn.BCEWithLogitsLoss().to(device)

# Optimizers and schedulers
g_opt = torch.optim.Adam(
    generator.parameters(), lr=1e-3, betas=[0.9, 0.999], eps=1e-8, weight_decay=0
)
d_opt = torch.optim.Adam(
    discriminator.parameters(), lr=1e-3, betas=[0.9, 0.999], eps=1e-8, weight_decay=0
)
g_step = torch.optim.lr_scheduler.MultiStepLR(g_opt, milestones=[5], gamma=0.5)
d_step = torch.optim.lr_scheduler.MultiStepLR(g_opt, milestones=[5], gamma=0.5)

# total_loss = pixel_loss + content_loss + 0.001 * adv_loss


def train_batch(
    images, targets, g_model, d_model, ema_model, g_opt, d_opt, g_step, d_step, device
):
    g_model.train()
    d_model.train()
    batch_size = images.size(0)
    real_label = torch.full([batch_size, 1], 1.0, dtype=torch.float32, device=device)
    fake_label = torch.full([batch_size, 1], 0.0, dtype=torch.float32, device=device)

    # Train generator
    g_opt.zero_grad()
    sr = g_model(images)
    pixel_loss = pixel_criterion(sr, targets).sum()
    feature_loss = content_criterion(sr, targets).sum()
    adv_loss = adv_criterion(d_model(sr), real_label).sum()
    g_loss = pixel_loss + feature_loss + 0.001 * adv_loss
    g_loss.backward()
    g_opt.step()

    # Train discriminator
    d_opt.zero_grad()
    gt_out = d_model(targets)
    d_loss_gt = adv_criterion(gt_out, real_label).sum()
    d_loss_gt.backward()

    sr_out = d_model(sr.detach())
    d_loss_sr = adv_criterion(sr_out, fake_label).sum()
    d_loss_sr.backward()
    d_opt.step()

    g_step.step()
    d_step.step()

    ema_model.update_parameters(g_model)
    return g_loss.item(), sr


def validate_batch(images, targets, g_model, d_model, device):
    g_model.eval()
    d_model.eval()
    batch_size = images.size(0)
    real_label = torch.full([batch_size, 1], 1.0, dtype=torch.float32, device=device)
    with torch.no_grad():
        sr = g_model(images)
        pixel_loss = pixel_criterion(sr, targets).sum()
        feature_loss = content_criterion(sr, targets).sum()
        adv_loss = adv_criterion(d_model(sr), real_label).sum()
        g_loss = pixel_loss + feature_loss + 0.001 * adv_loss

    return g_loss.item(), sr


def train_epoch(
    train_loader, val_loader, g_model, d_model, ema_model, g_opt, d_opt, g_step, d_step, device
):
    epoch_train_loss = epoch_val_loss = train_psnr = val_psnr = 0
    n_batches = len(train_loader) + len(val_loader)
    with tqdm(total=n_batches) as pbar:
        for image, targets in train_loader:
            image = image.to(device)
            targets = targets.to(device)
            g_loss, sr_images = train_batch(
                image,
                targets,
                g_model,
                d_model,
                ema_model,
                g_opt,
                d_opt,
                g_step,
                d_step,
                device,
            )
            train_psnr += psnr(sr_images, targets)
            epoch_train_loss += g_loss
            pbar.set_postfix({"train_loss": round(g_loss, 4)})
            pbar.update(1)
        for image, targets in val_loader:
            image = image.to(device)
            targets = targets.to(device)
            g_loss, sr_images = validate_batch(
                image,
                targets,
                g_model,
                d_model,
                device,
            )
            val_psnr += psnr(sr_images, targets)
            epoch_val_loss = g_loss
            pbar.set_postfix({"val_loss": round(g_loss, 4)})
            pbar.update(1)
    train_psnr /= len(train_loader)
    val_psnr /= len(val_loader)
    epoch_train_loss /= len(train_loader)
    epoch_val_loss /= len(val_loader)
    print(f"Train loss: {epoch_train_loss:.4f}, Val loss: {epoch_val_loss:.4f}",  # NOQA: T201
          f"Train PSNR: {train_psnr:.4f}, Val PSNR: {val_psnr:.4f}")
    return (train_psnr * len(train_loader) + val_psnr * len(val_loader)) / n_batches


if __name__ == "__main__":
    ds = EnhanceDataset("denoised/train/train/", "archive/train/gt/")
    loader = torch.utils.data.DataLoader(
        ds, batch_size=3, shuffle=True, collate_fn=collate_sr
    )
    val_ds = EnhanceDataset("denoised/val/val/", "archive/val/gt/")
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=3, shuffle=True, collate_fn=collate_sr
    )
    n_epochs = 10
    best_psnr = 0
    generator.load_state_dict(torch.load('generator-best-9.pth', weights_only=True))
    for epoch in range(n_epochs):
        current_psnr = train_epoch(
            loader,
            val_loader,
            generator,
            discriminator,
            ema_g_model,
            g_opt,
            d_opt,
            g_step,
            d_step,
            device,
        )
        if current_psnr > best_psnr:
            best_psnr = current_psnr
            torch.save(generator.state_dict(), f"generator-best-{epoch}.pth")
            torch.save(ema_g_model.state_dict(), f"ema-generator-best-{epoch}.pth")
            torch.save(discriminator.state_dict(), f"discriminator-best-{epoch}.pth")
            with open('best-psnr', 'w') as f:
                f.write(str(best_psnr))
