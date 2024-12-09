import torch
import torch.nn as nn


def double_conv(in_ch, out_ch):
    """Two convolutional layers with ReLU activation."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = double_conv(in_channels, 64)
        self.enc2 = double_conv(64, 128)
        self.enc3 = double_conv(128, 256)
        self.enc4 = double_conv(256, 512)

        # Bottleneck
        self.bottleneck = double_conv(512, 1024)

        # Decoder
        self.up3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec3 = double_conv(1024, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = double_conv(512, 256)

        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1 = double_conv(256, 128)

        self.up0 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec0 = double_conv(128, 64)

        # Final output layer
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)  # (64, 256, 256)
        p1 = self.pool(x1)  # (64, 128, 128)

        x2 = self.enc2(p1)  # (128, 128, 128)
        p2 = self.pool(x2)  # (128, 64, 64)

        x3 = self.enc3(p2)  # (256, 64, 64)
        p3 = self.pool(x3)  # (256, 32, 32)

        x4 = self.enc4(p3)  # (512, 32, 32)
        p4 = self.pool(x4)  # (512, 16, 16)

        # Bottleneck
        bottleneck = self.bottleneck(p4)  # (1024, 16, 16)

        # Decoder
        up3 = self.up3(bottleneck)  # (512, 32, 32)
        dec3 = self.dec3(torch.cat([up3, x4], dim=1))  # Concatenate along channel dimension

        up2 = self.up2(dec3)  # (256, 64, 64)
        dec2 = self.dec2(torch.cat([up2, x3], dim=1))  # Concatenate along channel dimension

        up1 = self.up1(dec2)  # (128, 128, 128)
        dec1 = self.dec1(torch.cat([up1, x2], dim=1))  # Concatenate along channel dimension

        up0 = self.up0(dec1)  # (64, 256, 256)
        dec0 = self.dec0(torch.cat([up0, x1], dim=1))  # Concatenate along channel dimension

        # Final output
        return self.final(dec0)
