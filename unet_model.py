import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    Double convolution block:
    Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net для segmentation мікрочастинок SEM
    """

    def __init__(self):
        super().__init__()

        # =========================
        # ENCODER
        # =========================

        self.down1 = DoubleConv(1, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        # =========================
        # BOTTLENECK
        # =========================

        self.middle = DoubleConv(64, 128)

        # =========================
        # DECODER
        # =========================

        self.up2 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.upconv2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        self.upconv1 = DoubleConv(64, 32)

        # =========================
        # OUTPUT
        # =========================

        self.out = nn.Conv2d(
            32,
            1,
            kernel_size=1
        )

    def forward(self, x):

        # =========================
        # ENCODER
        # =========================

        d1 = self.down1(x)

        p1 = self.pool1(d1)

        d2 = self.down2(p1)

        p2 = self.pool2(d2)

        # =========================
        # BOTTLENECK
        # =========================

        m = self.middle(p2)

        # =========================
        # DECODER
        # =========================

        u2 = self.up2(m)

        u2 = torch.cat([u2, d2], dim=1)

        u2 = self.upconv2(u2)

        u1 = self.up1(u2)

        u1 = torch.cat([u1, d1], dim=1)

        u1 = self.upconv1(u1)

        # =========================
        # OUTPUT LOGITS
        # IMPORTANT:
        # NO SIGMOID HERE
        # =========================

        return self.out(u1)