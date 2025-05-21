import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops

class DeforambleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        """
        Initialize the DeforambleConv module.
        """
        super(DeforambleConv, self).__init__()
        self.conv_offset = nn.Conv2d(in_channels, 18, kernel_size=kernel_size, padding=padding)
        self.conv = ops.DeformConv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        """
        Define the forward pass of the DeforambleConv module.
        """
        offset = self.conv_offset(x)
        return self.conv(x, offset)

# Double convolution module
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dcn=False):
        """
        Initialize the DoubleConv module.
        """
        super(DoubleConv, self).__init__()
        if dcn:
            self.conv = nn.Sequential(
                DeforambleConv(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                DeforambleConv(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:    
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        """
        Define the forward pass of the DoubleConv module.
        """
        return self.conv(x)

# Downsampling module
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Initialize the Down module.
        """
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        """
        Define the forward pass of the Down module.
        """
        return self.maxpool_conv(x)

# Upsampling module
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        """
        Initialize the Up module.
        """
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.post_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            self.post_conv = None

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        Define the forward pass of the Up module.
        """
        x1 = self.up(x1)
        if self.post_conv is not None:
            x1 = self.post_conv(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# Output convolution module
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Initialize the OutConv module.
        """
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Define the forward pass of the OutConv module.
        """
        x = self.conv(x)
        return x