import torch
import torch.nn as nn
from segmentation_models_pytorch.base.modules import Flatten, Activation


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = Activation(activation)
        super().__init__(conv2d, upsampling, activation)


class ClassificationHead(nn.Sequential):

    def __init__(self, in_channels, classes, pooling="avg", dropout=0.2, activation=None):
        if pooling not in ("max", "avg"):
            raise ValueError("Pooling should be one of ('max', 'avg'), got {}.".format(pooling))
        pool = nn.AdaptiveAvgPool2d(1) if pooling == 'avg' else nn.AdaptiveMaxPool2d(1)
        flatten = Flatten()
        dropout = nn.Dropout(p=dropout, inplace=True) if dropout else nn.Identity()
        linear = nn.Linear(in_channels, classes, bias=True)
        activation = Activation(activation)
        super().__init__(pool, flatten, dropout, linear, activation)


class MattingHead(nn.Module):

    def __init__(self, in_channels, src_channels, out_channels, kernel_size=3, upsampling=1):
        super(MattingHead, self).__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=upsampling)
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(out_channels + src_channels, out_channels, kernel_size, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )
        self.projection = nn.Conv2d(out_channels, 4, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, src, x):
        x = self.conv0(x)
        x = self.upsample(x)
        x = torch.cat([x, src], dim=1)
        x = self.conv1(x)
        fgr_residual, pha = self.projection(x).split([3, 1], dim=-3)
        fgr = fgr_residual + src
        fgr = fgr.clamp(0., 1.)
        pha = pha.clamp(0., 1.)

        return {
            'foreground': fgr,
            'alpha': pha
        }


class MattingHeadLight(nn.Module):

    def __init__(self, in_channels, src_channels, out_channels, kernel_size=3, upsampling=1):
        super(MattingHeadLight, self).__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=upsampling)
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(out_channels + src_channels, out_channels, kernel_size, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )
        self.projection = nn.Conv2d(out_channels, 4, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, src, x):
        x = self.conv0(x)
        x = self.upsample(x)
        x = torch.cat([x, src], dim=1)
        x = self.conv1(x)
        fgr_residual, pha = self.projection(x).split([3, 1], dim=-3)
        fgr = fgr_residual + src
        fgr = fgr.clamp(0., 1.)
        pha = pha.clamp(0., 1.)

        return {
            'foreground': fgr,
            'alpha': pha
        }
