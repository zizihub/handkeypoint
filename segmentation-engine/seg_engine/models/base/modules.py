import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.base.modules import SCSEModule
from ..base.head import SegmentationHead, ClassificationHead
from .build import POSTPROCESS_REGISTRY, CONV2D_REGISTRY
from ...config import configurable


@POSTPROCESS_REGISTRY.register()
class SCSE(nn.Module):
    @configurable
    def __init__(self, encoder_channels, reduction) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(1, len(encoder_channels)):
            if encoder_channels[i] < reduction:
                self.blocks.append(nn.Identity())
            else:
                self.blocks.append(SCSEModule(in_channels=encoder_channels[i],
                                              reduction=reduction))

    @classmethod
    def from_config(cls, cfg, encoder_channels):
        return {
            'encoder_channels': encoder_channels,
            'reduction': cfg.MODEL.POSTPROCESS.REDUCTION,
        }

    def forward(self, features):
        out_features = [features[0]]
        for i in range(1, len(self.blocks)+1):
            x = self.blocks[i-1](features[i])
            out_features.append(x)
        return out_features


@CONV2D_REGISTRY.register()
class SeparableConv2d(nn.Sequential):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
            *args,
            **kwargs
    ):
        dephtwise_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        pointwise_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias,
        )
        act = kwargs.pop('activation', nn.Identity())
        norm = kwargs.pop('norm', nn.Identity())
        if norm is nn.BatchNorm2d:
            norm = norm(out_channels)
        elif norm is nn.GroupNorm:
            num_groups = 22
            assert out_channels % num_groups == 0
            norm = norm(num_groups, out_channels)
        super().__init__(dephtwise_conv, pointwise_conv, act, norm)


@CONV2D_REGISTRY.register()
class Conv2d(nn.Sequential):
    def __init__(self, in_chan, out_chan, kernel_size=3, stride=1, padding=0, *args, **kwargs):
        dilation = kwargs.pop('dilation', 1)
        groups = kwargs.pop('groups', 1)
        bias = kwargs.pop('bias', True)
        norm = kwargs.pop('norm', nn.Identity())
        act = kwargs.pop('activation', nn.Identity())
        conv = nn.Conv2d(in_chan,
                         out_chan,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=kernel_size//2,
                         dilation=dilation,
                         groups=groups,
                         bias=bias)
        if norm is nn.BatchNorm2d:
            norm = norm(out_chan)
        elif norm is nn.GroupNorm:
            num_groups = 22
            assert out_chan % num_groups == 0
            norm = norm(num_groups, out_chan)
        super().__init__(conv, norm, act)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = Conv2d(2, 1, kernel_size, stride=1, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale


@POSTPROCESS_REGISTRY.register()
class FeatureFusionModule(nn.Module):
    @configurable
    def __init__(self, channels, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        channels = channels + 48
        self.convblk = Conv2d(channels, channels, kernel_size=1, stride=1, padding=0,
                              norm=nn.BatchNorm2d, activation=nn.ReLU(inplace=True))
        self.conv1 = nn.Conv2d(channels,
                               channels//4,
                               kernel_size=1,
                               stride=1,
                               bias=False)
        self.conv2 = nn.Conv2d(channels//4,
                               channels,
                               kernel_size=1,
                               stride=1,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.init_weight()

    @classmethod
    def from_config(cls, cfg, encoder_channels):
        return {
            'channels': cfg.MODEL.DECODER.CHANNELS,
        }

    def forward(self, fcp, fsp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


@POSTPROCESS_REGISTRY.register()
class SelfRegulation(nn.Module):
    @configurable
    def __init__(self, highres_out_channels, decoder_out_channels, classes, activation, upsampling):
        super(SelfRegulation, self).__init__()
        self.sr_logit_student_head = ClassificationHead(
            in_channels=highres_out_channels,
            classes=classes,
            dropout=0.2,
            activation='sigmoid',
        )
        self.sr_logit_teacher_head = ClassificationHead(
            in_channels=decoder_out_channels,
            classes=classes,
            dropout=0.2,
            activation='sigmoid',
        )
        self.sr_feature_teacher_head = SegmentationHead(
            in_channels=highres_out_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )

    def forward(self, result):
        result.update(sr_logit_student=self.sr_logit_student_head(result['p2']))
        result.update(sr_logit_teacher=self.sr_logit_teacher_head(result['fused_features']))
        result.update(sr_feature_teacher=self.sr_feature_teacher_head(result['p2']))
        return result

    @classmethod
    def from_config(cls, cfg, encoder_channels):
        return {
            'highres_out_channels': 48,
            'decoder_out_channels': cfg.MODEL.DECODER.CHANNELS,
            'classes': cfg.DATASET.NUM_CLASSES+1,
            'activation': None,
            'upsampling': 4,
        }


class DetailAggregate(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, *args, **kwargs):
        super(DetailAggregate, self).__init__()
        self.conv = Conv2d(in_chan, mid_chan, ks=3, stride=1, padding=1,
                           norm=nn.BatchNorm2d, activation=nn.ReLU(inplace=True))
        self.conv_out = nn.Conv2d(
            mid_chan, n_classes, kernel_size=1, bias=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params
