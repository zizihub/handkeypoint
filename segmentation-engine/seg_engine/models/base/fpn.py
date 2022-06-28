import torch
import torch.nn as nn
from .modules import Conv2d
import torch.nn.functional as F
import math
from .build import FPN_REGISTRY
from ...config import configurable


def xavier_init(module, gain=1, bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.xavier_uniform_(module.weight, gain=gain)
    else:
        nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is True:
        nn.init.constant_(module.bias, bias)


@FPN_REGISTRY.register()
class FPN(nn.Module):
    @configurable
    def __init__(self,
                 in_channels,
                 out_channels,
                 norm=None,
                 panet_buttomup=False):
        super(FPN, self).__init__()
        lateral_convs = []
        output_convs = []
        strides = (4, 8, 16, 16)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.panet_buttomup = panet_buttomup
        if self.panet_buttomup:
            self.panet_bottomup_conv1s = []
            self.panet_bottomup_conv2s = []
        use_bias = norm is None
        for idx, in_c in enumerate(in_channels):
            lateral_conv = Conv2d(in_chan=in_c,
                                  out_chan=out_channels,
                                  kernel_size=1,
                                  bias=use_bias,
                                  norm=norm,)
            output_conv = Conv2d(in_chan=out_channels,
                                 out_chan=out_channels,
                                 padding=1,
                                 bias=use_bias,
                                 norm=norm)
            stage = int(math.log2(strides[idx]))
            self.add_module('fpn_lateral{}'.format(stage), lateral_conv)
            self.add_module('fpn_output{}'.format(stage), output_conv)
            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
            if self.panet_buttomup:
                panet_conv1 = Conv2d(in_chan=out_channels,
                                     out_chan=out_channels,
                                     kernel_size=3,
                                     stride=2 if stage < 4 else 1,
                                     bias=True,
                                     norm=norm,
                                     activation=nn.ReLU(inplace=True),
                                     )
                panet_conv2 = Conv2d(in_chan=out_channels,
                                     out_chan=out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     bias=True,
                                     norm=norm,
                                     activation=nn.ReLU(inplace=True),
                                     )
                self.add_module('panet_buttomup_conv1_{}'.format(stage), panet_conv1)
                self.add_module('panet_buttomup_conv2_{}'.format(stage), panet_conv2)
                self.panet_bottomup_conv1s.append(panet_conv1)
                self.panet_bottomup_conv2s.append(panet_conv2)

        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

    @classmethod
    def from_config(cls, cfg, in_channels, conv2d):
        return {
            'in_channels': in_channels[:-4],                # s4,s8,s16,s16
            'out_channels': cfg.MODEL.FPN.OUT_CHANNELS,
            'norm': None,
            'panet_buttomup': cfg.MODEL.PANET,
        }

    def forward(self, features):
        assert len(features) == len(self.in_channels)
        out_features = []
        prev_feature = self.lateral_convs[0](features[-1])
        out_features.append(self.output_convs[0](prev_feature))
        for idx, (lateral_conv, output_conv) in enumerate(zip(self.lateral_convs, self.output_convs)):
            if idx > 0:
                feature = features[-idx-1]
                lateral_feature = lateral_conv(feature)
                if lateral_feature.shape != prev_feature.shape:
                    top_down_feature = F.interpolate(prev_feature, scale_factor=2.0, mode='nearest')
                else:
                    top_down_feature = prev_feature
                prev_feature = lateral_feature + top_down_feature
                out_features.insert(0, output_conv(prev_feature))
        if self.panet_buttomup:
            panet_out_features = [out_features[0]]
            bottom_up_feature = self.panet_bottomup_conv1s[0](out_features[0])
            for idx, (panet_conv1, panet_conv2) in enumerate(zip(self.panet_bottomup_conv1s, self.panet_bottomup_conv2s)):
                if idx > 0:
                    prev_feature = out_features[idx]
                    lateral_feature = bottom_up_feature + prev_feature
                    bottom_up_feature = panet_conv2(lateral_feature)
                    panet_out_features.append(bottom_up_feature)
                    bottom_up_feature = panet_conv1(bottom_up_feature)
            out_features = panet_out_features

        return out_features


@FPN_REGISTRY.register()
class BIFPN(nn.Module):
    @configurable
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs=4,
                 start_level=0,
                 end_level=-1,
                 stack=1,
                 conv2d=Conv2d,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False):
        super(BIFPN, self).__init__()
        assert isinstance(in_channels, tuple)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.stack = stack

        Custom_Conv2d = conv2d

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.stack_bifpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = Custom_Conv2d(
                in_channels[i],
                out_channels,
                kernel_size=1,
                norm=nn.BatchNorm2d,
                activation=nn.ReLU(inplace=True))
            self.lateral_convs.append(l_conv)
        for ii in range(stack):
            self.stack_bifpn_convs.append(BiFPNModule(conv2d=Custom_Conv2d,
                                                      channels=out_channels,
                                                      levels=self.backbone_end_level-self.start_level))
        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = Custom_Conv2d(
                    in_channels,
                    out_channels,
                    stride=2,
                    padding=1,
                    norm=nn.BatchNorm2d,
                    activation=nn.ReLU(inplace=True))
                self.fpn_convs.append(extra_fpn_conv)
        self.init_weights()

    @classmethod
    def from_config(cls, cfg, in_channels, conv2d):
        return {
            'in_channels': in_channels[-4:],
            'out_channels': cfg.MODEL.FPN.OUT_CHANNELS,
            'num_outs': 4,
            'start_level': 0,
            'end_level': -1,
            'stack': cfg.MODEL.FPN.STACK,
            'conv2d': conv2d,
            'add_extra_convs': False,
            'extra_convs_on_inputs': True,
            'relu_before_extra_convs': False,
            'no_norm_on_lateral': False,
        }

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # part 1: build top-down and down-top path with stack
        used_backbone_levels = len(laterals)
        for bifpn_module in self.stack_bifpn_convs:
            laterals = bifpn_module(laterals)
        outs = laterals
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[0](orig))
                else:
                    outs.append(self.fpn_convs[0](outs[-1]))
                for i in range(1, self.num_outs - used_backbone_levels):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)


class BiFPNModule(nn.Module):
    def __init__(self,
                 conv2d,
                 channels,
                 levels,
                 init=0.5,
                 eps=0.0001):
        super(BiFPNModule, self).__init__()
        self.eps = eps
        self.levels = levels
        self.bifpn_convs = nn.ModuleList()
        # weighted
        self.w1 = nn.Parameter(torch.Tensor(2, levels).fill_(init))
        self.relu1 = nn.ReLU()
        self.w2 = nn.Parameter(torch.Tensor(3, levels - 2).fill_(init))
        self.relu2 = nn.ReLU()
        for jj in range(2):
            for i in range(self.levels-1):  # 1,2,3
                fpn_conv = nn.Sequential(
                    conv2d(
                        channels,
                        channels,
                        padding=1,
                        norm=nn.BatchNorm2d,
                        activation=nn.ReLU(inplace=True))
                )
                self.bifpn_convs.append(fpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == self.levels
        # build top-down and down-top path with stack
        levels = self.levels
        # w relu
        w1 = self.relu1(self.w1)
        w1 /= torch.sum(w1, dim=0) + self.eps  # normalize
        w2 = self.relu2(self.w2)
        w2 /= torch.sum(w2, dim=0) + self.eps  # normalize
        # build top-down
        idx_bifpn = 0
        pathtd = inputs
        inputs_clone = []
        for in_tensor in inputs:
            inputs_clone.append(in_tensor.clone())

        for i in range(levels - 1, 0, -1):
            if pathtd[i].size(-1) != pathtd[i - 1].size(-1):
                temp = F.interpolate(pathtd[i],
                                     scale_factor=pathtd[i-1].size(-1) // pathtd[i].size(-1),
                                     mode='nearest')
            else:
                temp = pathtd[i]
            pathtd[i - 1] = (w1[0, i-1]*pathtd[i - 1] + w1[1, i-1]*temp)/(w1[0, i-1] + w1[1, i-1] + self.eps)
            pathtd[i - 1] = self.bifpn_convs[idx_bifpn](pathtd[i - 1])
            idx_bifpn = idx_bifpn + 1
            del temp
        # build down-top
        for i in range(0, levels - 2, 1):
            # dynamic for atrous different output strides
            if pathtd[i+1].size(-1) != pathtd[i].size(-1):
                temp = F.max_pool2d(pathtd[i], kernel_size=2)
            else:
                temp = pathtd[i]
            pathtd[i + 1] = (w2[0, i] * pathtd[i + 1] + w2[1, i] * temp +
                             w2[2, i] * inputs_clone[i + 1])/(w2[0, i] + w2[1, i] + w2[2, i] + self.eps)
            pathtd[i + 1] = self.bifpn_convs[idx_bifpn](pathtd[i + 1])
            idx_bifpn = idx_bifpn + 1
            del temp

        pathtd[levels - 1] = (w1[0, levels-1] * pathtd[levels - 1] + w1[1, levels-1] *
                              pathtd[levels - 2])/(w1[0, levels-1] + w1[1, levels-1] + self.eps)
        pathtd[levels - 1] = self.bifpn_convs[idx_bifpn](pathtd[levels - 1])
        return pathtd
