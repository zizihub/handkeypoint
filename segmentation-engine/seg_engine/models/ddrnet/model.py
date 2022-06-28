from re import L
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from ..base import SegmentationHead, MattingHead, MattingHeadLight
from seg_engine.models.base import build_fpn, build_post, Conv2d, SeparableConv2d, DAPPM
from ..build import META_ARCH_REGISTRY, BACKBONE_REGISTRY
from ...config import configurable
from ..encoders.resnet import BasicBlock, Bottleneck
from ..base import InvertedResidual, CondConvResidual, DepthwiseSeparableConv, ConvBnAct


__all__ = ['DeepDualResolutionNet']


@META_ARCH_REGISTRY.register()
class DeepDualResolutionNet(nn.Module):
    '''DeepDualResolutionNet
    we propose novel deep dual-resolution networks (DDRNets) 
    for real-time semantic segmentation of road scenes. 
    Besides, we design a new contextual information extractor 
    named Deep Aggregation Pyramid Pooling Module (DAPPM) 
    to enlarge effective receptive fields and fuse multi-scale context.
    '''
    @configurable
    def __init__(self,
                 block,
                 layers,
                 model_name,
                 in_channels=3,
                 encoder_weights: Optional[str] = "imagenet",
                 classes=19,
                 planes=64,
                 spp_planes=128,
                 head_planes=128,
                 activation: Optional[str] = None,
                 upsampling: int = 16,
                 BatchNorm2d=nn.BatchNorm2d,
                 cfg=None,):
        super(DeepDualResolutionNet, self).__init__()

        highres_planes = planes * 2
        self.augment = cfg.MODEL.DECODER.AUX
        self.conv2d = eval(cfg.MODEL.CONV2D)
        self.model_name = model_name
        bottleneck = Bottleneck
        bottleneck.expansion = 2

        self.conv1 = nn.Sequential(
            self.conv2d(in_channels, planes, kernel_size=3, stride=2, padding=1,
                        norm=BatchNorm2d, activation=nn.ReLU(inplace=True)),
            self.conv2d(planes, planes, kernel_size=3, stride=2, padding=1,
                        norm=BatchNorm2d, activation=nn.ReLU(inplace=True)),
        )

        self.relu = nn.ReLU(inplace=False)

        self.layer1 = self._make_layer(block, planes, planes, layers[0])
        self.layer2 = self._make_layer(block, planes, planes * 2, layers[1], stride=2)
        if model_name == 'DDRNet39':
            self.layer3_1 = self._make_layer(block, planes * 2, planes * 4, layers[2] // 2, stride=2)
            self.layer3_2 = self._make_layer(block, planes * 4, planes * 4, layers[2] // 2)
        else:
            self.layer3 = self._make_layer(block, planes * 2, planes * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, planes * 4, planes * 8, layers[3], stride=2)

        if model_name == 'DDRNet39':
            self.compression3_1 = self.conv2d(planes * 4, highres_planes, kernel_size=1, bias=False, norm=BatchNorm2d)
            self.compression3_2 = self.conv2d(planes * 4, highres_planes, kernel_size=1, bias=False, norm=BatchNorm2d)
        else:
            self.compression3 = self.conv2d(planes * 4, highres_planes, kernel_size=1, bias=False, norm=BatchNorm2d)
        self.compression4 = self.conv2d(planes * 8, highres_planes, kernel_size=1, bias=False, norm=BatchNorm2d)

        if model_name == 'DDRNet39':
            self.down3_1 = self.conv2d(highres_planes, planes * 4, kernel_size=3,
                                       stride=2, padding=1, bias=False, norm=BatchNorm2d)
            self.down3_2 = self.conv2d(highres_planes, planes * 4, kernel_size=3,
                                       stride=2, padding=1, bias=False, norm=BatchNorm2d)
        else:
            self.down3 = self.conv2d(highres_planes, planes * 4, kernel_size=3,
                                     stride=2, padding=1, bias=False, norm=BatchNorm2d)
        self.down4 = nn.Sequential(
            self.conv2d(highres_planes, planes * 4, kernel_size=3, stride=2, padding=1,
                        bias=False, norm=BatchNorm2d, activation=nn.ReLU(inplace=True)),
            self.conv2d(planes * 4, planes * 8, kernel_size=3, stride=2, padding=1, bias=False, norm=BatchNorm2d)
        )
        if model_name == 'DDRNet39':
            self.layer3_1_ = self._make_layer(block, planes * 2, highres_planes, layers[2] // 2)
            self.layer3_2_ = self._make_layer(block, highres_planes, highres_planes, layers[2] // 2)
        else:
            self.layer3_ = self._make_layer(block, planes * 2, highres_planes, 2)
        self.layer4_ = self._make_layer(block, highres_planes, highres_planes, layers[3])
        self.layer5_ = self._make_layer(bottleneck, highres_planes, highres_planes, 1)
        self.layer5 = self._make_layer(bottleneck, planes * 8, planes * 8, 1, stride=2)

        self.spp = DAPPM(planes * 16, spp_planes, planes * 4)
        self.conv2 = self.conv2d(planes * 4, head_planes, kernel_size=3, padding=1, bias=False,
                                 norm=BatchNorm2d, activation=nn.ReLU(inplace=True))

        ############# Customized Function to DDRNet #############
        post = build_post(cfg, (3, planes*block.expansion, planes*2*block.expansion,
                                planes*4*block.expansion, planes*8*block.expansion,
                                planes * 8*bottleneck.expansion))
        self.point_head = post.pop('PointRend', None)
        if self.point_head:
            upsampling = 0
        scse = post.pop('SCSE', None)
        self.add_module('scse', scse)

        if self.augment:
            self.segmentation_head_extra = SegmentationHead(
                in_channels=highres_planes,
                out_channels=classes,
                activation=activation,
                kernel_size=1,
                upsampling=upsampling,
            )

        if 'Matting' in cfg.TASK:
            self.head = MattingHeadLight(
                in_channels=head_planes,
                src_channels=3,
                out_channels=8,
                kernel_size=3,
                upsampling=upsampling,
            )
            self.head_mode = 'mat'
        else:
            self.head = SegmentationHead(
                in_channels=head_planes,
                out_channels=classes,
                activation=activation,
                kernel_size=1,
                upsampling=upsampling,
            )
            self.head_mode = 'seg'
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if encoder_weights == 'imagenet':
            print('>>> loading pretrained weight for {}'.format(model_name))
            self.load_state_dict(torch.load(
                '/data/xuziyan/.cache/torch/hub/checkpoints/{}.pth'.format(model_name)), strict=False)
        self.deploy = False
        self.name = model_name

    def load_state_dict(self, state_dict, **kwargs):
        super().load_state_dict(state_dict, **kwargs)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if (stride != 1 or inplanes != planes * block.expansion) and issubclass(block, (Bottleneck, BasicBlock)):
            downsample = self.conv2d(inplanes, planes * block.expansion, kernel_size=1,
                                     stride=stride, bias=False, norm=nn.BatchNorm2d)

        layers = []
        layers.append(block(inplanes, planes, stride=stride, downsample=downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == (blocks-1):
                layers.append(block(inplanes, planes, stride=1, no_relu=True))
            else:
                layers.append(block(inplanes, planes, stride=1, no_relu=False))

        return nn.Sequential(*layers)

    def forward(self, x):
        raw_x = x.clone()
        result = {'features': []}
        width_output = x.shape[-1] // 8
        height_output = x.shape[-2] // 8

        x = self.conv1(x)
        result['features'].append(x)
        x = self.layer1(x)
        if self.scse:
            x = self.scse.blocks[0](x)
        result['features'].append(x)

        x = self.layer2(self.relu(x))
        if self.scse:
            x = self.scse.blocks[1](x)
        result['features'].append(x)

        if self.model_name == 'DDRNet39':
            x = self.layer3_1(self.relu(x))
            result['features'].append(x)
            x_ = self.layer3_1_(self.relu(result['features'][2]))
            x = x + self.down3_1(self.relu(x_))
            x_ = x_ + F.interpolate(
                self.compression3_1(self.relu(result['features'][3])),
                size=[height_output, width_output],
                mode='bilinear')
            x = self.layer3_2(self.relu(x))
            result['features'].append(x)
            x_ = self.layer3_2_(self.relu(x_))
            x = x + self.down3_2(self.relu(x_))
            x_ = x_ + F.interpolate(
                self.compression3_2(self.relu(result['features'][4])),
                size=[height_output, width_output],
                mode='bilinear')
            if self.augment:
                temp = x_
        else:
            x = self.layer3(self.relu(x))
            if self.scse:
                x = self.scse.blocks[2](x)
            result['features'].append(x)
            x_ = self.layer3_(self.relu(result['features'][2]))
            x = x + self.down3(self.relu(x_))
            x_ = x_ + F.interpolate(
                self.compression3(self.relu(result['features'][3])),
                size=[height_output, width_output],
                mode='bilinear')
            if self.augment:
                temp = x_

        x = self.layer4(self.relu(x))
        if self.scse:
            x = self.scse.blocks[3](x)
        result['features'].append(x)
        x_ = self.layer4_(self.relu(x_))

        x = x + self.down4(self.relu(x_))
        x_ = x_ + F.interpolate(
            self.compression4(self.relu(result['features'][4 if not self.model_name == 'DDRNet39' else 5])),
            size=[height_output, width_output],
            mode='bilinear')

        x = self.layer5(self.relu(x))
        if self.scse:
            x = self.scse.blocks[4](x)
        result['features'].append(x)
        x_ = self.layer5_(self.relu(x_))
        x = F.interpolate(
            self.spp(x),
            size=[height_output, width_output],
            mode='bilinear')
        x = self.conv2(x + x_)
        result.update(fused_features=x)
        if self.head_mode == 'seg':
            # Segmentation
            result.update(masks=self.head(x))
        elif self.head_mode == 'mat':
            # Matting
            result.update(self.head(raw_x, x))
        else:
            raise NotImplementedError
        if self.augment and self.training:
            x_extra = self.segmentation_head_extra(temp)
            result.update(auxiliary=x_extra)
        if self.point_head:
            result.update(self.point_head(raw_x, result))
        if self.deploy:
            result.pop('features')
            result.pop('fused_features')
        return result

    @classmethod
    def from_config(cls, cfg, backbone):
        backbone.update(
            in_channels=4 if cfg.DATASET.POSTPROCESS.four_channel else 3,
            classes=cfg.DATASET.NUM_CLASSES+1,
            model_name=cfg.MODEL.ENCODER.NAME,
            encoder_weights=cfg.MODEL.PRETRAINED,
            cfg=cfg,
        )
        return backbone


@BACKBONE_REGISTRY.register()
def DDRNet15_slim(cfg, block):
    return {
        'block': block,
        'layers': [1, 1, 1, 1],
        'planes': 32,
        'spp_planes': cfg.MODEL.DECODER.CHANNELS[0],  # default 128
        'activation': None,
        'upsampling': 8,  # default 8
    }


@BACKBONE_REGISTRY.register()
def DDRNet23_slim(cfg, block):
    return {
        'block': block,
        'layers': [2, 2, 2, 2],
        'planes': 32,
        'spp_planes': cfg.MODEL.DECODER.CHANNELS[0],  # default 128
        'activation': None,
        'upsampling': 8,  # default 8
    }


@BACKBONE_REGISTRY.register()
def DDRNet23(cfg, block):
    return {
        'block': block,
        'layers': [2, 2, 2, 2],
        'planes': 64,
        'spp_planes': cfg.MODEL.DECODER.CHANNELS[0],   # default 128
        'activation': None,
        'upsampling': 8,  # default 8
    }


@BACKBONE_REGISTRY.register()
def DDRNet39(cfg, block):
    return {
        'block': block,
        'layers': [3, 4, 6, 3],
        'planes': 64,
        'spp_planes': cfg.MODEL.DECODER.CHANNELS[0],   # default 128
        'activation': None,
        'upsampling': 8,  # default 8
    }
