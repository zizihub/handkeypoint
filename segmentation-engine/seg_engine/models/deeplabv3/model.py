import torch
import torch.nn as nn
from seg_engine.models.base import build_fpn, build_post, build_spp, Conv2d, SeparableConv2d
from typing import Optional
from ..base import SegmentationHead, ClassificationHead, MattingHead, MattingHeadLight
from segmentation_models_pytorch.base import SegmentationModel
from ..build import META_ARCH_REGISTRY
from ...config import CfgNode, configurable
from ..encoders import get_encoder
from .decoder import DeepLabV3Decoder, DeepLabV3PlusDecoder

__all__ = ['DeepLabV3', 'DeepLabV3Plus']


@META_ARCH_REGISTRY.register()
class DeepLabV3(SegmentationModel):
    """DeepLabV3_ implemetation from "Rethinking Atrous Convolution for Semantic Image Segmentation"

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features
            two times smaller in spatial dimentions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and
            other pretrained weights (see table with available weights for each encoder_name)
        decoder_channels: A number of convolution filters in ASPP module. Default is 256
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Avaliable options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"identity"**, **callable** and **None**.
            Default is **None**
        upsampling: Final upsampling factor. Default is 8 to preserve input-output spatial shape identity
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax" (could be **None** to return logits)
    Returns:
        ``torch.nn.Module``: **DeepLabV3**

    .. _DeeplabV3:
        https://arxiv.org/abs/1706.05587

    """
    @configurable
    def __init__(
            self,
            encoder_name: str = "resnet34",
            encoder_depth: int = 5,
            encoder_weights: Optional[str] = "imagenet",
            decoder_channels: int = 256,
            in_channels: int = 3,
            classes: int = 1,
            activation: Optional[str] = None,
            upsampling: int = 8,
            aux_params: Optional[dict] = None,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )
        self.encoder.make_dilated(
            stage_list=[4, 5],
            dilation_list=[2, 4]
        )

        self.decoder = DeepLabV3Decoder(
            in_channels=self.encoder.out_channels[-1],
            out_channels=decoder_channels,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None


@META_ARCH_REGISTRY.register()
class DeepLabV3Plus(SegmentationModel):
    """DeepLabV3+ implemetation from "Encoder-Decoder with Atrous Separable
    Convolution for Semantic Image Segmentation"

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features
            two times smaller in spatial dimentions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and
            other pretrained weights (see table with available weights for each encoder_name)
        encoder_output_stride: Downsampling factor for last encoder features (see original paper for explanation)
        decoder_atrous_rates: Dilation rates for ASPP module (should be a tuple of 3 integer values)
        decoder_channels: A number of convolution filters in ASPP module. Default is 256
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Avaliable options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"identity"**, **callable** and **None**.
            Default is **None**
        upsampling: Final upsampling factor. Default is 4 to preserve input-output spatial shape identity
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax" (could be **None** to return logits)
    Returns:
        ``torch.nn.Module``: **DeepLabV3Plus**

    Reference:
        https://arxiv.org/abs/1802.02611v3

    """
    @configurable
    def __init__(
            self,
            encoder_name: str = "resnet34",
            encoder_depth: int = 5,
            encoder_weights: Optional[str] = "imagenet",
            encoder_output_stride: int = 16,
            decoder_channels: int = 256,
            decoder_atrous_rates: tuple = (3, 6, 9),
            in_channels: int = 3,
            classes: int = 1,
            activation: Optional[str] = None,
            upsampling: int = 4,
            aux_params: Optional[dict] = None,
            cfg: CfgNode = None,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        if encoder_output_stride == 8:
            self.encoder.make_dilated(
                stage_list=[4, 5],
                dilation_list=[2, 4]
            )

        elif encoder_output_stride == 16:
            self.encoder.make_dilated(
                stage_list=[5],
                dilation_list=[2]
            )
        else:
            raise ValueError(
                "Encoder output stride should be 8 or 16, got {}".format(
                    encoder_output_stride)
            )

        if aux_params:
            self.segmentation_head_aux = nn.Sequential(
                Conv2d(in_chan=self.encoder.out_channels[-2],
                       out_chan=self.encoder.out_channels[-2],
                       padding=1,
                       norm=nn.BatchNorm2d,
                       activation=nn.ReLU(),),
                SegmentationHead(
                    in_channels=self.encoder.out_channels[-2],
                    out_channels=classes,
                    activation=activation,
                    kernel_size=1,
                    upsampling=encoder_output_stride,
                )
            )
        else:
            self.segmentation_head_aux = None
        self.post = nn.Module()
        post = build_post(cfg, self.encoder.out_channels)
        self.post.add_module('point_head', post.pop('PointRend', None))
        if self.post.point_head:
            upsampling = 0
        scse = post.pop('SCSE', None)
        fpn = build_fpn(cfg, self.encoder.out_channels)
        kwargs = post.pop('Decoupled', {})
        ffm = post.pop('FeatureFusionModule', None)
        kwargs.update(spp=build_spp(cfg))

        if fpn:
            self.encoder._out_channels = (cfg.MODEL.FPN.OUT_CHANNELS,)*6
        self.decoder = DeepLabV3PlusDecoder(
            encoder_channels=self.encoder.out_channels,
            out_channels=decoder_channels,
            atrous_rates=decoder_atrous_rates,
            output_stride=encoder_output_stride,
            **kwargs
        )

        self.decoder.add_module('scse', scse)
        self.decoder.add_module('fpn', fpn)
        self.decoder.add_module('ffm', ffm)

        self.post.add_module('self_regulation', post.pop('SelfRegulation', None))
        if 'Matting' in cfg.TASK:
            self.head = MattingHeadLight(
                in_channels=self.decoder.out_channels,
                src_channels=3,
                out_channels=8,
                kernel_size=3,
                upsampling=upsampling,
            )
            self.head_mode = 'mat'
        else:
            self.head = SegmentationHead(
                in_channels=self.decoder.out_channels,
                out_channels=classes,
                activation=activation,
                kernel_size=1,
                upsampling=upsampling,
            )
            self.head_mode = 'seg'

        self.deploy = False
        self.name = "deeplabv3p-{}".format(encoder_name)

    @classmethod
    def from_config(cls, cfg, block):
        return {
            'encoder_name': cfg.MODEL.ENCODER.NAME,
            'encoder_depth': 5,
            'encoder_weights': cfg.MODEL.PRETRAINED,
            'encoder_output_stride': cfg.MODEL.ENCODER.OUTPUT_STRIDE,
            'decoder_channels': cfg.MODEL.DECODER.CHANNELS[0],
            'decoder_atrous_rates': cfg.MODEL.DECODER.ASTROUS_RATES,
            'in_channels': 4 if cfg.DATASET.POSTPROCESS.four_channel else 3,
            'classes': cfg.DATASET.NUM_CLASSES+1,
            'activation': None,
            'upsampling': 4,
            'aux_params': cfg.MODEL.DECODER.AUX,
            'cfg': cfg,
        }

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        result = {}
        # encoder
        result.update(features=self.encoder(x))
        # decoder
        result.update(self.decoder(*result['features']))
        if self.head_mode == 'seg':
            # Segmentation
            result.update(masks=self.head(result['fused_features']))
        elif self.head_mode == 'mat':
            # Matting
            result.update(self.head(x, result['fused_features']))
        else:
            raise NotImplementedError
        # point render
        if self.post.point_head:
            result.update(self.post.point_head(x, result))
        # classification head
        if self.segmentation_head_aux is not None:
            print(result['features'][-2].shape)
            result.update(auxiliary=self.segmentation_head_aux(result['features'][-2]))
        # Self Regulation
        if self.post.self_regulation:
            result.update(self.post.self_regulation(result))
        if self.deploy:
            result.pop('features')
            result.pop('fused_features')
        return result
