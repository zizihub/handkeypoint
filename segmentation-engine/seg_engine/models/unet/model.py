from typing import Optional, Union, List
from .decoder import UnetDecoder
from ..encoders import get_encoder
from ..base import SegmentationHead, ClassificationHead, MattingHead, DetailAggregate
from segmentation_models_pytorch.base import SegmentationModel
from ..build import META_ARCH_REGISTRY
from ...config import CfgNode, configurable
from torch import nn


@META_ARCH_REGISTRY.register()
class Unet(SegmentationModel):
    """Unet_ is a fully convolution neural network for image semantic segmentation. Consist of *encoder* 
    and *decoder* parts connected with *skip connections*. Encoder extract features of different spatial 
    resolution (skip connections) which are used by decoder to define accurate segmentation mask. Use *concatenation*
    for fusing decoder blocks with skip connections.

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features 
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and 
            other pretrained weights (see table with available weights for each encoder_name)
        decoder_channels: List of integers which specify **in_channels** parameter for convolutions used in decoder.
            Length of the list should be the same as **encoder_depth**
        decoder_use_batchnorm: If **True**, BatchNorm2d layer between Conv2D and Activation layers
            is used. If **"inplace"** InplaceABN will be used, allows to decrease memory consumption.
            Available options are **True, False, "inplace"**
        decoder_attention_type: Attention module used in decoder of the model. Available options are **None** and **scse**.
            SCSE paper - https://arxiv.org/abs/1808.08127
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
            Default is **None**
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build 
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax" (could be **None** to return logits)

    Returns:
        ``torch.nn.Module``: Unet

    .. _Unet:
        https://arxiv.org/abs/1505.04597

    """
    @configurable
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
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

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        if 'Matting' in cfg.TASK:
            self.head = MattingHead(
                in_channels=decoder_channels[-1],
                src_channels=3,
                out_channels=16,
                kernel_size=3,
            )
            self.head_mode = 'mat'
        else:
            self.head = SegmentationHead(
                in_channels=decoder_channels[-1],
                out_channels=classes,
                activation=activation,
                kernel_size=3,
            )
            self.head_mode = 'seg'
        self.post = nn.Module()
        # detail aggregation
        if self.training and 'DetailAggregateLoss' in cfg.SOLVER.LOSS:
            self.post.add_module('conv_out_sp2',
                                 DetailAggregate(
                                     in_chan=self.encoder.out_channels[2],
                                     mid_chan=64,
                                     n_classes=1,
                                 ))
            self.post.add_module('conv_out_sp4',
                                 DetailAggregate(
                                     in_chan=self.encoder.out_channels[3],
                                     mid_chan=64,
                                     n_classes=1,
                                 ))
            self.post.add_module('conv_out_sp8',
                                 DetailAggregate(
                                     in_chan=self.encoder.out_channels[4],
                                     mid_chan=64,
                                     n_classes=1,
                                 ))
        self.deploy = False
        self.name = "u-{}".format(encoder_name)

    @classmethod
    def from_config(cls, cfg, block):
        return {
            'encoder_name': cfg.MODEL.ENCODER.NAME,
            'encoder_depth': 5,
            'encoder_weights': cfg.MODEL.PRETRAINED,
            'decoder_use_batchnorm': True,
            'decoder_channels': cfg.MODEL.DECODER.CHANNELS,
            'decoder_attention_type': 'scse' if 'SCSE' in cfg.MODEL.POSTPROCESS.NAME else None,
            'in_channels': 4 if cfg.DATASET.POSTPROCESS.four_channel else 3,
            'classes': cfg.DATASET.NUM_CLASSES+1,
            'activation': None,
            'aux_params': cfg.MODEL.DECODER.AUX,
            'cfg': cfg,
        }

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        H, W = x.size()[2:]
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
        if self.training and hasattr(self.post, 'conv_out_sp2'):
            feat_out_sp2 = self.post.conv_out_sp2(result['features'][2])
            feat_out_sp4 = self.post.conv_out_sp4(result['features'][3])
            feat_out_sp8 = self.post.conv_out_sp8(result['features'][4])
            result.update({
                'boundary_2': feat_out_sp2,
                'boundary_4': feat_out_sp4,
                'boundary_8': feat_out_sp8
            })
        if self.deploy:
            result.pop('features')
            result.pop('fused_features')
        return result
