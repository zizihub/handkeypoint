from timm import create_model
from timm.models.mobilenetv3 import MobileNetV3Features
from timm.models.efficientnet_blocks import SqueezeExcite
from timm.models.efficientnet_builder import decode_arch_def, round_channels
import torch.nn as nn
from ._base import EncoderMixin
from functools import partial


def make_divisible(v, divisor=8, min_value=None, round_limit=.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


class MobileNetV3Encoder(nn.Module, EncoderMixin):
    def __init__(self, model, width_mult, depth=5, **kwargs):
        super().__init__()
        self._depth = depth
        if 'small' in str(model):
            self.mode = 'small'
            _out_channels = (16*width_mult, 16*width_mult, 24*width_mult, 48*width_mult, 576*width_mult)
            self._out_channels = tuple(map(make_divisible, _out_channels))
        elif 'large' in str(model):
            self.mode = 'large'
            _out_channels = (16*width_mult, 24*width_mult, 40*width_mult, 112*width_mult, 960*width_mult)
            self._out_channels = tuple(map(make_divisible, _out_channels))
        else:
            self.mode = 'None'
            raise ValueError(
                'MobileNetV3 mode should be small or large, got {}'.format(self.mode))
        self._out_channels = (3,) + self._out_channels
        self._in_channels = 3
        se_layer = partial(SqueezeExcite, gate_layer='hard_sigmoid',
                           force_act_layer=nn.ReLU, rd_round_fn=round_channels)
        try:
            model = create_model(model_name=model,
                                 scriptable=True,   # torch.jit scriptable
                                 exportable=True,   # onnx export
                                 features_only=True)
        except:
            arch_def = kwargs.pop('arch_def', None)
            model = MobileNetV3Features(block_args=decode_arch_def(arch_def),
                                        round_chs_fn=partial(round_channels, multiplier=width_mult),
                                        se_layer=se_layer)
        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1
        self.act1 = model.act1
        self.blocks = model.blocks

    def get_stages(self):
        if self.mode == 'small':
            return [
                nn.Identity(),
                nn.Sequential(self.conv_stem, self.bn1, self.act1),
                self.blocks[0],
                self.blocks[1],
                self.blocks[2:4],
                self.blocks[4:],
            ]
        elif self.mode == 'large':
            return [
                nn.Identity(),
                nn.Sequential(self.conv_stem, self.bn1, self.act1, self.blocks[0]),
                self.blocks[1],
                self.blocks[2],
                self.blocks[3:5],
                self.blocks[5:],
            ]
        else:
            ValueError('MobileNetV3 mode should be small or large, got {}'.format(self.mode))

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)
        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop('conv_head.weight')
        state_dict.pop('conv_head.bias')
        state_dict.pop('classifier.weight')
        state_dict.pop('classifier.bias')
        super().load_state_dict(state_dict, **kwargs)


mobilenetv3_weights = {
    'tf_mobilenetv3_large_075': {
        'imagenet': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_large_075-150ee8b0.pth'
    },
    'tf_mobilenetv3_large_100': {
        'imagenet': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_large_100-427764d5.pth'
    },
    'tf_mobilenetv3_large_minimal_100': {
        'imagenet': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_large_minimal_100-8596ae28.pth'
    },
    'tf_mobilenetv3_small_025': {
        'imagenet': '/data/zhangziwei/.cache/torch/hub/checkpoints/mobilenetv3_small_025.pth'
    },
    'tf_mobilenetv3_small_075': {
        'imagenet': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_small_075-da427f52.pth'
    },
    'tf_mobilenetv3_small_100': {
        'imagenet': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_small_100-37f49e2b.pth'
    },
    'tf_mobilenetv3_small_minimal_010': {
        'imagenet': ''
    },
    'tf_mobilenetv3_small_minimal_025': {
        'imagenet': '/data/zhangziwei/.cache/torch/hub/checkpoints/mobilenetv3_small_minimal_025.pth'
    },
    'tf_mobilenetv3_small_minimal_100': {
        'imagenet': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_small_minimal_100-922a7843.pth'
    },


}

pretrained_settings = {}
for model_name, sources in mobilenetv3_weights.items():
    pretrained_settings[model_name] = {}
    for source_name, source_url in sources.items():
        pretrained_settings[model_name][source_name] = {
            "url": source_url,
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'input_space': 'RGB',
        }


timm_mobilenetv3_encoders = {
    'timm-mobilenetv3_large_075': {
        'encoder': MobileNetV3Encoder,
        'pretrained_settings': pretrained_settings['tf_mobilenetv3_large_075'],
        'params': {
            'model': 'tf_mobilenetv3_large_075',
            'width_mult': 0.75
        }
    },
    'timm-mobilenetv3_large_100': {
        'encoder': MobileNetV3Encoder,
        'pretrained_settings': pretrained_settings['tf_mobilenetv3_large_100'],
        'params': {
            'model': 'tf_mobilenetv3_large_100',
            'width_mult': 1.0
        }
    },
    'timm-mobilenetv3_large_minimal_100': {
        'encoder': MobileNetV3Encoder,
        'pretrained_settings': pretrained_settings['tf_mobilenetv3_large_minimal_100'],
        'params': {
            'model': 'tf_mobilenetv3_large_minimal_100',
            'width_mult': 1.0
        }
    },
    'timm-mobilenetv3_small_025': {
        'encoder': MobileNetV3Encoder,
        'pretrained_settings': pretrained_settings['tf_mobilenetv3_small_025'],
        'params': {
            'model': 'tf_mobilenetv3_small_025',
            'width_mult': 0.25,
            'pad_type': 'same',
            'arch_def': [
                # stage 0, 112x112 in
                ['ds_r1_k3_s2_e1_c16_se0.25_nre'],  # relu
                # stage 1, 56x56 in
                ['ir_r1_k3_s2_e4.5_c24_nre', 'ir_r1_k3_s1_e3.67_c24_nre'],  # relu
                # stage 2, 28x28 in
                ['ir_r1_k5_s2_e4_c40_se0.25', 'ir_r2_k5_s1_e6_c40_se0.25'],  # hard-swish
                # stage 3, 14x14 in
                ['ir_r2_k5_s1_e3_c48_se0.25'],  # hard-swish
                # stage 4, 14x14in
                ['ir_r3_k5_s2_e6_c96_se0.25'],  # hard-swish
                # stage 6, 7x7 in
                ['cn_r1_k1_s1_c576'],  # hard-swish
            ]
        }
    },
    'timm-mobilenetv3_small_075': {
        'encoder': MobileNetV3Encoder,
        'pretrained_settings': pretrained_settings['tf_mobilenetv3_small_075'],
        'params': {
            'model': 'tf_mobilenetv3_small_075',
            'width_mult': 0.75
        }
    },
    'timm-mobilenetv3_small_100': {
        'encoder': MobileNetV3Encoder,
        'pretrained_settings': pretrained_settings['tf_mobilenetv3_small_100'],
        'params': {
            'model': 'tf_mobilenetv3_small_100',
            'width_mult': 1.0
        }
    },
    'timm-mobilenetv3_small_minimal_100': {
        'encoder': MobileNetV3Encoder,
        'pretrained_settings': pretrained_settings['tf_mobilenetv3_small_minimal_100'],
        'params': {
            'model': 'tf_mobilenetv3_small_minimal_100',
            'width_mult': 1.0
        }
    },
    'timm-mobilenetv3_small_minimal_010': {
        'encoder': MobileNetV3Encoder,
        'pretrained_settings': pretrained_settings['tf_mobilenetv3_small_minimal_010'],
        'params': {
            'model': 'mobilenetv3_small_minimal_010',
            'width_mult': 0.10,
            'arch_def': [
                # stage 0, 112x112 in
                ['ds_r1_k3_s2_e1_c16'],
                # stage 1, 56x56 in
                ['ir_r1_k3_s2_e4.5_c24', 'ir_r1_k3_s1_e3.67_c24'],
                # stage 2, 28x28 in
                ['ir_r1_k3_s2_e4_c40', 'ir_r2_k3_s1_e6_c40'],
                # stage 3, 14x14 in
                ['ir_r2_k3_s1_e3_c48'],
                # stage 4, 14x14in
                ['ir_r3_k3_s2_e6_c96'],
                # stage 6, 7x7 in
                ['cn_r1_k1_s1_c576'],
            ]
        }
    },
    'timm-mobilenetv3_small_minimal_025': {
        'encoder': MobileNetV3Encoder,
        'pretrained_settings': pretrained_settings['tf_mobilenetv3_small_minimal_025'],
        'params': {
            'model': 'mobilenetv3_small_minimal_025',
            'width_mult': 0.25,
            'arch_def': [
                # stage 0, 112x112 in
                ['ds_r1_k3_s2_e1_c16'],
                # stage 1, 56x56 in
                ['ir_r1_k3_s2_e4.5_c24', 'ir_r1_k3_s1_e3.67_c24'],
                # stage 2, 28x28 in
                ['ir_r1_k3_s2_e4_c40', 'ir_r2_k3_s1_e6_c40'],
                # stage 3, 14x14 in
                ['ir_r2_k3_s1_e3_c48'],
                # stage 4, 14x14in
                ['ir_r3_k3_s2_e6_c96'],
                # stage 6, 7x7 in
                ['cn_r1_k1_s1_c576'],
            ]
        }
    },
    'timm-mobilenetv3_small_minimal_050': {
        'encoder': MobileNetV3Encoder,
        'pretrained_settings': pretrained_settings['tf_mobilenetv3_small_minimal_025'],
        'params': {
            'model': 'mobilenetv3_small_minimal_050',
            'width_mult': 0.5,
            'arch_def': [
                # stage 0, 112x112 in
                ['ds_r1_k3_s2_e1_c16'],
                # stage 1, 56x56 in
                ['ir_r1_k3_s2_e4.5_c24', 'ir_r1_k3_s1_e3.67_c24'],
                # stage 2, 28x28 in
                ['ir_r1_k3_s2_e4_c40', 'ir_r2_k3_s1_e6_c40'],
                # stage 3, 14x14 in
                ['ir_r2_k3_s1_e3_c48'],
                # stage 4, 14x14in
                ['ir_r3_k3_s2_e6_c96'],
                # stage 6, 7x7 in
                ['cn_r1_k1_s1_c576'],
            ]
        }
    },
    'timm-mobilenetv3_small_minimal_010': {
        'encoder': MobileNetV3Encoder,
        'pretrained_settings': pretrained_settings['tf_mobilenetv3_small_minimal_100'],
        'params': {
            'model': 'mobilenetv3_small_minimal_010',
            'width_mult': 0.1,
            'arch_def': [
                # stage 0, 112x112 in
                ['ds_r1_k3_s2_e1_c16'],
                # stage 1, 56x56 in
                ['ir_r1_k3_s2_e4.5_c24', 'ir_r1_k3_s1_e3.67_c24'],
                # stage 2, 28x28 in
                ['ir_r1_k3_s2_e4_c40', 'ir_r2_k3_s1_e6_c40'],
                # stage 3, 14x14 in
                ['ir_r2_k3_s1_e3_c48'],
                # stage 4, 14x14in
                ['ir_r3_k3_s2_e6_c96'],
                # stage 6, 7x7 in
                ['cn_r1_k1_s1_c576'],
            ]
        }
    },
}
