import torch
import torch.nn as nn
from torch.nn import init
import math
from ._base import EncoderMixin
from ..base import Conv2d


class AddBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super(AddBottleneck, self).__init__()
        assert block_num > 1, print("block number should be larger than 1.")
        self.conv_list = nn.ModuleList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(out_planes//2, out_planes//2, kernel_size=3, stride=2,
                          padding=1, groups=out_planes//2, bias=False),
                nn.BatchNorm2d(out_planes//2),
            )
            self.skip = nn.Sequential(
                nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=2, padding=1, groups=in_planes, bias=False),
                nn.BatchNorm2d(in_planes),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_planes),
            )
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(Conv2d(in_planes, out_planes//2, kernel_size=1, bias=False,
                                             norm=nn.BatchNorm2d, activation=nn.ReLU(inplace=True)))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(Conv2d(out_planes//2, out_planes//2, stride=stride, bias=False,
                                             norm=nn.BatchNorm2d, activation=nn.ReLU(inplace=True)))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(Conv2d(out_planes//2, out_planes//4, stride=stride, bias=False,
                                             norm=nn.BatchNorm2d, activation=nn.ReLU(inplace=True)))
            elif idx < block_num - 1:
                self.conv_list.append(Conv2d(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx+1)), bias=False,
                                             norm=nn.BatchNorm2d, activation=nn.ReLU(inplace=True)))
            else:
                self.conv_list.append(Conv2d(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx)), bias=False,
                                             norm=nn.BatchNorm2d, activation=nn.ReLU(inplace=True)))

    def forward(self, x):
        out_list = []
        out = x

        for idx, conv in enumerate(self.conv_list):
            if idx == 0 and self.stride == 2:
                out = self.avd_layer(conv(out))
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            x = self.skip(x)

        return torch.cat(out_list, dim=1) + x


class CatBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super(CatBottleneck, self).__init__()
        assert block_num > 1, print("block number should be larger than 1.")
        self.conv_list = nn.ModuleList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(out_planes//2, out_planes//2, kernel_size=3, stride=2,
                          padding=1, groups=out_planes//2, bias=False),
                nn.BatchNorm2d(out_planes//2),
            )
            self.skip = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(Conv2d(in_planes, out_planes//2, kernel_size=1, bias=False,
                                             norm=nn.BatchNorm2d, activation=nn.ReLU(inplace=True)))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(Conv2d(out_planes//2, out_planes//2, stride=stride, bias=False,
                                             norm=nn.BatchNorm2d, activation=nn.ReLU(inplace=True)))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(Conv2d(out_planes//2, out_planes//4, stride=stride, bias=False,
                                             norm=nn.BatchNorm2d, activation=nn.ReLU(inplace=True)))
            elif idx < block_num - 1:
                self.conv_list.append(Conv2d(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx+1)), bias=False,
                                             norm=nn.BatchNorm2d, activation=nn.ReLU(inplace=True)))
            else:
                self.conv_list.append(Conv2d(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx)), bias=False,
                                             norm=nn.BatchNorm2d, activation=nn.ReLU(inplace=True)))

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)

        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)
        out = torch.cat(out_list, dim=1)
        return out


class STDCNet(nn.Module):
    """STDCNet

    Args:
        nn ([type]): [description]
    """

    def __init__(self, base, layers, block_num, type="cat", pretrain_model='', use_conv_last=False):
        super(STDCNet, self).__init__()
        if type == "cat":
            block = CatBottleneck
        elif type == "add":
            block = AddBottleneck
        self.use_conv_last = use_conv_last
        self.features = self._make_layers(base, layers, block_num, block)
        self.conv_last = Conv2d(base*16, max(1024, base*16), 1, 1, bias=False,
                                norm=nn.BatchNorm2d, activation=nn.ReLU(inplace=True))
        if isinstance(self, STDCNet1):
            self.mode = 'stdc1'
        elif isinstance(self, STDCNet2):
            self.mode = 'stdc2'
        else:
            raise NotImplementedError
        if pretrain_model:
            print('use pretrain model {}'.format(pretrain_model))
            self.init_weight(pretrain_model)
        else:
            self.init_params()

    def init_weight(self, pretrain_model):

        state_dict = torch.load(pretrain_model)["state_dict"]
        self_state_dict = self.state_dict()
        for k, v in state_dict.items():
            self_state_dict.update({k: v})
        self.load_state_dict(self_state_dict)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _make_layers(self, base, layers, block_num, block):
        features = []
        features += [Conv2d(3, base//2, 3, 2, bias=False, norm=nn.BatchNorm2d, activation=nn.ReLU(inplace=True))]
        features += [Conv2d(base//2, base, 3, 2, bias=False, norm=nn.BatchNorm2d, activation=nn.ReLU(inplace=True))]

        for i, layer in enumerate(layers):
            for j in range(layer):
                if i == 0 and j == 0:
                    features.append(block(base, base*4, block_num, 2))
                elif j == 0:
                    features.append(block(base*int(math.pow(2, i+1)), base*int(math.pow(2, i+2)), block_num, 2))
                else:
                    features.append(block(base*int(math.pow(2, i+2)), base*int(math.pow(2, i+2)), block_num, 1))

        return nn.Sequential(*features)

    def forward(self, x):
        out = self.features(x)
        return out


class STDCNet2(STDCNet):
    """STDC2Net

    Args:
        nn ([type]): [description]
    """

    def __init__(self, base=64, layers=[4, 5, 3], block_num=4, type="cat", pretrain_model='', use_conv_last=False):
        super(STDCNet2, self).__init__(base=base,
                                       layers=layers,
                                       block_num=block_num,
                                       type=type,
                                       pretrain_model=pretrain_model,
                                       use_conv_last=use_conv_last)


class STDCNet1(STDCNet):
    """STDC1Net

    Args:
        nn ([type]): [description]
    """

    def __init__(self, base=64, layers=[2, 2, 2], block_num=4, type="cat", pretrain_model='', use_conv_last=False):
        super(STDCNet1, self).__init__(base=base,
                                       layers=layers,
                                       block_num=block_num,
                                       type=type,
                                       pretrain_model=pretrain_model,
                                       use_conv_last=use_conv_last)


class STDCNetEncoder(nn.Module, EncoderMixin):
    def __init__(self, mode, depth=5):
        super(STDCNetEncoder, self).__init__()
        if mode == 'STDCNet1':
            self.backbone = STDCNet1()
        elif mode == 'STDCNet2':
            self.backbone = STDCNet2()
        else:
            raise NotImplementedError
        self._depth = depth
        self._out_channels = (3, 32, 64, 256, 512, 1024)
        self.mode = mode

    def get_stages(self):
        if self.mode == 'STDCNet1':
            return [
                nn.Identity(),
                self.backbone.features[:1],
                self.backbone.features[1:2],
                self.backbone.features[2:4],
                self.backbone.features[4:6],
                self.backbone.features[6:],
            ]
        elif self.mode == 'STDCNet2':
            return [
                nn.Identity(),
                self.backbone.features[:1],
                self.backbone.features[1:2],
                self.backbone.features[2:6],
                self.backbone.features[6:11],
                self.backbone.features[11:],
            ]

    def forward(self, x):
        stages = self.get_stages()
        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)
        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop('fc.weight')
        state_dict.pop('bn.weight')
        state_dict.pop('bn.bias')
        state_dict.pop('bn.running_mean')
        state_dict.pop('bn.running_var')
        state_dict.pop('bn.num_batches_tracked')
        state_dict.pop('linear.weight')
        state_dict_new = {}
        for k, v in state_dict.items():
            state_dict_new['backbone.'+k] = v
        super().load_state_dict(state_dict_new, **kwargs)


stdcnet_weights = {
    'STDCNet1':
    {
        'imagenet': '/data/zhangziwei/.cache/torch/hub/checkpoints/stdcnet1_cls.pth'
    },
    'STDCNet2':
    {
        'imagenet': '/data/zhangziwei/.cache/torch/hub/checkpoints/stdcnet2_cls.pth'
    }
}
pretrained_settings = {}
for model_name, sources in stdcnet_weights.items():
    pretrained_settings[model_name] = {}
    for source_name, source_url in sources.items():
        pretrained_settings[model_name][source_name] = {
            "url": source_url,
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'input_space': 'RGB',
        }


stdcnet_encoders = {
    'stdcnet1': {
        'encoder': STDCNetEncoder,
        'pretrained_settings': pretrained_settings['STDCNet1'],
        'params': {
            'mode': 'STDCNet1',
        }
    },
    'stdcnet2': {
        'encoder': STDCNetEncoder,
        'pretrained_settings': pretrained_settings['STDCNet2'],
        'params': {
            'mode': 'STDCNet2',
        }
    },
}
