import torch
from torch._C import device, dtype
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import SelectAdaptivePool2d, create_conv2d
from .build import HEAD_REGISTRY
from ..config import configurable
from .modules import SEModule, GeM


@ HEAD_REGISTRY.register()
class BasicHead(nn.Module):
    @ configurable
    def __init__(self, n_reg, in_plane, out_plane=None, mode='normal', dropout=0.0):
        super(BasicHead, self).__init__()
        if mode == 'normal':
            self.head = nn.Sequential(SelectAdaptivePool2d(pool_type='avg', flatten=True),
                                      nn.Dropout(p=dropout),
                                      nn.Linear(out_plane, n_reg))
        else:
            self.head = nn.Sequential(nn.Identity(),
                                      nn.Linear(in_plane, n_reg))
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.head(x)
        x = F.sigmoid(x)
        return {
            'ranking': x
        }

    @ classmethod
    def from_config(cls, cfg, C):
        return {
            'n_reg': cfg.DATASET.NUM_REG,
            'in_plane': C,
            'out_plane': C,
            'mode': cfg.MODEL.HEAD.MODE,
            'dropout': cfg.MODEL.HEAD.DROPOUT,
        }


@ HEAD_REGISTRY.register()
class ClsHead(nn.Module):
    @ configurable
    def __init__(self, n_reg, in_plane, out_plane=None, mode='normal', dropout=0.0):
        super(ClsHead, self).__init__()
        self.n_reg = n_reg
        self.head = nn.Sequential(SelectAdaptivePool2d(pool_type='avg', flatten=True),
                                  nn.Dropout(p=dropout),
                                  nn.Linear(out_plane, n_reg))
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.head(x)
        return {
            # expectation of the distribution
            'dist_label': x,
            'ranking': (x.argmax(dim=1, keepdim=True)/(self.n_reg-1)).clamp(0, 1),
        }

    @ classmethod
    def from_config(cls, cfg, C):
        return {
            'n_reg': cfg.DATASET.NUM_REG,
            'in_plane': C,
            'out_plane': C,
            'mode': cfg.MODEL.HEAD.MODE,
            'dropout': cfg.MODEL.HEAD.DROPOUT,
        }


@ HEAD_REGISTRY.register()
class DistributedlabelHead(nn.Module):
    @ configurable
    def __init__(self, n_reg, in_plane, out_plane=None, mode='normal', dropout=0.0):
        super(DistributedlabelHead, self).__init__()
        self.head = nn.Sequential(SelectAdaptivePool2d(pool_type='avg', flatten=True),
                                  nn.Dropout(p=dropout),
                                  nn.Linear(out_plane, 101))
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

    def forward(self, x):
        rank = torch.Tensor([i for i in range(101)]).to(x.device)
        x = self.head(x)
        x = F.softmax(x, dim=1)
        return {
            # distribuition
            'dist_label': x,
            # expectation of the distribution
            'ranking': ((rank * x).sum(dim=1, keepdim=True)/100).clamp(0, 1)
        }

    @ classmethod
    def from_config(cls, cfg, C):
        return {
            'n_reg': cfg.DATASET.NUM_REG,
            'in_plane': C,
            'out_plane': C,
            'mode': cfg.MODEL.HEAD.MODE,
            'dropout': cfg.MODEL.HEAD.DROPOUT,
        }
