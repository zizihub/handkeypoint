from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from .build import LOSS_REGISTRY
from ..config import configurable
from .knowledge_distillation import *


@LOSS_REGISTRY.register()
class L1Loss(nn.Module):
    @configurable
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, outputs, true, **kwargs):
        if isinstance(outputs, dict):
            pred = outputs['ranking']
        else:
            pred = outputs
        return F.l1_loss(pred, true)

    @classmethod
    def from_config(cls, cfg):
        return {}


@LOSS_REGISTRY.register()
class SmoothL1Loss(nn.Module):
    @configurable
    def __init__(self):
        super(SmoothL1Loss, self).__init__()

    def forward(self, outputs, true, **kwargs):
        if isinstance(outputs, dict):
            pred = outputs['ranking']
        else:
            pred = outputs
        return F.smooth_l1_loss(pred, true)

    @classmethod
    def from_config(cls, cfg):
        return {}


@LOSS_REGISTRY.register()
class MSELoss(nn.Module):
    @configurable
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, outputs, true, **kwargs):
        if isinstance(outputs, dict):
            pred = outputs['ranking']
        else:
            pred = outputs
        return F.mse_loss(pred, true)

    @classmethod
    def from_config(cls, cfg):
        return {}


@LOSS_REGISTRY.register()
class CrossEntropyLoss(nn.Module):
    @configurable
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, outputs, true, **kwargs):
        true = kwargs['dist_label'].argmax(dim=1)
        pred = outputs['dist_label']
        return F.cross_entropy(pred, true)

    @classmethod
    def from_config(cls, cfg):
        return {}


@LOSS_REGISTRY.register()
class KLDivLoss(nn.Module):
    @configurable
    def __init__(self):
        super(KLDivLoss, self).__init__()

    def forward(self, outputs, true, **kwargs):
        true = kwargs['dist_label']
        pred = outputs['dist_label']
        return F.kl_div(pred.log(), true)

    @classmethod
    def from_config(cls, cfg):
        return {}
