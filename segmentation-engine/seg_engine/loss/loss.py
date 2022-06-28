import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from torch.nn import functional as F
from torch.autograd import Variable
from .dice_loss import DC_and_Focal_loss
from .build import LOSS_REGISTRY
from ..config import configurable


@LOSS_REGISTRY.register()
class MattingLoss(nn.MSELoss):
    @configurable
    def __init__(self):
        super(MattingLoss, self).__init__()

    @classmethod
    def from_config(cls, cfg):
        return {}

    def forward(self, outs, targets, **kwargs):
        """
        Args:
            pred_fgr: Shape(B, 3, H, W)
            pred_pha: Shape(B, 1, H, W)
            true_fgr: Shape(B, 3, H, W)
            true_pha: Shape(B, 1, H, W)
        """
        pred_pha, pred_fgr = outs['alpha'], outs['foreground']
        true_pha, true_fgr = targets, kwargs['foreground']
        loss = dict()
        # Alpha losses
        loss['pha_l1'] = F.l1_loss(pred_pha, true_pha)
        loss['pha_laplacian'] = laplacian_loss(pred_pha, true_pha)
        loss['pha_coherence'] = F.mse_loss(pred_pha, true_pha) * 5
        # Foreground losses
        true_msk = true_pha.gt(0)
        pred_fgr = pred_fgr * true_msk
        true_fgr = true_fgr * true_msk
        loss['fgr_l1'] = F.l1_loss(pred_fgr, true_fgr)
        loss['fgr_coherence'] = F.mse_loss(pred_fgr, true_fgr) * 5
        return loss


def laplacian_loss(pred, true, max_levels=5):
    kernel = gauss_kernel(device=pred.device, dtype=pred.dtype)
    pred_pyramid = laplacian_pyramid(pred, kernel, max_levels)
    true_pyramid = laplacian_pyramid(true, kernel, max_levels)
    loss = 0
    for level in range(max_levels):
        loss += (2 ** level) * F.l1_loss(pred_pyramid[level], true_pyramid[level])
    return loss / max_levels


def laplacian_pyramid(img, kernel, max_levels):
    current = img
    pyramid = []
    for _ in range(max_levels):
        current = crop_to_even_size(current)
        down = downsample(current, kernel)
        up = upsample(down, kernel)
        diff = current - up
        pyramid.append(diff)
        current = down
    return pyramid


def gauss_kernel(device='cpu', dtype=torch.float32):
    kernel = torch.tensor([[1,  4,  6,  4, 1],
                           [4, 16, 24, 16, 4],
                           [6, 24, 36, 24, 6],
                           [4, 16, 24, 16, 4],
                           [1,  4,  6,  4, 1]], device=device, dtype=dtype)
    kernel /= 256
    kernel = kernel[None, None, :, :]
    return kernel


def gauss_convolution(img, kernel):
    B, C, H, W = img.shape
    img = img.reshape(B * C, 1, H, W)
    img = F.pad(img, (2, 2, 2, 2), mode='reflect')
    img = F.conv2d(img, kernel)
    img = img.reshape(B, C, H, W)
    return img


def downsample(img, kernel):
    img = gauss_convolution(img, kernel)
    img = img[:, :, ::2, ::2]
    return img


def upsample(img, kernel):
    B, C, H, W = img.shape
    out = torch.zeros((B, C, H * 2, W * 2), device=img.device, dtype=img.dtype)
    out[:, :, ::2, ::2] = img * 4
    out = gauss_convolution(out, kernel)
    return out


def crop_to_even_size(img):
    H, W = img.shape[2:]
    H = H - H % 2
    W = W - W % 2
    return img[:, :, :H, :W]


@ LOSS_REGISTRY.register()
class SelfRegulationLoss(nn.Module):
    @ configurable
    def __init__(self):
        super(SelfRegulationLoss, self).__init__()
        self.sr_logit_loss = nn.BCELoss()
        self.sr_feature_loss = nn.CrossEntropyLoss()
        self.loss = DC_and_Focal_loss()

    def forward(self, results, target):
        dice_focal_loss = self.loss(results['masks'], target)
        sr_logit_loss = self.sr_logit_loss(results['sr_logit_student'], results['sr_logit_teacher'].detach())
        sr_feature_loss = self.sr_feature_loss(results['masks'], results['sr_feature_teacher'].argmax(dim=1))
        return {
            'dice_focal_loss': dice_focal_loss,
            'sr_logit_loss': sr_logit_loss,
            'sr_feature_loss': sr_feature_loss,
        }

    @classmethod
    def from_config(cls, cfg=None):
        return {}
