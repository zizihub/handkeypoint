import torch
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F
import torch.nn as nn
from .dice_loss import DC_and_Focal_loss
from .build import LOSS_REGISTRY
from ..config import configurable


class ImageGradientLoss(_WeightedLoss):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(ImageGradientLoss, self).__init__(size_average, reduce, reduction)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, pred, gray_image):
        size = pred.size()
        pred = pred.argmax(1).view(size[0], 1, size[2], size[3]).float()
        gradient_tensor_x = torch.Tensor([[1.0, 0.0, -1.0],
                                          [2.0, 0.0, -2.0],
                                          [1.0, 0.0, -1.0]]).to(self.device).view((1, 1, 3, 3))

        gradient_tensor_y = torch.Tensor([[1.0, 2.0, 1.0],
                                          [0.0, 0.0, 0.0],
                                          [-1.0, -2.0, -1.0]]).to(self.device).view((1, 1, 3, 3))

        I_x = F.conv2d(gray_image, gradient_tensor_x)
        G_x = F.conv2d(pred, gradient_tensor_x)

        I_y = F.conv2d(gray_image, gradient_tensor_y)
        G_y = F.conv2d(pred, gradient_tensor_y)

        G = torch.sqrt(torch.pow(G_x, 2) + torch.pow(G_y, 2))

        gradient = 1 - torch.pow(I_x * G_x + I_y * G_y, 2)

        image_gradient_loss = torch.sum(torch.mul(G, gradient)) / torch.sum(G)

        image_gradient_loss = image_gradient_loss if image_gradient_loss > 0 else 0

        return image_gradient_loss


@LOSS_REGISTRY.register()
class DiceFocalandImageGradientLoss(nn.Module):

    @configurable
    def __init__(self, gradient_loss_weight):
        super(DiceFocalandImageGradientLoss, self).__init__()
        self.image_gradient_loss = ImageGradientLoss()
        self.dice_focal_loss = DC_and_Focal_loss(soft_dice_kwargs={'batch_dice': False, 'do_bg': True, 'smooth': 1., 'square': False},
                                                 focal_kwargs={'alpha': 0.25, 'gamma': 2, 'balance_index': 0, 'smooth': 1e-5})
        self.gradient_loss_weight = gradient_loss_weight

    @classmethod
    def from_config(cls, cfg):
        return {
            'gradient_loss_weight': cfg.SOLVER.GRADIENT_LOSS_WEIGHT
        }

    def forward(self, out, target, **kwargs):
        net_output = out['masks']
        grayscale = kwargs['grayscale']
        ig_loss = self.image_gradient_loss(net_output, grayscale)
        df_loss = self.dice_focal_loss(net_output, target)
        return self.gradient_loss_weight * ig_loss + df_loss
