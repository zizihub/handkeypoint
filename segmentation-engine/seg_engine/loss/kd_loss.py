import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from torch.nn import functional as F
from torch.autograd import Variable
from .build import LOSS_REGISTRY
from ..config import configurable
import scipy.ndimage as nd


class SoftTarget(nn.Module):
    '''
    Distilling the Knowledge in a Neural Network
    https://arxiv.org/pdf/1503.02531.pdf
    '''

    def __init__(self, T):
        super(SoftTarget, self).__init__()
        self.T = T

    def forward(self, out_s, out_t):
        loss = F.kl_div(F.log_softmax(out_s/self.T, dim=1),
                        F.softmax(out_t/self.T, dim=1),
                        reduction='mean') * self.T * self.T

        return loss


class DistanceLoss(nn.Module):
    '''
    DistanceLoss
    '''

    def __init__(self, mode="l2"):
        super().__init__()
        assert mode in ["l1", "l2", "smooth_l1"]
        if mode == "l1":
            self.loss_func = nn.L1Loss(reduction='mean')
        elif mode == "l2":
            self.loss_func = nn.MSELoss(reduction='mean')
        elif mode == "smooth_l1":
            self.loss_func = nn.SmoothL1Loss(reduction='mean')

    def forward(self, out_s, out_t):
        return self.loss_func(out_s, out_t)


class DML(nn.Module):
    '''
    Deep Mutual Learning
    https://zpascal.net/cvpr2018/Zhang_Deep_Mutual_Learning_CVPR_2018_paper.pdf
    '''

    def __init__(self):
        super(DML, self).__init__()

    def forward(self, out1, out2):
        loss = F.kl_div(F.log_softmax(out1, dim=1),
                        F.softmax(out2, dim=1),
                        reduction='mean')

        return loss


class CWD(nn.Module):
    '''
    Channel-wise Distillation for Semantic Segmentation
    https://arxiv.org/abs/2011.13256
    '''

    def __init__(self,
                 #  student_channels,
                 #  teacher_channels,
                 tau=4.0,
                 weight=3.0,
                 ):
        super(CWD, self).__init__()
        self.tau = tau
        self.loss_weight = weight
        # self.align = nn.ModuleList()
        # for student_channel, teacher_channel in zip(student_channels, teacher_channels):
        #     if student_channel != teacher_channel:
        #         self.align.append(nn.Conv2d(student_channel, teacher_channel, kernel_size=1, stride=1, padding=0))
        #     else:
        #         self.align.append(nn.Identity())
        # self.align = self.align.cuda()

    def forward_single_layer(self, preds_S, preds_T):
        """Forward function."""
        assert preds_S.shape[-2:] == preds_T.shape[-2:], 'the output dim of teacher and student differ'
        N, C, W, H = preds_S.shape
        # preds_S = self.align[idx](preds_S)
        loss = torch.mean(
            F.softmax(preds_T.view(-1, W*H)/self.tau, dim=1) *
            (F.log_softmax(preds_T.view(-1, W*H)/self.tau, dim=1) - F.log_softmax(preds_S.view(-1, W*H)/self.tau, dim=1))
        ) * (self.tau ** 2)
        return self.loss_weight * loss

    def forward(self, preds_S, preds_T):
        # cwd_loss = 0.0
        # compute last 4 feature cwd
        # for i in range(-1, -2, -1):
        cwd_loss = self.forward_single_layer(preds_S, preds_T)
        return cwd_loss


@LOSS_REGISTRY.register()
class KnowledgeDistillationLoss(nn.Module):
    @configurable
    def __init__(self, alpha, beta, temperature, mode):
        super(KnowledgeDistillationLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.T = temperature
        self.ce_loss = LOSS_REGISTRY.get('DC_and_Focal_loss')()
        kd_loss_map = {
            'soft target': SoftTarget(T=temperature),
            'DML': DML(),
            'DistanceLoss': DistanceLoss(),
            'CWD': CWD(),
        }
        self.kd_loss = kd_loss_map[mode]

    def forward(self, outputs, teacher_outputs, labels):
        kd = self.kd_loss(outputs['masks'], teacher_outputs['masks']) * self.alpha
        ce = self.ce_loss(outputs, labels) * self.beta
        return (kd, ce)

    @ classmethod
    def from_config(cls, cfg):
        return {
            "alpha": cfg.SOLVER.KD.ALPHA,
            "beta": cfg.SOLVER.KD.BETA,
            "temperature": cfg.SOLVER.KD.TEMPERATURE,
            "mode": cfg.SOLVER.KD.MODE,
        }


def L2(f_):
    return (((f_**2).sum(dim=1))**0.5).reshape(f_.shape[0], 1, f_.shape[2], f_.shape[3]) + 1e-8


def similarity(feat):
    feat = feat.float()
    tmp = L2(feat).detach()
    feat = feat/tmp
    feat = feat.reshape(feat.shape[0], feat.shape[1], -1)
    return torch.einsum('icm,icn->imn', [feat, feat])


def sim_dis_compute(f_S, f_T):
    sim_err = ((similarity(f_T) - similarity(f_S))**2)/((f_T.shape[-1]*f_T.shape[-2])**2)/f_T.shape[0]
    sim_dis = sim_err.sum()
    return sim_dis


'''
KD loss
'''


class OhemCrossEntropy2d(nn.Module):
    def __init__(self, ignore_label=255, thresh=0.7, min_kept=100000, factor=8):
        super(OhemCrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.factor = factor
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)

    def find_threshold(self, np_predict, np_target):
        # downsample 1/8
        factor = self.factor
        predict = nd.zoom(np_predict, (1.0, 1.0, 1.0/factor, 1.0/factor), order=1)
        target = nd.zoom(np_target, (1.0, 1.0/factor, 1.0/factor), order=0)

        n, c, h, w = predict.shape
        min_kept = self.min_kept // (factor*factor)  # int(self.min_kept_ratio * n * h * w)

        input_label = target.ravel().astype(np.int32)
        input_prob = np.rollaxis(predict, 1).reshape((c, -1))

        valid_flag = input_label != self.ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()
        if min_kept >= num_valid:
            threshold = 1.0
        elif num_valid > 0:
            prob = input_prob[:, valid_flag]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            threshold = self.thresh
            if min_kept > 0:
                k_th = min(len(pred), min_kept)-1
                new_array = np.partition(pred, k_th)
                new_threshold = new_array[k_th]
                if new_threshold > self.thresh:
                    threshold = new_threshold
        return threshold

    def generate_new_target(self, predict, target):
        np_predict = predict.data.cpu().numpy()
        np_target = target.data.cpu().numpy()
        n, c, h, w = np_predict.shape

        threshold = self.find_threshold(np_predict, np_target)

        input_label = np_target.ravel().astype(np.int32)
        input_prob = np.rollaxis(np_predict, 1).reshape((c, -1))

        valid_flag = input_label != self.ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()

        if num_valid > 0:
            prob = input_prob[:, valid_flag]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            kept_flag = pred <= threshold
            valid_inds = valid_inds[kept_flag]
            print('Labels: {} {}'.format(len(valid_inds), threshold))

        label = input_label[valid_inds].copy()
        input_label.fill(self.ignore_label)
        input_label[valid_inds] = label
        new_target = torch.from_numpy(input_label.reshape(target.size())).long().cuda(target.get_device())
        return new_target

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad

        input_prob = F.softmax(predict, 1)
        target = self.generate_new_target(input_prob, target)
        return self.criterion(predict, target)


class CriterionAdditionalGP(nn.Module):
    def __init__(self, D_net, lambda_gp):
        super(CriterionAdditionalGP, self).__init__()
        self.D = D_net
        self.lambda_gp = lambda_gp

    def forward(self, d_in_S, d_in_T):
        assert d_in_S.shape == d_in_T.shape, 'the output dim of D with teacher and student as input differ'

        real_images = d_in_T
        fake_images = d_in_S
        # Compute gradient penalty
        alpha = torch.rand(real_images.size(0), 1, 1, 1).cuda().expand_as(real_images)
        interpolated = Variable(alpha * real_images.data + (1 - alpha) * fake_images.data, requires_grad=True)
        out = self.D(interpolated)
        grad = torch.autograd.grad(outputs=out[0],
                                   inputs=interpolated,
                                   grad_outputs=torch.ones(out[0].size()).cuda(),
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        grad = grad.view(grad.size(0), -1)
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

        # Backward + Optimize
        d_loss = self.lambda_gp * d_loss_gp
        return d_loss


class CriterionAdvForG(nn.Module):
    def __init__(self, adv_type):
        super(CriterionAdvForG, self).__init__()
        if (adv_type != 'wgan-gp') and (adv_type != 'hinge'):
            raise ValueError('adv_type should be wgan-gp or hinge')
        self.adv_loss = adv_type

    def forward(self, d_out_S, d_out_S_no_use):
        g_out_fake = d_out_S
        if self.adv_loss == 'wgan-gp':
            g_loss_fake = - g_out_fake.mean()
        elif self.adv_loss == 'hinge':
            g_loss_fake = - g_out_fake.mean()
        else:
            raise ValueError('args.adv_loss should be wgan-gp or hinge')
        return g_loss_fake


class CriterionAdv(nn.Module):
    def __init__(self, adv_type):
        super(CriterionAdv, self).__init__()
        if (adv_type != 'wgan-gp') and (adv_type != 'hinge'):
            raise ValueError('adv_type should be wgan-gp or hinge')
        self.adv_loss = adv_type

    def forward(self, d_out_S, d_out_T):
        assert d_out_S.shape == d_out_T.shape, 'the output dim of D with teacher and student as input differ'
        '''teacher output'''
        d_out_real = d_out_T
        if self.adv_loss == 'wgan-gp':
            d_loss_real = - torch.mean(d_out_real)
        elif self.adv_loss == 'hinge':
            d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
        else:
            raise ValueError('args.adv_loss should be wgan-gp or hinge')

        # apply Gumbel Softmax
        '''student output'''
        d_out_fake = d_out_S
        if self.adv_loss == 'wgan-gp':
            d_loss_fake = d_out_fake.mean()
        elif self.adv_loss == 'hinge':
            d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()
        else:
            raise ValueError('args.adv_loss should be wgan-gp or hinge')
        return d_loss_real + d_loss_fake


class CriterionDSN(nn.Module):
    '''
    DSN : We need to consider two supervision for the model.
    '''

    def __init__(self, ignore_index=255, use_weight=True, reduce=True):
        super(CriterionDSN, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduce=reduce)
        if not reduce:
            print("disabled the reduce.")

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        loss1 = self.criterion(scale_pred, target)

        scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        loss2 = self.criterion(scale_pred, target)

        return loss1 + loss2*0.4


class CriterionOhemDSN(nn.Module):
    '''
    DSN : We need to consider two supervision for the model.
    '''

    def __init__(self, ignore_index=255, thresh=0.7, min_kept=100000, use_weight=True, reduce=True):
        super(CriterionOhemDSN, self).__init__()
        self.ignore_index = ignore_index
        self.criterion1 = OhemCrossEntropy2d(ignore_index, thresh, min_kept)
        self.criterion2 = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduce=reduce)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        loss1 = self.criterion1(scale_pred, target)

        scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        loss2 = self.criterion2(scale_pred, target)

        return loss1 + loss2*0.4


class CriterionPixelWise(nn.Module):
    '''
    arxiv:
    paper: structued knowledge distillatio for dense prediction
    '''

    def __init__(self, ignore_index=255, use_weight=True):
        super(CriterionPixelWise, self).__init__()
        self.ignore_index = ignore_index
        self.kldloss = nn.KLDivLoss(reduction='mean')

    def forward(self, preds_S, preds_T):
        preds_T.detach()
        assert preds_S.shape == preds_T.shape, 'the output dim of teacher and student differ'
        N, C, W, H = preds_S.shape
        softmax_pred_T = F.softmax(preds_T.permute(0, 2, 3, 1).contiguous().view(-1, C), dim=1)
        softmax_pred_S = F.log_softmax(preds_S.permute(0, 2, 3, 1).contiguous().view(-1, C), dim=1)
        loss = torch.sum(self.kldloss(softmax_pred_S, softmax_pred_T)) / W / H
        return loss


class CriterionPairWiseforWholeFeatAfterPool(nn.Module):
    '''inter pair-wise loss from inter feature maps
    scale: parameter to resize the feature map by pooling
    feat_ind: which layer to apply the pair-wise loss
    '''

    def __init__(self, scale, feat_ind):
        super(CriterionPairWiseforWholeFeatAfterPool, self).__init__()
        self.criterion = sim_dis_compute
        self.feat_ind = feat_ind
        self.scale = scale

    def forward(self, preds_S, preds_T):
        feat_S = preds_S[self.feat_ind]
        feat_T = preds_T[self.feat_ind]
        feat_T.detach()

        total_w, total_h = feat_T.shape[2], feat_T.shape[3]
        patch_w, patch_h = int(total_w*self.scale), int(total_h*self.scale)
        maxpool = nn.MaxPool2d(kernel_size=(patch_w, patch_h), stride=(
            patch_w, patch_h), padding=0, ceil_mode=True)  # change
        loss = self.criterion(maxpool(feat_S), maxpool(feat_T))
        return loss
