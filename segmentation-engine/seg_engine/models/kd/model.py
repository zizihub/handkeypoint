from seg_engine.optim.build import build_optimizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from .sagan import Discriminator
from ...loss import CWD


class KDModel:
    def __init__(self, student_net, teacher_net, criterion, mode):
        self.student_net = student_net
        self.teacher_net = teacher_net
        self.criterion = criterion
        self.mode = mode
        if mode == 'CWD':
            self.cwd = CWD((256,), (256,))

        # * forward pipeline map
        self.loss_forward = {
            'DML': self.forward_dml,
            'soft target': self.forward_st,
            'DistanceLoss': self.forward_st,
            'CWD': self.forward_st,
        }

    def forward(self, inputs, targets):
        loss_dic = {}
        preds_T = self.teacher_net(inputs)
        preds_S = self.student_net(inputs)
        loss_dic = self.loss_forward[self.mode](preds_T, preds_S, targets)
        return loss_dic

    def forward_st(self, mask_T, mask_S, targets):
        kd, ce = self.criterion(mask_S, mask_T, targets)
        loss = kd + ce
        loss.backward()
        return {
            "kd_loss": kd.item(),
            "ce_loss": ce.item()
        }

    def forward_dml(self, mask_T, mask_S, targets):
        kd_t, ce_t = self.criterion(mask_T, mask_S, targets)
        kd_s, ce_s = self.criterion(mask_S, mask_T, targets)
        loss_T = kd_t + ce_t
        loss_S = kd_s + ce_s
        loss_T.backward(retain_graph=True)
        loss_S.backward()
        return {
            "kd_teacher": kd_t.item(),
            "ce_teacher": ce_t.item(),
            "kd_student": kd_s.item(),
            "ce_student": ce_s.item(),
        }

    # def forward_cwd(self, preds_T, preds_S, targets):
    #     feature_T = (preds_T['fused_features'],)
    #     feature_S = (preds_S['fused_features'],)
    #     cwd_loss = self.cwd(feature_S, feature_T)
    #     # kd_loss, ce_loss = self.criterion(preds_S, preds_T, targets)
    #     loss = cwd_loss
    #     loss.backward()
    #     return {
    #         # "kd_loss": kd_loss.item(),
    #         # "ce_loss": ce_loss.item(),
    #         "cwd_loss": cwd_loss.item(),
    #     }

    def predict(self, inputs):
        with torch.no_grad():
            pred = self.student_net(inputs)
        return pred

    def train(self):
        if self.mode == 'DML':
            self.student_net.train()
            self.teacher_net.train()
        else:
            self.student_net.train()
            self.teacher_net.eval()

    def eval(self):
        self.student_net.eval()
        self.teacher_net.eval()
