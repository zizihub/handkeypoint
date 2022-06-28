import torch.nn.functional as f
import torch.nn as nn
import torch


class Model(nn.Module):
    def __init__(self,
                 backbone,
                 heads,
                 cfg=None):
        super(Model, self).__init__()
        self.backbone = backbone
        self.heads = nn.ModuleDict()
        self.embedding = False
        self.deploy_flag = cfg.DEPLOY
        for k, v in heads.items():
            self.heads[k] = v

    def forward(self, x):
        features = self.backbone(x)
        if isinstance(features, list):
            features = features[-1]
        if self.heads:
            for head in self.heads.values():
                x = head(features)
        if self.embedding:
            if not self.deploy_flag:
                features = f.adaptive_avg_pool2d(features, (1, 1))
                features = features.flatten(1, -1)
                return x, features
            else:
                return x
        else:
            if not self.deploy_flag:
                return x, features
            else:
                return x


class Distiller(nn.Module):
    """Distiller for knowledge distillation process"""

    def __init__(self, teacher_nets, student_net, kd_loss, config):
        """init

        Args:
            teacher_nets (nn.Module): teacher networks
            student_net (nn.Module): student network
            kd_loss (nn.Module): knowledge distillation loss
        """
        super(Distiller, self).__init__()
        self.t_nets = teacher_nets
        self.s_net = student_net
        self.kd_loss = kd_loss
        self.mode = config.SOLVER.LOSS.NAME[0]
        self.kd = config.SOLVER.LOSS.MODE
        self.device = config.DEVICE

    def forward(self, x, **kwargs):
        s_output, s_features = self.s_net(x)
        if self.kd == 'DML':
            t_output, t_feature = self.t_nets(x)
        if not self.training:
            return s_output, s_features
        if self.mode == 'KnowledgeDistillationLoss':
            if self.kd == 'DML':
                loss_dic = self.forward_dml(t_output, s_output, **kwargs)
            elif self.kd == 'soft target':
                loss_dic = self.forward_soft_target(x, s_output, **kwargs)
        elif self.mode == 'ConstractiveRepresentationLoss':
            loss_dic = self.forward_crd(x, s_output, s_features, **kwargs)
        else:
            raise NotImplementedError
        return s_output, s_features, loss_dic

    def forward_dml(self, t_output, s_output, **kwargs):
        targets = kwargs.pop('targets', None)
        targets = targets.to(self.device) if targets != None else targets
        loss_dic = {}

        for k, loss_ in self.kd_loss.items():
            loss1 = loss_(s_output, t_output, targets)
            loss2 = loss_(t_output, s_output, targets)
            loss_dic.update(loss1=loss1, loss2=loss2)
        return loss_dic

    def forward_soft_target(self, x, s_output, **kwargs):
        targets = kwargs.pop('targets', None)
        targets = targets.to(self.device) if targets != None else targets

        loss_dic = {}
        for k, loss_ in self.kd_loss.items():
            loss = 0.0
            for t_net in self.t_nets:
                with torch.no_grad():
                    out, _ = t_net.eval()(x)
                loss += loss_(s_output, out, targets)
            loss /= len(self.t_nets)
            loss_dic.update(**{k: loss})
        return loss_dic

    def forward_crd(self, x, s_output, s_features, **kwargs):
        targets = kwargs.pop('targets', None)
        pos_index = kwargs.pop('pos_index', None)
        sample_index = kwargs.pop('sample_index', None)
        targets = targets.to(self.device) if targets != None else targets
        pos_index = pos_index.to(self.device) if pos_index != None else pos_index
        sample_index = sample_index.to(self.device) if sample_index != None else sample_index

        loss_dic = {}
        for k, loss_ in self.kd_loss.items():
            loss = 0.0
            for t_net in self.t_nets:
                with torch.no_grad():
                    output, feature = t_net.eval()(x)
                loss += loss_(s_output, s_features, output, feature, targets, pos_index, sample_index)
            loss /= len(self.t_nets)
            loss_dic.update(**{k: loss})
            return loss_dic
