from torch import nn
import torch
import torch.nn.functional as F


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class SimAM(nn.Module):
    '''
    SimAM: A Simple, Parameter-Free Attention Module for Convolutional Neural Networks
    http://proceedings.mlr.press/v139/yang21o/yang21o.pdf
    '''
    # X: input feature [N, C, H, W]
    # lambda: coefficient λ in Eqn (5)

    def __init__(self, lambda_=10e-4):
        super(SimAM, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        # spatial size
        n = x.shape[2] * x.shape[3] - 1
        # square of (t - u)
        d = (x - x.mean(dim=[2, 3])).pow(2)
        # d.sum() / n is channel variance
        v = d.sum(dim=[2, 3]) / n
        # E_inv groups all importance of x
        E_inv = d / (4 * (v + self.lambda_)) + 0.5
        # return attended features
        return x * F.sigmoid(E_inv)


class ChannelAttention(nn.Module):
    '''
    CBAM channel模块
    '''

    def __init__(self, in_planes, rotio=16):
        super(ChannelAttention, self).__init__()
        # 全局池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 全局池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // rotio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttention(nn.Module):
    '''
    CBAM Spatial模块
    '''

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return se


class SEModule(nn.Module):
    '''
    SE Module
    '''

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = x.mean((2, 3), keepdim=True)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x
