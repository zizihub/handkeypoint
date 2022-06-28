import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import Conv2d
from .build import POSTPROCESS_REGISTRY
from ...config import configurable


class FlowField(nn.Module):
    def __init__(self, inplane, norm_layer):
        """
        implementation of body generation part
        :param inplane:
        :param norm_layer:
        """
        super(FlowField, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(inplane, inplane, kernel_size=3, groups=inplane, stride=2),
            norm_layer(inplane),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplane, inplane, kernel_size=3, groups=inplane, stride=2),
            norm_layer(inplane),
            nn.ReLU(inplace=True)
        )

        self.flow_make = nn.Conv2d(inplane * 2, 2, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        size = x.size()[2:]
        seg_down = self.down(x)
        seg_down = F.upsample(seg_down, size=size, mode="bilinear", align_corners=True)
        flow = self.flow_make(torch.cat([x, seg_down], dim=1))
        seg_flow_warp = self.flow_warp(x, flow, size)
        seg_edge = x - seg_flow_warp
        return seg_flow_warp, seg_edge

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        # new
        h_grid = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w_gird = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w_gird.unsqueeze(2), h_grid.unsqueeze(2)), 2)

        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid)
        return output


@POSTPROCESS_REGISTRY.register()
def Decoupled(cfg, encoder_channels):
    inplane = cfg.MODEL.DECODER.CHANNELS
    if 'FPN' in cfg.MODEL.FPN.NAME:
        highres_in_channels = cfg.MODEL.FPN.OUT_CHANNELS
    else:
        highres_in_channels = encoder_channels[-4]
    highres_out_channels = 48
    decoupled = {}
    decoupled.update(flow_field=FlowField(inplane, nn.BatchNorm2d))
    decoupled.update(edge_out=nn.Sequential(
        Conv2d(inplane, highres_out_channels, padding=1, bias=False,
               norm=nn.BatchNorm2d, activation=nn.ReLU(inplace=True)),
        Conv2d(highres_out_channels, 1, kernel_size=1, bias=False)
    ))
    decoupled.update(bot_refine=Conv2d(highres_in_channels, highres_out_channels, kernel_size=1, bias=False))
    decoupled.update(edge_fusion=Conv2d(inplane+highres_out_channels, inplane, kernel_size=1, bias=False))
    decoupled.update(final_seg=nn.Sequential(
        Conv2d(inplane+inplane, inplane, bias=False, norm=nn.BatchNorm2d, activation=nn.ReLU(inplace=True)),
        Conv2d(inplane, inplane, bias=False, norm=nn.BatchNorm2d, activation=nn.ReLU(inplace=True))
    ))
    # decoupled.update(dsn_seg_body=nn.Sequential(
    #     Conv2d(inplane, inplane, padding=1, bias=False, norm=nn.BatchNorm2d, activation=nn.ReLU(inplace=True)),
    #     Conv2d(inplane, cfg.DATASET.NUM_CLASSES+1, bias=False)
    # ))
    return decoupled
