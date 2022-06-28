"""
BSD 3-Clause License

Copyright (c) Soumith Chintala 2016,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import torch
from torch import nn
from torch.nn import functional as F
from ..base import Conv2d, SeparableConv2d, ASPP, LRASPP

__all__ = ["DeepLabV3Decoder"]


class DeepLabV3Decoder(nn.Sequential):
    def __init__(self, in_channels, out_channels=256, atrous_rates=(12, 24, 36)):
        super().__init__(
            ASPP(in_channels, out_channels, atrous_rates),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.out_channels = out_channels

    def forward(self, *features):
        return super().forward(features[-1])


class DeepLabV3PlusDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        out_channels=256,
        atrous_rates=(12, 24, 36),
        output_stride=16,
        **kwargs
    ):
        super().__init__()
        if output_stride not in {8, 16}:
            raise ValueError(
                "Output stride should be 8 or 16, got {}.".format(output_stride))

        self.out_channels = out_channels
        self.output_stride = output_stride
        self.aspp = nn.Sequential(
            kwargs.pop('spp')(in_channels=encoder_channels[-1],
                              mid_channels=out_channels,
                              out_channels=out_channels,
                              atrous_rates=atrous_rates,
                              separable=kwargs.pop('separable', True)),
            SeparableConv2d(out_channels, out_channels,
                            kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        scale_factor = 2 if output_stride == 8 else 4
        self.up = nn.UpsamplingBilinear2d(scale_factor=scale_factor)

        highres_in_channels = encoder_channels[-4]
        self.highres_out_channels = 48   # proposed by authors of paper
        # highres_out_channels = out_channels
        self.block1 = nn.Sequential(
            nn.Conv2d(highres_in_channels, self.highres_out_channels,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(self.highres_out_channels),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            SeparableConv2d(
                self.highres_out_channels + out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, *features):
        result = {}
        if self.scse:
            features = self.scse(features)
        if self.fpn:
            features = self.fpn(features[-4:])
            result.update(features=features)
        aspp_features = self.aspp(features[-1])
        low_feat = features[-4]
        high_res_features = self.block1(low_feat)
        result.update(p2=high_res_features)
        aspp_features = self.up(aspp_features)
        if self.ffm:
            concat_features = self.ffm(aspp_features, high_res_features)
        else:
            concat_features = torch.cat([aspp_features, high_res_features], dim=1)

        fused_features = self.block2(concat_features)
        result.update(fused_features=fused_features)
        return result
