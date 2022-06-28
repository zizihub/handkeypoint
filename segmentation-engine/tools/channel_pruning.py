import torch
import os


class ChannelPruning:
    def __init__(self, model_path):
        self.model_path = model_path

    def channel_pruning(self):
        ckpts = torch.load(self.model_path, map_location='cpu')['net']
        for k, v in ckpts.items():
            if not v.dim() == 4:
                continue
            print(k, torch.abs(v[:, :]).shape)


if __name__ == '__main__':
    model_path = '../log/Face/2021-09-16/timm-mobilenetv3_small_minimal_025_BIFPNx3+oc88_DeepLabV3Plusx16+oc256_SCSE_sz256_dcfl_occluded/timm-mobilenetv3_small_minimal_025_BIFPNx3+oc88_DeepLabV3Plusx16+oc256_SCSE_sz256_dcfl_occluded_best.pth'
    cp = ChannelPruning(model_path)
    cp.channel_pruning()
