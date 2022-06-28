import torch
from test_cfg import setup
import sys
sys.path.append('..')   # noqa: E403
from reg_engine import build_model


def rename_ckpt(name):
    ckpts = torch.load(name, map_location='cpu')['net']
    new_ckpts = {}
    for k, v in ckpts.items():
        if 's_net' in k:
            nk = k.replace('s_net.', '')
            print('#### {} ====> {}'.format(k, nk))
            new_ckpts[nk] = v
        else:
            new_ckpts[k] = v
    ckpts['net'] = new_ckpts
    torch.save(ckpts, name)
    print('##### new checkpoint saved')


def rename_all_ckpts():
    import os
    for root, dirs, files in os.walk('../log/HandGestureRecognition/timm-tf_mobilenetv3_small_minimal_100_BasicHead_small_sz128x128_kdl_0828-5fold-kd-mbl-t10'):
        for f in files:
            if f.endswith('pth'):
                print(f)
                rename_ckpt(os.path.join(root, f))


def compare_ckpts(nm1, nm2):
    ckpt1 = torch.load(nm1, map_location='cpu')['net']
    ckpt2 = torch.load(nm2, map_location='cpu')['net']
    for (k1, v1), (k2, v2) in zip(ckpt1.items(), ckpt2.items()):
        if k1 == k2:
            print(all(v1 == v2))


if __name__ == '__main__':
    rename_all_ckpts()
