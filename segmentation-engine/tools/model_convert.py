import sys
import os
import glob
sys.path.append('..')  # noqa: E402
from seg_engine.config import get_cfg, get_outname
from seg_engine import build_model
from PIL import Image
from torchvision import transforms
import torch


def setup(args):
    '''
    Create configs and perform basic setups.
    '''
    config_path = glob.glob(os.path.join(
        os.path.dirname(args.checkpoint), '*.yaml'))
    my_cfg = config_path[0]
    cfg = get_cfg()
    cfg.merge_from_file(my_cfg)
    output_name = get_outname(cfg)
    cfg.merge_from_list(['OUTPUT_NAME', output_name])
    cfg.freeze()
    return cfg


def torch_save_pt(args):
    cfg = setup(args)
    model = build_model(cfg)
    model.load_state_dict(torch.load(
        args.checkpoint, map_location='cpu')['net'])
    model.eval()
    test_tensor = torch.randn(
        4, 4 if cfg.DATASET.POSTPROCESS.four_channel else 3, cfg.INPUT.SIZE[0], cfg.INPUT.SIZE[1])
    out = model.forward(test_tensor)
    for k, v in out.items():
        print('{} shape: {}'.format(
            k, v.shape if isinstance(v, torch.Tensor) else len(v)))
        print('{} shape: {}'.format(
            k, v[0, 0, :4, :4] if isinstance(v, torch.Tensor) else len(v)))
    torch.save(model, args.save_name)
    print('>>>>>> model save', args.save_name)
    model_pt = torch.load(args.save_name, map_location='cpu')
    out_pt = model_pt.forward(test_tensor)
    for k, v in out_pt.items():
        print('{} shape: {}'.format(
            k, v.shape if isinstance(v, torch.Tensor) else len(v)))
        print('{} shape: {}'.format(
            k, v[0, 0, :4, :4] if isinstance(v, torch.Tensor) else len(v)))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="model convert")
    parser.add_argument(
        "--checkpoint",
        default="../log/HandGestureRecognition/alexnet_rw_BasicHead_large_sz128x128_kdl_0729-r50/alexnet_rw_BasicHead_large_sz128x128_kdl_0729-r50_best.pth")
    parser.add_argument(
        "--save_name",
        default="../checkpoint/alex-hgr.pt")
    args = parser.parse_args()
    torch_save_pt(args)
