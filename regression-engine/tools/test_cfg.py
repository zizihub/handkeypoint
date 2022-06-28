import sys
sys.path.append('..')   # noqa: E403
from reg_engine import build_model, build_loss, build_optimizer, build_scheduler
import torch
from ptflops.flops_counter import get_model_complexity_info, flops_to_string, params_to_string
from reg_engine.config import get_cfg, get_outname
import numpy as np
from torch.utils.data import DataLoader


def setup(my_cfg='test.yaml'):
    '''
    Create configs and perform basic setups.
    '''
    cfg = get_cfg()
    cfg.merge_from_file(my_cfg)
    output_name = get_outname(cfg)
    cfg.merge_from_list(['OUTPUT_NAME', output_name])
    cfg.merge_from_list(['DEPLOY', True])
    cfg.freeze()
    return cfg


# --------------------------------------------
#              test model
# --------------------------------------------

def test_model(args):
    cfg = setup(args.cfg)
    print(cfg)
    model = build_model(cfg)
    model.eval()
    if 0:
        ckpt = torch.load(cfg.MODEL.WEIGHTS, map_location='cpu')
        model.load_state_dict(ckpt['net'])
    x = (3, cfg.INPUT.SIZE[0], cfg.INPUT.SIZE[1])
    print('='*40)
    print('input shape: {}'.format(x))
    macs, params = get_model_complexity_info(model,
                                             x,
                                             as_strings=False,
                                             print_per_layer_stat=True,
                                             verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', flops_to_string(macs)))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params_to_string(params)))
    print('{:<30}  {:<8}'.format('Storage Size: ', params_to_string(params*4)))
    print('='*40)
    x = torch.randn(12, *x)
    with torch.no_grad():
        out = model(x)
        if isinstance(out, dict):
            for k, v in out.items():
                if isinstance(v, torch.Tensor):
                    print('{} shape: {}'.format(k, v.shape))
                else:
                    for i, out in enumerate(v):
                        print('feature {} shape: {}'.format(i, out.shape))
        else:
            print('shape: {}'.format(out.shape))


# --------------------------------------------
#              test loss
# --------------------------------------------

def test_loss(args):
    cfg = setup(args.cfg)
    for k, loss in build_loss(cfg).items():
        print('>>>> testing {}'.format(k))
        x = torch.rand([64, 1], dtype=torch.float32)
        y = torch.rand([64, 1], dtype=torch.float64)
        print('basic-head:', loss(x, y))


# --------------------------------------------
#              test dataset
# --------------------------------------------

def test_dataloader(args):
    from dataset import RegressionDataset, get_transform
    cfg = setup(args.cfg)
    train_trans = get_transform(train=True)
    valid_trans = get_transform(train=False)
    train_dataset = RegressionDataset('train', train_trans, cfg)
    ori_dataset = RegressionDataset('train', valid_trans, cfg)
    print(train_dataset)
    im_tensors = []
    ori_tensors = []
    for i in np.random.randint(0, 1000, size=80):
        img = train_dataset.__getitem__(i)['inputs']
        im_tensors.append(img)
        ori_img = ori_dataset.__getitem__(i)['inputs']
        ori_tensors.append(ori_img)
    vis_tensor(im_tensors, ori_tensors)


def vis_tensor(im_tensors, ori_tensors, col=8):
    import cv2
    import numpy as np
    img_aug_vstack = None
    img_aug_hstack = None
    for i, (img, ori_img) in enumerate(zip(im_tensors, ori_tensors)):
        img_aug = (img.detach().numpy().transpose(
            (1, 2, 0))[:, :, ::-1] + 1) * 127.5
        ori_img = (ori_img.detach().numpy().transpose(
            (1, 2, 0))[:, :, ::-1] + 1) * 127.5
        img_ori_img = np.hstack((ori_img, img_aug))
        img_ori_img = cv2.resize(img_ori_img, (1024, 512))
        # print(i+1)
        # print("hstack", img_aug_vstack.shape if img_aug_vstack is not None else 0)
        # print("vstack", img_aug_vstack.shape if img_aug_vstack is not None else 0)
        if img_aug_hstack is None:
            img_aug_hstack = img_ori_img
        else:
            img_aug_hstack = np.hstack((img_aug_hstack, img_ori_img))
        if (i+1) % col == 0:
            if img_aug_vstack is None:
                img_aug_vstack = img_aug_hstack.copy()
            else:
                img_aug_vstack = np.vstack((img_aug_vstack, img_aug_hstack))
            img_aug_hstack = None
    cv2.imwrite('vis.jpg', img_aug_vstack)
    print('vis.jpg saved')


# --------------------------------------------
#              others
# --------------------------------------------

def redirect():
    import os
    from collections import defaultdict
    import shutil
    src = '../log/HandGestureRecognition/others'
    move_list = defaultdict(list)
    for folder in os.listdir(src):
        for f in os.listdir(os.path.join(src, folder)):
            if f.endswith('.log'):
                date = f[-23:-13]
                os.makedirs(os.path.join(
                    os.path.dirname(src), date), exist_ok=True)
                move_list[date].append(os.path.join(src, folder))
                break
    for date, values in move_list.items():
        for v in values:
            base = os.path.basename(v)
            dst = os.path.join(os.path.dirname(src), date, base)
            shutil.move(v, dst)


if __name__ == '__main__':
    import argparse
    parse = argparse.ArgumentParser()
    parse.add_argument('--cfg', type=str, default='./test.yaml', help='test config yaml')
    parse.add_argument('--mode', type=str, default='model', help='test mode')
    args = parse.parse_args()
    if args.mode == 'loss':
        test_loss(args)
    elif args.mode == 'model':
        test_model(args)
    elif args.mode == 'dataset':
        test_dataloader(args)
