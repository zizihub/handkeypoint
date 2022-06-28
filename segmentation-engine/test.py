#!/usr/bin/python
# -*- encoding: utf-8 -*-

from seg_engine.dataset.transform import Resize, ZeroPadding
from seg_engine.models.build import build_model
from seg_engine.config import get_cfg, get_outname
from seg_engine.utils import vis_parsing_maps
from torchvision.transforms.functional import to_grayscale, to_tensor, normalize
import torch

import os
import sys
import os.path as osp
import numpy as np
from PIL import Image, ImageOps
import torchvision.transforms as transforms
import cv2
from copy import deepcopy
from tqdm import tqdm
from time import time
import torch.nn.functional as F
# sys.path.append('../Ultra-Light-Fast-Generic-Face-Detector-1MB')
# from vision.utils.misc import Timer  # NOQA: 402
# from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor  # NOQA: 402
# from vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor  # NOQA: 402
# from vision.ssd.config.fd_config import define_img_size  # NOQA:402


def setup(my_cfg='./face_config.yaml'):
    '''
    Create configs and perform basic setups.
    '''
    cfg = get_cfg()
    cfg.merge_from_file(my_cfg)
    output_name = get_outname(cfg)
    cfg.merge_from_list(['OUTPUT_NAME', output_name])
    cfg.freeze()
    return cfg, my_cfg


def mask_recover(frame, vis_parsing_anno_color, box, face_img, ratio):
    frame = np.array(frame)
    face_img = np.array(face_img)
    vis_im = frame.copy().astype(np.uint8)
    vis_parsing_anno_reverse = cv2.resize(
        vis_parsing_anno_color, (face_img.shape[1], face_img.shape[0]))
    blank_mat = np.zeros(
        (frame.shape[0], frame.shape[1], 3), dtype=np.uint8) + 255
    blank_mat[int(box[1]*(1-ratio)):int(box[3]*(1+ratio)),
              int(box[0]*(1-ratio)):int(box[2]*(1+ratio))] = vis_parsing_anno_reverse
    vis_img = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_BGR2RGB), 0.4, blank_mat, 0.6, 0)
    return vis_img, blank_mat


def face_model_init():
    input_img_size = 640
    # must put define_img_size() before 'import create_mb_tiny_fd, create_mb_tiny_fd_predictor'
    define_img_size(input_img_size)
    label_path = "../Ultra-Light-Fast-Generic-Face-Detector-1MB/models/voc-model-labels.txt"
    class_names = [name.strip() for name in open(label_path).readlines()]
    test_device = 'cpu'
    candidate_size = 1000
    threshold = 0.7
    model_path = "../Ultra-Light-Fast-Generic-Face-Detector-1MB/models/pretrained/version-slim-640.pth"
    net = create_mb_tiny_fd(
        len(class_names), is_test=True, device=test_device)
    predictor = create_mb_tiny_fd_predictor(
        net, candidate_size=candidate_size, device=test_device)
    net.load(model_path)
    return predictor, threshold, candidate_size


def postprocessing(img, parsing):
    eye_mat = np.array(img).copy()
    eye_mat = cv2.cvtColor(eye_mat, cv2.COLOR_RGB2GRAY)
    eye_mat[parsing != 3] = 0
    ret, thres = cv2.threshold(eye_mat, 127, 255, cv2.THRESH_BINARY)
    parsing[thres == 255] = 8
    return parsing


@torch.no_grad()
def test(config=None):
    FACE_DETECT = False
    if config is None:
        config, _ = setup()
    respth = osp.join('./log', config.TASK, config.DATE, config.OUTPUT_NAME, 'demo_fuse')
    dspth = '/data2/zhangziwei/datasets/CelebA-HQ-img/image'
    os.makedirs(respth, exist_ok=True)
    net = deepcopy(build_model(config))
    net.cuda()
    save_pth = osp.join('./log', config.TASK, config.DATE, config.OUTPUT_NAME,
                        '{}_best.pth'.format(config.OUTPUT_NAME))
    net.load_state_dict(torch.load(save_pth)['net'])
    net.eval()

    transform = transforms.Compose([
        ZeroPadding(),
        transforms.Resize((config.INPUT.SIZE[0], config.INPUT.SIZE[1])),
    ])
    cost = []

    if FACE_DETECT:
        # init face-model
        predictor, threshold, candidate_size = face_model_init()

    for i, image_path in enumerate(tqdm(os.listdir(dspth)[:])):
        if i % 1000 != 0:
            continue
        img_rgb = Image.open(osp.join(dspth, image_path))
        img_size = img_rgb.size
        # use face detect model
        if FACE_DETECT:
            img_rgb = np.array(img_rgb)
            boxes, labels, probs = predictor.predict(
                img_rgb, candidate_size / 2, threshold)
            if boxes.size(0) == 0:
                continue
            box = boxes[0, :]
            ratio = 0.5
            face_img = img_rgb[int(box[1]*(1-ratio)):int(box[3]*(1+ratio)),
                               int(box[0]*(1-ratio)):int(box[2]*(1+ratio))]
            face_img = Image.fromarray(face_img)
            if face_img is None:
                continue
            image = face_img.resize((config.INPUT.SIZE[0], config.INPUT.SIZE[1]), Image.BILINEAR)
            img_rgb = Image.fromarray(img_rgb)
        else:
            image = ImageOps.pad(img_rgb, (max(img_size), max(img_size)))
        if config.DATASET.POSTPROCESS.four_channel:
            # mask_channel = Image.new('L', (config.INPUT.SIZE[0], config.INPUT.SIZE[1]), 0)
            mask_channel = Image.open(osp.join(dspth, image_path).replace('jpg', 'png').replace('image', 'mask-face'))
            mask_channel = normalize(to_tensor(transform(mask_channel)), mean=(0.5), std=(0.5))
            # image = img_rgb.resize((config.INPUT.SIZE[0], config.INPUT.SIZE[1]), Image.BILINEAR)
        img = normalize(to_tensor(transform(image)), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        if config.DATASET.POSTPROCESS.four_channel:
            img = torch.cat([img, mask_channel])
        img = torch.unsqueeze(img, 0)
        img = img.cuda()
        start = time()
        out = net(img)
        preds = out['masks']
        if 'fine' in out:
            preds = out['fine']
        cost.append(time() - start)

        if FACE_DETECT:
            # preds upsample to ori
            preds = F.interpolate(preds,
                                  size=(512, 512),
                                  mode='bilinear',
                                  align_corners=False)
            parsing = preds.squeeze(0).cpu().numpy().argmax(0)
            # parsing = postprocessing(img_rgb.resize((512, 512), Image.BILINEAR), parsing)
            vis_im, mask = vis_parsing_maps(img_rgb.resize((512, 512), Image.BILINEAR),
                                            parsing,
                                            demo=False)
            vis_mask, mask = mask_recover(img_rgb, mask, box, face_img, ratio)
        else:
            # preds upsample to ori
            preds = F.interpolate(preds,
                                  size=image.size[::-1],
                                  mode='bilinear',
                                  align_corners=False)
            parsing = preds.squeeze(0).cpu().numpy().argmax(0)
            # parsing = postprocessing(img_rgb.resize((512, 512), Image.BILINEAR), parsing)
            vis_mask, mask = vis_parsing_maps(image,
                                              img_size,
                                              parsing,
                                              demo=True)
        # save mask
        # cv2.imwrite(osp.join(respth, image_path)[:-4] + '.png', mask)
        # save vis_img
        cv2.imwrite(osp.join(respth, image_path), vis_mask, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    print('avg cost: {}s'.format(np.mean(cost)))


if __name__ == "__main__":
    test()
