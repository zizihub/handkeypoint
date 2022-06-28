#!/usr/bin/python
# -*- encoding: utf-8 -*-
import os
import torch
import random
from PIL import Image
from seg_engine.dataset.transform_mat import MotionAugmentation
from seg_engine.dataset.dataset import MattingDataset


class PortraitMatting(MattingDataset):
    def __init__(self, cfg, mode='train'):
        resize = cfg.INPUT.SIZE
        self.mode = mode
        if mode == 'train':
            self.transform = VideoMatteTrainAugmentation(resize)
            rootpth = cfg.DATASET.TRAIN_PATH
        elif mode == 'val':
            self.transform = VideoMatteValidAugmentation(resize)
            rootpth = cfg.DATASET.TEST_PATH
        else:
            raise ValueError
        self.n_classes = 1
        self.classes_name = {}
        super(PortraitMatting, self).__init__(
            mode=mode,
            size=cfg.INPUT.SIZE[0],
            rootpth_list=rootpth,
        )


class PortraitSegmentation(MattingDataset):
    def __init__(self, cfg, mode='train'):
        resize = cfg.INPUT.SIZE
        self.mode = mode
        if mode == 'train':
            self.transform = VideoMatteTrainAugmentation(resize)
            rootpth = cfg.DATASET.TRAIN_PATH
        elif mode == 'val':
            self.transform = VideoMatteValidAugmentation(resize)
            rootpth = cfg.DATASET.TEST_PATH
        else:
            raise ValueError
        self.n_classes = 1
        self.classes_name = {
            0: 'background',
            1: 'foreground'
        }
        super(PortraitSegmentation, self).__init__(
            mode=mode,
            size=cfg.INPUT.SIZE[0],
            rootpth_list=rootpth,
        )

    def __getitem__(self, idx):
        fgrs, phas = self._get_fgr_pha(idx)
        img_path = self.img_path_list[idx].replace('/fgr/', '/image/')
        if os.path.exists(img_path):
            imgs = [self._downsample_if_needed(Image.open(img_path).convert('RGB'))]
            if self.transform is not None:
                fgrs, phas, imgs = self.transform(fgrs, phas, imgs)
        else:
            if self.transform is not None:
                if random.random() < 0.5:
                    bgrs = self._get_random_image_background()
                else:
                    bgrs = self._get_random_video_background()
                fgrs, phas, bgrs = self.transform(fgrs, phas, bgrs)
                imgs = self._gen_fake_img(fgrs, phas, bgrs)
        return {
            'img': imgs.squeeze(0),
            'label': torch.where(phas > 0.5, 1, 0).squeeze(0).type(torch.int64),
            'foreground': fgrs.squeeze(0),
        }


class VideoMatteTrainAugmentation(MotionAugmentation):
    def __init__(self, size):
        super().__init__(
            size=size,
            prob_fgr_affine=0.3,
            prob_bgr_affine=0.3,
            prob_noise=0.1,
            prob_color_jitter=0.3,
            prob_grayscale=0.02,
            # prob_sharpness=0.1,
            prob_blur=0.02,
            prob_hflip=0.5,
            # prob_pause=0.03,
        )


class VideoMatteValidAugmentation(MotionAugmentation):
    def __init__(self, size):
        super().__init__(
            size=size,
            prob_fgr_affine=0,
            prob_bgr_affine=0,
            prob_noise=0,
            prob_color_jitter=0,
            prob_grayscale=0,
            # prob_sharpness=0,
            prob_blur=0,
            prob_hflip=0,
            # prob_pause=0,
            static_affine=False,
            aspect_ratio_range=(1.0, 1.0),
        )
