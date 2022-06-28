#!/usr/bin/python
# -*- encoding: utf-8 -*-
from seg_engine.dataset.transform import Compose, ColorJitter, HorizontalFlip, RandomCrop, RandomScale, ZeroPadding, Resize, MotionBlur ,Rotate
from seg_engine.dataset.dataset import SegmentationDataset
from collections import defaultdict
import torchvision.transforms as transforms
import albumentations as albu
import cv2


class SkyParsing(SegmentationDataset):
    def __init__(self, cfg, mode='train'):
        super(SkyParsing, self).__init__(
            rootpth_list=cfg.DATASET.TRAIN_PATH,
            n_classes=cfg.DATASET.NUM_CLASSES,
            mode=mode,
            cropsize=cfg.INPUT.CROP,
            resize=cfg.INPUT.SIZE,
            task=cfg.TASK,
            **cfg.DATASET.POSTPROCESS
        )

        resize = cfg.INPUT.SIZE
        cropsize = cfg.INPUT.CROP
        task = cfg.TASK

        if self.n_classes+1 == 2:
            if 'SKY' in task:
                self.classes_name = {
                    0: 'background',
                    1: 'sky'
                }

        trans_train = [
            # ZeroPadding(),
            ColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.5,
                p=0.5),
            HorizontalFlip(left_right=False),
            Rotate(),
            # RandomScale((0.5, 0.75, 1.0, 1.25, 1.5)),
            Resize(resize),
            # RandomCrop(cropsize),
            
        ]
        #  pre-processing
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.trans_train = Compose(trans_train)

    def mask_distribution(self):
        mask_dic = defaultdict(int)
        for idx in range(len(self.imgs)):
            label = self.get_target(idx)
            if len(set(label.flatten().tolist())) > 2:
                print('[{}/{}] class index: {}'.format(idx,
                      len(self.imgs), set(label.flatten().tolist())))
            for c in self.classes_name.keys():
                mask_dic[self.classes_name[c]] += label[label == c].shape[0]
        total_pixel = sum(mask_dic.values())
        for k, v in mask_dic.items():
            print('Class {:<15}: {}%'.format(k, round(v*100/total_pixel, 2)))
