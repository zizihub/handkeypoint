#!/usr/bin/python
# -*- encoding: utf-8 -*-
from seg_engine.dataset.transform import Compose, ColorJitter, HorizontalFlip, RandomCrop, RandomScale, ZeroPadding, Resize, MotionBlur
from seg_engine.dataset.dataset import SegmentationDataset
from collections import defaultdict
import torchvision.transforms as transforms
import albumentations as albu
import cv2


class FaceMask(SegmentationDataset):
    def __init__(self, cfg, mode='train'):
        super(FaceMask, self).__init__(
            rootpth_list=cfg.DATASET.TRAIN_PATH,
            n_classes=cfg.DATASET.NUM_CLASSES,
            mode=mode,
            cropsize=cfg.INPUT.CROP,
            resize=cfg.INPUT.SIZE,
            task=cfg.TASK,
            **cfg.DATASET.POSTPROCESS
        )

        n_classes = cfg.DATASET.NUM_CLASSES
        resize = cfg.INPUT.SIZE
        cropsize = cfg.INPUT.CROP
        task = cfg.TASK

        if n_classes+1 == 10:
            self.classes_name = {
                0: 'background',
                1: 'facial skin',
                2: 'left brow',
                3: 'right brow',
                4: 'left eye',
                5: 'right eye',
                6: 'nose',
                7: 'upper lip',
                8: 'inner mouth',
                9: 'lower lip',
            }
        elif n_classes+1 == 2:
            if 'Face' in task:
                self.classes_name = {
                    0: 'background',
                    1: 'face',
                }
            elif 'Hair' in task:
                self.classes_name = {
                    0: 'background',
                    1: 'hair'
                }
            else:
                raise ValueError()
        elif n_classes+1 == 3:
            self.classes_name = {
                0: 'background',
                1: 'hair',
                2: 'face',
            }
        elif n_classes+1 == 7:
            self.classes_name = {
                0: 'background',
                1: 'facial skin',
                2: 'brow',
                3: 'eye',
                4: 'nose',
                5: 'lip',
                6: 'inner mouth',
            }
        elif n_classes+1 == 8:
            self.classes_name = {
                0: 'background',
                1: 'facial skin',
                2: 'brow',
                3: 'eye',
                4: 'nose',
                5: 'inner mouth',
                6: 'upper lip',
                7: 'lower lip',
            }
        elif n_classes+1 == 12:
            self.classes_name = {
                0: 'background',
                1: 'facial skin',
                2: 'left brow',
                3: 'right brow',
                4: 'left eye',
                5: 'right eye',
                6: 'left ear',
                7: 'right ear',
                8: 'nose',
                9: 'upper lip',
                10: 'inner mouth',
                11: 'lower lip',
            }
        elif n_classes+1 == 18:
            # CVPR21
            self.classes_name = {
                0: 'background',
                1: 'face_skin',
                2: 'l_brow',
                3: 'r_brow',
                4: 'l_eye',
                5: 'r_eye',
                6: 'nose',
                7: 'l_lip',
                8: 'mouth',
                9: 'u_lip',
                10: 'hair',
                11: 'l_eye_shadow',
                12: 'r_eye_shadow',
                13: 'l_ear',
                14: 'r_ear',
                15: 'hat',
                16: 'e_glasses',
                17: 'else_skin',
            }
        else:
            # CelebAMask-HQ
            self.classes_name = {
                0: 'background',
                1: 'skin',
                2: 'l_brow',
                3: 'r_brow',
                4: 'l_eye',
                5: 'r_eye',
                6: 'eye_g',
                7: 'l_ear',
                8: 'r_ear',
                9: 'ear_r',
                10: 'nose',
                11: 'mouth',
                12: 'u_lip',
                13: 'l_lip',
                14: 'neck',
                15: 'neck_l',
                16: 'cloth',
                17: 'hair',
                18: 'hat',
            }

        if 'Face' in task:
            trans_train = [ZeroPadding()]
        elif 'Hair' in task:
            trans_train = []
        else:
            raise ValueError()

        trans_train.extend([
            MotionBlur(p=0.5),
            ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                p=0.5),
            HorizontalFlip(left_right=False),
            RandomScale((0.5, 0.75, 1.0, 1.25, 1.5)),
            Resize(resize),
            RandomCrop(cropsize)
        ])

        #  pre-processing
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.trans_train = Compose(trans_train)
        # four channels mask augmentation
        self.fc_trans = albu.Compose(
            albu.OneOf(
                [
                    albu.ElasticTransform(border_mode=cv2.BORDER_CONSTANT, value=0),
                    albu.ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT, value=0),
                    albu.CoarseDropout(),
                    albu.GridDistortion(border_mode=cv2.BORDER_CONSTANT, value=0),
                ], p=0.9)
        )

    def mask_distribution(self):
        mask_dic = defaultdict(int)
        for idx in range(len(self.imgs)):
            label = self.get_target(idx)
            if len(set(label.flatten().tolist())) > 2:
                print('[{}/{}] class index: {}'.format(idx, len(self.imgs), set(label.flatten().tolist())))
            for c in self.classes_name.keys():
                mask_dic[self.classes_name[c]] += label[label == c].shape[0]
        total_pixel = sum(mask_dic.values())
        for k, v in mask_dic.items():
            print('Class {:<15}: {}%'.format(k, round(v*100/total_pixel, 2)))
