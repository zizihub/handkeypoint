#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os.path as osp
import os
import cv2
from PIL import Image
import numpy as np
from multiprocessing import Pool
import time
import shutil


def replace_cvpr21(mask, ori_mask):
    # facial skin
    mask[ori_mask == 1] = 1
    # left brow (viewer side)
    mask[ori_mask == 2] = 1
    # right brow
    mask[ori_mask == 3] = 1
    # left eye
    mask[ori_mask == 4] = 1
    # right eye
    mask[ori_mask == 5] = 1
    # nose
    mask[ori_mask == 6] = 1
    # lower lip
    mask[ori_mask == 7] = 1
    # inner mouth
    mask[ori_mask == 8] = 1
    # upper lip
    mask[ori_mask == 9] = 1
    #  hair
    mask[ori_mask == 10] = 0
    # left eye shadow
    mask[ori_mask == 11] = 1
    # right eye shadow
    mask[ori_mask == 12] = 1
    # left ear
    mask[ori_mask == 13] = 0
    # right ear
    mask[ori_mask == 14] = 0
    # hat
    mask[ori_mask == 15] = 0
    # glasses
    mask[ori_mask == 16] = 0
    # else skin
    mask[ori_mask == 17] = 0
    return mask


def replace_LAPA(mask, ori_mask):
    # facial skin
    mask[ori_mask == 1] = 1
    # left brow (viewer side)
    mask[ori_mask == 2] = 1
    # right brow
    mask[ori_mask == 3] = 1
    # left eye
    mask[ori_mask == 4] = 1
    # right eye
    mask[ori_mask == 5] = 1
    # nose
    mask[ori_mask == 6] = 1
    # upper lip
    mask[ori_mask == 7] = 1
    # inner mouth
    mask[ori_mask == 8] = 1
    # lower lip
    mask[ori_mask == 9] = 1
    # hair
    mask[ori_mask == 10] = 0
    return mask


def replace_CelebA(mask, ori_mask):
    # facial skin
    mask[ori_mask == 1] = 1
    # left brow
    mask[ori_mask == 2] = 1
    # right brow
    mask[ori_mask == 3] = 1
    # left eye
    mask[ori_mask == 4] = 1
    # right eye
    mask[ori_mask == 5] = 1
    # eye glasses
    mask[ori_mask == 6] = 0
    # left ear
    mask[ori_mask == 7] = 0
    # right ear
    mask[ori_mask == 8] = 0
    # ear rings
    mask[ori_mask == 9] = 0
    # nose
    mask[ori_mask == 10] = 1
    #  mouth
    mask[ori_mask == 11] = 1
    # upper lip
    mask[ori_mask == 12] = 1
    # lower lip
    mask[ori_mask == 13] = 1
    # neck
    mask[ori_mask == 14] = 0
    # necklace
    mask[ori_mask == 15] = 0
    # cloth
    mask[ori_mask == 16] = 0
    # hair
    mask[ori_mask == 17] = 0
    # hat
    mask[ori_mask == 18] = 0
    # hand
    mask[ori_mask == 19] = 0
    return mask


def replace_KHair500k(mask, ori_mask):
    mask[ori_mask > 127] = 1
    mask[ori_mask <= 127] = 0
    return mask


class SegmentPreprocessor:
    def __init__(self, num_class='face') -> None:
        # CelebA-HQ process
        self.face_sep_mask = '/data2/zhangziwei/datasets/CelebA-HQ-img/CelebAMask-HQ-mask-anno'
        self.mask_path = '/data2/zhangziwei/datasets/CelebA-HQ-img/mask-{}'.format(num_class)
        self.num = num_class

        # Others
        self.mode = 'KHair500k'
        if self.mode == 'CVPR21':
            # CVPR21
            self.src = '/data2/zhangziwei/datasets/CVPR21-SFPC/mask-17'
            self.dst = '/data2/zhangziwei/datasets/CVPR21-SFPC/mask-{}'.format(num_class)
        elif self.mode == 'CVPR21-Crop':
            # CVPR21 Crop
            self.src = '/data2/zhangziwei/datasets/CVPR21-SFPC-Crop/mask-17'
            self.dst = '/data2/zhangziwei/datasets/CVPR21-SFPC-Crop/mask-{}'.format(num_class)
        elif self.mode == 'LAPA':
            # LAPA
            self.src = '/data2/zhangziwei/datasets/LaPa/labels'
            self.dst = '/data2/zhangziwei/datasets/LaPa/mask-{}'.format(num_class)
        elif self.mode == 'CelebA-Occ':
            # Synthesis CelebA with Hand
            self.src = '/data2/zhangziwei/datasets/CelebA-HQ-Occlusion/mask-full'
            self.dst = '/data2/zhangziwei/datasets/CelebA-HQ-Occlusion/mask-{}'.format(num_class)
        elif self.mode == 'KHair500k':
            self.src = '/data2/zhangziwei/datasets/KHairstyle500k/mask'
            self.dst = '/data2/zhangziwei/datasets/KHairstyle500k/mask-{}'.format(num_class)
        else:
            raise ValueError

    def _get_mask_cfg(self, num=2):
        if num == 7:
            atts = ['skin', 'l_brow', 'r_brow', 'l_eye',
                    'r_eye', 'nose', 'u_lip', 'mouth', 'l_lip']
            classes = {'skin': 1,
                       'l_brow': 2,
                       'r_brow': 2,
                       'l_eye': 3,
                       'r_eye': 3,
                       'nose': 4,
                       'u_lip': 5,
                       'mouth': 6,
                       'l_lip': 7, }
        elif num == 'face':
            atts = {'skin', 'l_brow', 'r_brow', 'l_eye',
                    'r_eye', 'nose', 'u_lip', 'mouth', 'l_lip'}
            classes = {'skin': 1,
                       'l_brow': 1,
                       'r_brow': 1,
                       'l_eye': 1,
                       'r_eye': 1,
                       'nose': 1,
                       'u_lip': 1,
                       'mouth': 1,
                       'l_lip': 1, }
        elif num == 6:
            atts = {'skin', 'l_brow', 'r_brow', 'l_eye',
                    'r_eye', 'nose', 'u_lip', 'mouth', 'l_lip'}
            classes = {'skin': 1,
                       'l_brow': 2,
                       'r_brow': 2,
                       'l_eye': 3,
                       'r_eye': 3,
                       'nose': 4,
                       'u_lip': 5,
                       'mouth': 6,
                       'l_lip': 5, }
        elif num == 1:
            atts = ['hair']
            classes = {'hair': 1, }
        elif num == 'face-hair':
            atts = ['hair', 'skin', 'l_brow', 'r_brow', 'l_eye',
                    'r_eye', 'eye_g', 'nose', 'l_ear', 'r_ear',
                    'u_lip', 'mouth', 'l_lip']
            classes = {'hair': 1,
                       'skin': 2,
                       'l_brow': 2,
                       'r_brow': 2,
                       'l_eye': 2,
                       'r_eye': 2,
                       'eye_g': 2,
                       'l_ear': 2,
                       'r_ear': 2,
                       'nose': 2,
                       'u_lip': 2,
                       'mouth': 2,
                       'l_lip': 2, }
        elif num == 11:
            atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye',
                    'l_ear', 'r_ear', 'nose', 'u_lip', 'mouth', 'l_lip']

            classes = {
                'skin': 1,
                'l_brow': 2,
                'r_brow': 3,
                'l_eye': 4,
                'r_eye': 5,
                'l_ear': 6,
                'r_ear': 7,
                'nose': 8,
                'u_lip': 9,
                'mouth': 10,
                'l_lip': 11,
            }
        elif num == 17:
            atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'nose',  'l_lip',
                    'mouth', 'u_lip',  'hair',  'l_ear', 'r_ear', 'hat', ]

            classes = {
                'skin': 1,
                'l_brow': 2,
                'r_brow': 3,
                'l_eye': 4,
                'r_eye': 5,
                'nose': 6,
                'l_lip': 7,
                'mouth': 8,
                'u_lip': 9,
                'hair': 10,
                'l_ear': 13,
                'r_ear': 14,
                'hat': 15,
                'eye_g': 16,
            }

        else:
            atts = ['skin', 'l_brow', 'r_brow', 'l_eye',
                    'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r', 'nose', 'u_lip', 'mouth',
                    'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']

            classes = {
                'skin': 1,
                'l_brow': 2,
                'r_brow': 3,
                'l_eye': 4,
                'r_eye': 5,
                'eye_g': 6,
                'l_ear': 7,
                'r_ear': 8,
                'ear_r': 9,
                'nose': 10,
                'mouth': 11,
                'u_lip': 12,
                'l_lip': 13,
                'neck': 14,
                'neck_l': 15,
                'cloth': 16,
                'hair': 17,
                'hat': 18,
            }

        return atts, classes

    def _preprocessing(self, i):
        atts, classes = self._get_mask_cfg(self.num)
        for j in range(i * 2000, (i + 1) * 2000):

            mask = np.zeros((512, 512))

            for l, att in enumerate(atts, 1):
                file_name = ''.join([str(j).rjust(5, '0'), '_', att, '.png'])
                path = osp.join(self.face_sep_mask, str(i), file_name)
                label = classes[att]
                if os.path.exists(path):
                    sep_mask = np.array(Image.open(path).convert('P'))
                    # print(np.unique(sep_mask))
                    mask[sep_mask == 225] = label
            cv2.imwrite('{}/{}.png'.format(self.mask_path, j), mask)
            print('Processing {}/{}...'.format(j, 30000), end='\r', flush=True)

    def mp_preprocessing(self):
        '''
        CelebA-HQ fragment preprocess
        '''
        os.makedirs(self.mask_path, exist_ok=True)
        start = time.time()
        p = Pool(16)
        for i in range(15):
            p.apply_async(self._preprocessing, args=(i,))
        p.close()
        p.join()
        print('\nTime cost: {}s'.format(round(time.time()-start, 4)))
        print('All subprocesses done.')

    def _change_mask(self, src_path, dst_path):
        def custom_mask(ori_mask, mode='CVPR21'):
            mask = ori_mask.copy()
            # CVPR21 & CVPR21 Crop
            if mode == 'CVPR21' or mode == 'CVPR21-Crop':
                mask = replace_cvpr21(mask, ori_mask)
            # LAPA
            elif mode == 'LAPA':
                mask = replace_LAPA(mask, ori_mask)
            elif mode == 'CelebA-Occ':
                mask = replace_CelebA(mask, ori_mask)
            elif mode == 'KHair500k':
                mask = replace_KHair500k(mask, ori_mask)
            else:
                raise ValueError
            return mask
        # read mask as grayscale image
        mask = np.array(Image.open(src_path).convert('L'))
        mask_new = custom_mask(mask, mode=self.mode)
        cv2.imwrite(dst_path, mask_new)
        print('pre_mask: %-50s  ==>  post_mask: %25s' % (str(np.unique(mask)), str(np.unique(mask_new))))

    def mp_change_mask(self):
        start = time.time()
        p = Pool(32)
        os.makedirs(self.dst, exist_ok=True)
        for nm in os.listdir(self.src):
            src_path = osp.join(self.src, nm)
            dst_path = osp.join(self.dst, nm)
            # print(src_path, dst_path)
            p.apply_async(self._change_mask, args=(src_path, dst_path,))
        p.close()
        p.join()
        print('\nTime cost: {}s'.format(round(time.time()-start, 4)))
        print('All subprocesses done.')


def folder_combine(src):
    dst = os.path.join(os.path.dirname(src), 'image')
    os.makedirs(dst, exist_ok=True)
    tot = 0
    cur = 0
    for root, dirs, files in os.walk(src):
        if dirs:
            continue
        tot += len(files)
        for f in files:
            cur += 1
            src_path = os.path.join(root, f)
            dst_path = os.path.join(dst, f)
            shutil.move(src_path, dst_path)
            print('[{}/{}]moving {} ====> {}'.format(cur, tot, src_path, dst_path))


def test_mask(src, col=2):
    img_list = np.random.choice(os.listdir(src), size=10)
    img_mask_vstack = None
    img_mask_hstack = None
    for i, img in enumerate(img_list):
        img_p = os.path.join(src, img)
        mask_path = img_p.replace('image', 'mask-hair').replace('jpg', 'png').replace('JPG', 'png')
        print(i, img_p)
        print(i, mask_path)
        img = cv2.imread(img_p)
        mask = cv2.imread(mask_path)*100
        img = cv2.resize(img, (mask.shape[1], mask.shape[0]))
        img_mask = np.hstack((img, mask))
        img_mask = cv2.resize(img_mask, (1024, 512))
        if i % col != 0 and i != 0:
            img_mask_hstack = np.hstack((img_mask_hstack, img_mask))
            if isinstance(img_mask_vstack, np.ndarray):
                img_mask_vstack = np.vstack((img_mask_vstack, img_mask_hstack))
            else:
                img_mask_vstack = img_mask_hstack.copy()
            img_mask_hstack = None
        else:
            img_mask_hstack = img_mask
    cv2.imwrite('vis.jpg', img_mask_vstack)


# ----------------------------------------------------------------
# Matting Preprocess Functions
# ----------------------------------------------------------------

def split_matting_from_fgr():
    src = '/data1/zhangziwei/datasets/PortraitDataset/image matting/HumanHalf/four_channel_imgs'
    for root, dirs, files in os.walk(src):
        if files:
            continue
        for d in dirs:
            if not d.startswith('matting'):
                continue
            print(os.path.join(root, d))
            os.rename(os.path.join(root, d), os.path.join(root, d).replace('matting_', ''))


if __name__ == "__main__":
    # sp = SegmentPreprocessor(num_class='hair')
    # sp.mp_change_mask()
    # test_mask('/data2/zhangziwei/datasets/KHairstyle500k/image')
    # sp.mp_preprocessing()
    split_matting_from_fgr()
