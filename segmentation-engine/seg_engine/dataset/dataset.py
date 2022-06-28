#!/usr/bin/python
# -*- encoding: utf-8 -*-
from __future__ import division
import torch
import random
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_grayscale, to_tensor, normalize
import os.path as osp
import os
from PIL import Image
import numpy as np
import cv2
from .utils import mask_to_binary_edges
from .transform import Resize


class SegmentationDataset(Dataset):
    '''
    SegmentationDataset Folder structure:
        ---Datasets Folder:
            ---image: images folder, shape (H, W, C)
            ---mask: mask folder with .png grayscale mask, shape (H, W)
    '''

    def __init__(self, rootpth_list, n_classes, mode, cropsize, resize, task, *args, **kwargs):
        super(SegmentationDataset, self).__init__()
        assert mode in ('train', 'val', 'all')
        self.mode = mode
        self.ignore_lb = 255
        self.n_classes = n_classes
        self.classes_name = {}
        self.mask_size = ()
        self.resize = resize
        self.cropsize = cropsize
        self.rootpth_list = rootpth_list
        self.imgs = []
        # get combined datasets
        for rootpth in rootpth_list:
            if mode == 'train':
                self.imgs.extend(list(map(lambda x: os.path.join(rootpth, 'image', x),
                                          os.listdir(osp.join(rootpth, 'image'))[:-1500])))
            elif mode == 'val':
                self.imgs.extend(list(map(lambda x: os.path.join(rootpth, 'image', x),
                                          os.listdir(osp.join(rootpth, 'image'))[-1500:])))
            else:
                self.imgs.extend(list(map(lambda x: os.path.join(rootpth, 'image', x),
                                          os.listdir(osp.join(rootpth, 'image')))))

        if self.n_classes == 1 and 'Face' in task:
            self.lb_folder = 'face'
        elif self.n_classes == 1 and 'Hair' in task:
            self.lb_folder = 'hair'
        elif self.n_classes == 18:
            self.lb_folder = 'full'
        elif self.n_classes == 2:
            self.lb_folder = 'face-hair'
        else:
            self.lb_folder = self.n_classes
        self.resize_trans = Resize(resize)
        print('loading {} {} dataset...'.format(self.lb_folder, self.mode))
        self.imgs_num = len(self.imgs)
        self.grayscale = kwargs.pop('grayscale', False)
        self.edge_map = kwargs.pop('edge_map', False)
        self.four_channel = kwargs.pop('four_channel', False)

    def __getitem__(self, idx):
        impth = self.imgs[idx]
        img = Image.open(impth).convert('RGB')
        mask_dir = os.path.dirname(impth).replace(
            'image', 'mask-{}'.format(self.lb_folder))
        mask_nm = os.path.basename(impth).replace('jpg', 'png')
        maskpth = os.path.join(mask_dir, mask_nm)
        label = Image.open(maskpth).convert('L')
        im_lb = dict(im=img, lb=label)
        if self.mode == 'train':
            im_lb = self.trans_train(im_lb)
        else:
            im_lb = self.resize_trans(im_lb)
        img, label = im_lb['im'], im_lb['lb']
        img_tensor = self.to_tensor(img)
        if self.four_channel:
            mask_channel = self._random_aug_mask(label)
            mask_channel = to_tensor(mask_channel)
            # mask_channel = self._min_max_scaler(mask_channel)
            normalize(mask_channel, mean=(0.5), std=(0.5), inplace=True)
            img_tensor = torch.cat([img_tensor, mask_channel])
        label = np.array(label).astype(np.int64)[np.newaxis, :]
        # print(np.unique(label))
        # exit()
        #label[label == 225] = 1

        result = {'img': img_tensor, 'label': torch.from_numpy(label)}
        if self.grayscale:
            gray = to_tensor(to_grayscale(img))
            result.update(grayscale=gray)

        if self.edge_map:
            # _edgemap = np.array(mask_trained)
            _edgemap = label[:, :, :]  # c, h, w
            _edgemap = mask_to_binary_edges(
                _edgemap, 2, self.n_classes+1)  # h, w
            edgemap = torch.from_numpy(_edgemap).float()
            result.update(edge_map=edgemap)
        return result

    def get_target(self, idx):
        impth = self.imgs[idx]
        mask_dir = os.path.dirname(impth).replace(
            'image', 'mask-{}'.format(self.lb_folder))
        mask_nm = os.path.basename(impth).replace('jpg', 'png')
        maskpth = os.path.join(mask_dir, mask_nm)
        label = cv2.imread(maskpth, 0)
        return label

    def _random_aug_mask(self, mask):
        mask = np.array(mask).copy()
        if random.random() > 0.3:
            return Image.fromarray(self.fc_trans(image=mask)['image'])
        else:
            return Image.new('L', self.cropsize, 0)

    def _min_max_scaler(self, tensor):
        tensor = torch.from_numpy(
            np.array(tensor, np.int32, copy=False)).unsqueeze(0).float()
        divisor = tensor.max() - tensor.min()
        divisor = max(divisor, 1)
        return (tensor - tensor.min()) / divisor

    def __len__(self):
        return len(self.imgs)

    def __repr__(self):
        str_output = '='*27
        str_output += ('\n{} Dataset'.format(self.mode.upper()))
        str_output += ('\nDataset Path: {}'.format(self.rootpth_list))
        str_output += ('\nDataset total: {}'.format(len(self)))
        return str_output


class MattingDataset(Dataset):
    '''
    MattingDataset Folder structure:
        ---Datasets Folder:
            ---image: images folder, shape (H, W, C)
            ---matte: mask folder with .png grayscale mask, shape (H, W)
    '''

    def __init__(self, size, rootpth_list, mode):
        super(MattingDataset, self).__init__()
        assert mode in ('train', 'val', 'all')
        self.size = size
        # get combined datasets
        if mode == 'train':
            background_video_dir = '/data1/zhangziwei/datasets/PortraitDataset/video background/video_train_bg'
            background_image_dir = '/data1/zhangziwei/datasets/PortraitDataset/image background/image_train_bg_SD'
        elif mode == 'val':
            background_video_dir = '/data1/zhangziwei/datasets/PortraitDataset/video background/video_test_bg'
            background_image_dir = '/data1/zhangziwei/datasets/PortraitDataset/image background/image_test_bg'
        self.background_image_dir = background_image_dir
        self.background_image_files = os.listdir(background_image_dir)
        self.background_video_dir = background_video_dir
        self.background_video_clips = sorted(os.listdir(background_video_dir))
        self.background_video_frames = [sorted(os.listdir(os.path.join(background_video_dir, clip)))
                                        for clip in self.background_video_clips]
        self.rootpth_list = rootpth_list
        self.img_path_list = []
        for rootpth in rootpth_list:
            self.imagematte_dir = rootpth
            for root, dirs, files in os.walk(os.path.join(rootpth, 'fgr')):
                if dirs:
                    continue
                self.img_path_list.extend(
                    list(map(lambda x: os.path.join(root, x), files)))

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        fgrs, phas = self._get_fgr_pha(idx)
        img_path = self.img_path_list[idx].replace('/fgr/', '/image/')
        if os.path.exists(img_path):
            imgs = [self._downsample_if_needed(
                Image.open(img_path).convert('RGB'))]
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
            'label': phas.squeeze(0),
            'foreground': fgrs.squeeze(0),
        }

    def _gen_fake_img(self, fgrs, phas, bgrs):
        imgs = []
        for fgr, pha, bgr in zip(fgrs, phas, bgrs):
            img = fgr * pha + (1 - pha) * bgr
            imgs.append(img)
        imgs = torch.from_numpy(np.stack(imgs))
        return imgs

    def _get_random_image_background(self):
        with Image.open(os.path.join(self.background_image_dir, random.choice(self.background_image_files))) as bgr:
            bgr = self._downsample_if_needed(bgr.convert('RGB'))
        bgrs = [bgr]
        return bgrs

    def _get_random_video_background(self):
        clip_idx = random.choice(range(len(self.background_video_clips)))
        frame_count = len(self.background_video_frames[clip_idx])
        frame_idx = random.choice(range(frame_count))
        clip = self.background_video_clips[clip_idx]
        frame = self.background_video_frames[clip_idx][frame_idx]
        with Image.open(os.path.join(self.background_video_dir, clip, frame)) as bgr:
            bgr = self._downsample_if_needed(bgr.convert('RGB'))
        bgrs = [bgr]
        return bgrs

    def _get_fgr_pha(self, idx):
        with Image.open(self.img_path_list[idx]) as fgr, \
                Image.open(self.img_path_list[idx].replace('/fgr/', '/pha/')) as pha:
            fgr = self._downsample_if_needed(fgr.convert('RGB'))
            pha = self._downsample_if_needed(pha.convert('L'))
        fgrs = [fgr]
        phas = [pha]
        return fgrs, phas

    def _downsample_if_needed(self, img):
        w, h = img.size
        if min(w, h) > self.size:
            scale = self.size / min(w, h)
            w = int(scale * w)
            h = int(scale * h)
            img = img.resize((w, h))
        return img

    def __repr__(self):
        str_output = '='*27
        str_output += ('\n{} Dataset'.format(self.mode.upper()))
        str_output += ('\nDataset Path: {}'.format(self.rootpth_list))
        str_output += ('\nDataset total: {}'.format(len(self)))
        return str_output
