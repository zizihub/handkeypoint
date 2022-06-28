from albumentations.augmentations import transforms
import cv2
import os
import os.path as osp
import pandas as pd
import numpy as np
import shutil
from PIL import Image, ImageSequence
from collections import defaultdict
import albumentations as albu

transform = albu.Compose(
    transforms=[
        # albu.RandomResizedCrop(360, 240, scale=(0.5, 0.75), p=1.0)
        albu.Resize(512, 512)
    ])
helen_idx = {
    0: 'background',
    1: 'facial skin',
    2: 'left brow (viewer side)',
    3: 'right brow',
    4: 'left eye',
    5: 'right eye',
    6: 'nose',
    7: 'upper lip',
    8: 'inner mouth',
    9: 'lower lip',
    10: 'hair',
}
celebA_idx = {
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


def custom_mask(mask):
    if 1:
        mask[mask == 3] = 2     # 统一眉毛
        mask[mask == 4] = 3
        mask[mask == 5] = 3     # 统一眼睛
        mask[mask == 6] = 4
        mask[mask == 7] = 5
        mask[mask == 8] = 6
        mask[mask == 9] = 7
        mask[mask == 10] = 0    # 头发置为背景
    return mask


class ImageInspector(object):
    def __init__(self, src_path, mask_path=None, transform=None) -> None:
        super().__init__()
        self.src_path = src_path
        self.mask_path = mask_path
        if 0:
            self.op_dir = '/Volumes/Lexar/data/helenstar_release'
            self.face_op_dir = osp.join(self.op_dir, 'face')
            self.mask_op_dir = osp.join(self.op_dir, 'mask')
            os.makedirs(self.face_op_dir, exist_ok=True)
            os.makedirs(self.mask_op_dir, exist_ok=True)
        self.imshow = False
        self.imshow_mask = True
        self.transform = transform
        self.input_size = 512

    def cls_process(self, image):
        print(image.shape)

    def inspect_image(self):
        cur, tot = 0, 0
        shape_dic = defaultdict(int)
        for root, dirs, files in os.walk(self.src_path):
            if dirs:
                continue
            tot += len(files)
            for im_name in files:
                cur += 1
                print('loading image\t{}/{}...'.format(cur, tot))
                img = np.array(Image.open(osp.join(root, im_name)).convert('RGB'))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                if self.transform:
                    img = self.transform(image=img)['image']
                if img is None:
                    print('\nType None: {}\n'.format(osp.join(root, im_name)))
                    continue
                if self.imshow:
                    self.cls_process(img)
                    cv2.imshow(im_name, img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                elif self.imshow_mask:
                    mask = custom_mask(self.inspect_mask(im_name))
                    print('class index:', set(mask.flatten().tolist()))
                    # box = self.get_face_box(mask)
                    # img = self.face_crop(img, box)
                    # mask = self.face_crop(mask, box)
                    # self.save_img_mask(im_name, img, mask)
                    self.vis_parsing_maps(im=img,
                                          parsing_anno=mask,
                                          stride=1)
                shape_dic[img.shape] += 1
            print('{} folder has {}'.format(root, shape_dic))

    def get_face_box(self, mask):
        mask_tmp = mask.copy()
        mask_tmp[mask_tmp == 0] = 255
        ret, thresh = cv2.threshold(mask_tmp, 127, 255, cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        for contour in contours:
            cnt = contour if len(cnt) < len(contour) else cnt
        x, y, w, h = cv2.boundingRect(cnt)
        # print(x, y, w, h)
        # cv2.rectangle(mask_tmp, (x, y), (x+w, y+h), (0, 0, 255), 2)
        # cv2.imshow('show', mask_tmp)
        # cv2.waitKey(0)
        return (x, y, x+w, y+h)

    def face_crop(self, image, box):
        ratio = 0.1
        face_img = image[int(box[1]*(1-ratio)):int(box[3]*(1+ratio)),
                         int(box[0]*(1-ratio)):int(box[2]*(1+ratio))]
        if face_img is None:
            return image
        face_img = cv2.resize(face_img, (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)
        return face_img

    def inspect_mask(self, img_nm):
        mask = np.array(Image.open(osp.join(self.mask_path, img_nm.replace('.jpg', '.png'))).convert('L'))
        mask = cv2.resize(mask, (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)
        # print('class index:', set(mask.flatten().tolist()))
        print('image shape:', mask.shape)
        return mask

    def save_img_mask(self, im_name, img, mask):
        cv2.imwrite(osp.join(self.face_op_dir, im_name), img)
        cv2.imwrite(osp.join(self.mask_op_dir, im_name.replace('jpg', 'png')), mask)

    def vis_parsing_maps(self, im, parsing_anno, stride):
        # Colors for all 20 parts
        part_colors = [[255, 0, 0], [255, 85, 0], [105, 128, 112],
                       [18, 153, 255], [255, 0, 170],
                       [0, 255, 0], [85, 0, 255], [170, 255, 0],
                       [0, 255, 85], [0, 255, 170],
                       [0, 0, 255], [85, 0, 255], [170, 0, 255],
                       [0, 85, 255], [0, 170, 255],
                       [255, 255, 0], [255, 255, 85], [255, 255, 170],
                       [255, 0, 255], [255, 85, 255], [255, 170, 255],
                       [0, 255, 255], [85, 255, 255], [170, 255, 255]]
        vis_im = im.copy().astype(np.uint8)
        vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
        vis_parsing_anno = cv2.resize(
            vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
        vis_parsing_anno_color = np.zeros(
            (vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

        num_of_class = np.max(vis_parsing_anno)

        for pi in range(1, num_of_class + 1):
            index = np.where(vis_parsing_anno == pi)
            vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

        vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
        # print(vis_parsing_anno_color.shape, vis_im.shape)
        vis_im = cv2.addWeighted(cv2.cvtColor(
            vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)
        cv2.imshow('demo', np.hstack([im, vis_im]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def seperate_data_helenstar():
    src_path = '/Volumes/Lexar/data/helenstar_release'
    for folder in ['train', 'test']:
        cur = osp.join(src_path, folder)
        for im_nm in os.listdir(cur):
            if 'image' in im_nm:
                src = osp.join(cur, im_nm)
                dst = osp.join(src_path, 'helen_star_image', im_nm)
                shutil.move(src, dst)
                print('moving {}...'.format(im_nm))
            elif 'label' in im_nm:
                src = osp.join(cur, im_nm)
                dst = osp.join(src_path, 'helen_star_label', im_nm)
                shutil.move(src, dst)
                print('moving {}...'.format(im_nm))
            else:
                continue


def change_name_helenstar():
    src_path = '/Volumes/Lexar/data/helenstar_release'
    for folder in ['helen_star_image', 'helen_star_label']:
        cur = osp.join(src_path, folder)
        for im_nm in os.listdir(cur):
            src = osp.join(cur, im_nm)
            im_nm = im_nm.replace('_label', '').replace('_image', '')
            dst = osp.join(cur, im_nm)
            os.rename(src, dst)


if __name__ == '__main__':
    II = ImageInspector(src_path='/Volumes/Lexar/data/LaPa/train/images',
                        mask_path='/Volumes/Lexar/data/LaPa/train/labels',
                        transform=transform)
    II.inspect_image()
