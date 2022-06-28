import os
import cv2
import os.path as osp
from PIL import Image
import numpy as np


face_data = '/Volumes/Lexar/data/CelebA-HQ-img/CelebA-HQ-img'
face_sep_mask = '/Volumes/Lexar/data/CelebA-HQ-img/CelebAMask-HQ-mask-anno'
mask_path = '/Volumes/Lexar/data/CelebA-HQ-img/mask'


def mask_preprocess():
    counter = 0
    total = 0
    for i in range(15):

        atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
                'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']

        for j in range(i * 2000, (i + 1) * 2000):

            mask = np.zeros((512, 512))

            for i, att in enumerate(atts, 1):
                total += 1
                file_name = ''.join([str(j).rjust(5, '0'), '_', att, '.png'])
                path = osp.join(face_sep_mask, str(i), file_name)

                if os.path.exists(path):
                    counter += 1
                    sep_mask = np.array(Image.open(path).convert('P'))
                    mask[sep_mask == 225] = i
            cv2.imshow('{}/{}.png'.format(mask_path, j), mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            print(j)

    print(counter, total)


def main():
    face_data = '/Volumes/Lexar/CelebA-HQ-img/CelebA-HQ-img'
    im_list = os.listdir(face_data)
    tot = len(im_list)
    for i, im_nm in enumerate(im_list):
        if not im_nm.endswith('.jpg'):
            continue
        img = cv2.imread(os.path.join(face_data, im_nm))
        if img is None:
            print('{} is None'.format(im_nm))
        print('loading {}/{}.'.format(i, tot), flush=True, end='\r')


if __name__ == '__main__':
    mask_preprocess()
