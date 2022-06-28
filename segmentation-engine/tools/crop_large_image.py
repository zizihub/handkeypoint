import os
import numpy as np
import cv2


class CropFaceFunc:
    def __init__(self):
        self.image_folder = '/data2/zhangziwei/datasets/CVPR21-SFPC/image'
        self.dst_image = '/data2/zhangziwei/datasets/CVPR21-SFPC-Crop/image'
        self.dst_mask = '/data2/zhangziwei/datasets/CVPR21-SFPC-Crop/mask-17'
        os.makedirs(self.dst_image, exist_ok=True)
        os.makedirs(self.dst_mask, exist_ok=True)

    @staticmethod
    def square_padding(x, y, w, h):
        ll = (w - h) // 2
        if ll >= 0:
            return x, y-ll, w, h+ll*2
        else:
            return x+ll, y, w-ll*2, h

    def binary_mask_to_box(self, binary_mask):
        binary_mask = np.array(binary_mask, np.uint8)
        contours, hierarchy = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        areas = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            areas.append(area)
        # 取最大面积的连通区域
        idx = areas.index(np.max(areas))
        x, y, w, h = cv2.boundingRect(contours[idx])
        # square padding
        x, y, w, h = self.square_padding(x, y, w, h)
        # ratio expanded
        img_h, img_w = binary_mask.shape
        ratio = 1.25
        w_r = (ratio-1) * w // 2
        h_r = (ratio-1) * h // 2
        bounding_box = [max(x-w_r, 0), max(y-h_r, 0), min(x+w+w_r, img_w), min(y+h+h_r, img_h)]
        return bounding_box

    def crop_face(self):
        tot = len(os.listdir(self.image_folder))
        for i, f in enumerate(os.listdir(self.image_folder)):
            img_p = os.path.join(self.image_folder, f)
            print('>>> [{}/{}] cropping {}'.format(i+1, tot, img_p))
            mask_p = img_p.replace('image', 'mask-17').replace('jpg', 'png')
            img = cv2.imread(img_p)
            mask = cv2.imread(mask_p, 0)
            bbox = self.binary_mask_to_box(mask)
            self.save_new_img_mask(img, mask, bbox, f)

    def save_new_img_mask(self, img, mask, bbox, f):
        x1, y1, x2, y2 = bbox
        img_dst = os.path.join(self.dst_image, f)
        mask_dst = os.path.join(self.dst_mask, f.replace('jpg', 'png'))
        cv2.imwrite(img_dst, img[int(y1):int(y2), int(x1):int(x2)])
        cv2.imwrite(mask_dst, mask[int(y1):int(y2), int(x1):int(x2)])


if __name__ == '__main__':
    cf = CropFaceFunc()
    cf.crop_face()
