from PIL import Image, ImageOps
import random
import cv2
import numpy as np
import os


class ImageSynthese:
    def __init__(self):
        self.save_path = '/Volumes/Lexar/data/face_data/CelebA-HQ-Occlusion2'
        self.fg_folder = '/Volumes/Lexar/data/hand_data/RealHands/{}/color'.format(random.choice(["user02", "user03"]))
        self.bg_folder = '/Volumes/Lexar/data/face_data/CelebA-HQ/CelebA-HQ-img'
        self.mask_folder = 'mask-full'
        self.hand_class = 18

    @staticmethod
    def cv2_show_image(name, img):
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _random_select(self, mask='hand'):
        if mask == 'hand':
            img_p = os.path.join(self.fg_folder, random.choice(os.listdir(self.fg_folder)))
            mask_p = img_p.replace('color', 'mask').replace('_mask', '')
            # mask_p = img_p.replace('white', 'masks')
        elif mask == 'face':
            img_p = os.path.join(self.bg_folder, random.choice(os.listdir(self.bg_folder)))
            mask_p = img_p.replace('CelebA-HQ-img', self.mask_folder).replace('.jpg', '.png')
        else:
            raise NotImplementedError
        return img_p, mask_p

    def _color_adjust(self, img, mask, img_face, mask_face):
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2YUV)
        img_face = cv2.cvtColor(np.array(img_face), cv2.COLOR_RGB2YUV)
        mask = np.array(mask)
        hand_yuv = []
        face_yuv = []
        # hand yuv
        for r in range(mask.shape[0]):
            for c in range(mask.shape[1]):
                if mask[r][c] == 255:
                    hand_yuv.append(img[r][c])
        hand_yuv_mean = np.mean(hand_yuv, axis=0)
        # face yuv
        for r in range(mask_face.shape[0]):
            for c in range(mask_face.shape[1]):
                if mask_face[r][c] == 1:
                    face_yuv.append(img_face[r][c])
        face_yuv_mean = np.mean(face_yuv, axis=0)
        # yuv range [0, 255]
        img = np.clip(img - hand_yuv_mean + face_yuv_mean, 0, 255)
        return Image.fromarray(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_YUV2RGB))

    def _random_aug(self, img, mask, img_face, mask_face):
        img = self._color_adjust(img, mask, img_face, mask_face)
        H, W = img_face.size
        h, w = img.size
        if random.random() < 0.33:
            img = img.rotate(270, Image.NEAREST, expand=1)
            mask = mask.rotate(270, Image.NEAREST, expand=1)
            x, y = 0, 0
        elif random.random() < 0.66:
            img = img.rotate(90, Image.NEAREST, expand=1)
            mask = mask.rotate(90, Image.NEAREST, expand=1)
            x, y = H-w, 0
        else:
            x, y = H-h, W-w

        return img, mask, x, y

    def _mask_bin(self, mask):
        mask = np.array(mask)
        # mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.erode(mask, np.ones((3, 3), dtype=np.uint8), iterations=3)
        mask = cv2.blur(mask, (5, 5))
        mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        return Image.fromarray(mask)

    def _gen_synthesis_img(self):
        os.makedirs(os.path.join(self.save_path, 'image'), exist_ok=True)
        os.makedirs(os.path.join(self.save_path, self.mask_folder), exist_ok=True)

        src_img, src_mask = self._random_select(mask='hand')
        dst_img, dst_mask = self._random_select(mask='face')
        # dst
        img_face = Image.open(dst_img)
        mask_face = cv2.resize(cv2.imread(dst_mask, 0), img_face.size,
                               interpolation=cv2.INTER_NEAREST)
        hand_resize = (max(img_face.size)*4//5, max(img_face.size)*16//25)
        # src
        mask_hand = Image.open(src_mask).convert('L').resize(hand_resize, Image.NEAREST)
        mask_hand = self._mask_bin(mask_hand)
        img_hand = Image.open(src_img).resize(hand_resize, Image.BILINEAR)

        # random augmentation··
        img_hand, mask_hand, x, y = self._random_aug(img_hand, mask_hand, img_face, mask_face)

        img_face.paste(img_hand, (x, y), mask_hand)
        # mask paste
        mask_hand = np.array(mask_hand)
        for r in range(mask_hand.shape[0]):
            for c in range(mask_hand.shape[1]):
                if mask_hand[r][c] == 255:
                    mask_face[r+y][c+x] = self.hand_class
        mask_face = cv2.resize(mask_face, img_face.size, interpolation=cv2.INTER_NEAREST)
        sys_img_save = os.path.join(self.save_path, 'image', os.path.splitext(os.path.basename(dst_img))[0]+'_oc.jpg')
        sys_mask_save = os.path.join(self.save_path, self.mask_folder,
                                     os.path.splitext(os.path.basename(dst_mask))[0]+'_oc.png')
        print(np.unique(mask_face))
        print('>>> Done:', sys_img_save)
        if 0:
            cv2.imshow('demo', np.array(img_face)[:, :, ::-1])
            cv2.imshow('mask', mask_face*10)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        img_face.save(sys_img_save)
        cv2.imwrite(sys_mask_save, mask_face)

    def mp_syn_img(self):
        from multiprocessing import Pool
        p = Pool(4)
        for _ in range(10000):
            p.apply_async(self._gen_synthesis_img)
        p.close()
        p.join()

    def check_img_mask(self):
        img_folder = os.path.join(self.save_path, 'image')
        for f in os.listdir(img_folder):
            img_path = os.path.join(img_folder, f)
            mask_path = img_path.replace('image', self.mask_folder).replace('jpg', 'png')
            img_check, mask_check = os.path.exists(img_path), os.path.exists(mask_path)
            assert img_check is True, "invalid {} image path".format(img_path)
            assert mask_check is True, "invalid {} mask path".format(mask_path)
            print('>>> Done check', f)


if __name__ == "__main__":
    func = ImageSynthese()
    # func._gen_synthesis_img()
    func.mp_syn_img()
