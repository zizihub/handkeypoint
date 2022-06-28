#!/usr/bin/python
# -*- encoding: utf-8 -*-
from PIL import Image, ImageOps
import PIL.ImageEnhance as ImageEnhance
import albumentations.augmentations.functional as F
import random
import numpy as np
import cv2


class ZeroPadding(object):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, im_lb):
        if isinstance(im_lb, dict):
            im = im_lb['im']
            lb = im_lb['lb']
            w, h = im.size
            if w == h:
                return im_lb
            max_edge = max(im.size)
            return dict(im=ImageOps.pad(im, (max_edge, max_edge)),
                        lb=ImageOps.pad(lb, (max_edge, max_edge)))
        elif isinstance(im_lb, Image.Image):
            w, h = im_lb.size
            if w == h:
                return im_lb
            max_edge = max(im_lb.size)
            return ImageOps.pad(im_lb, (max_edge, max_edge))


class Resize(object):
    def __init__(self, resize, *args, **kwargs) -> None:
        super().__init__()
        self.resize = resize

    def __call__(self, im_lb):
        im = np.array(im_lb['im'])
        lb = np.array(im_lb['lb'])
        return dict(im=Image.fromarray(F.resize(im, self.resize[0], self.resize[1], interpolation=cv2.INTER_LINEAR)),
                    lb=Image.fromarray(F.resize(lb, self.resize[0], self.resize[1], interpolation=cv2.INTER_NEAREST)))


class RandomCrop(object):
    def __init__(self, size, *args, **kwargs):
        self.size = size

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        assert im.size == lb.size
        W, H = self.size
        w, h = im.size
        if (W, H) == (w, h):
            return dict(im=im, lb=lb)

        im = np.array(im)
        lb = np.array(lb)
        h_start = random.random()
        w_start = random.random()
        im_re = F.random_crop(im, H, W, h_start, w_start)
        lb_re = F.random_crop(lb, H, W, h_start, w_start)
        return dict(im=Image.fromarray(im_re), lb=Image.fromarray(lb_re))


class HorizontalFlip(object):
    def __init__(self, p=0.5, left_right=False, *args, **kwargs):
        self.p = p
        self.left_right = left_right

    def __call__(self, im_lb):
        if random.random() > self.p:
            return im_lb
        else:
            im = im_lb['im']
            lb = im_lb['lb']

            # atts = [1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye', 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r',
            #         10 'nose', 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']
            flip_lb = np.array(lb)
            if self.left_right:
                # CVPR21 dataset
                flip_lb[lb == 2] = 3
                flip_lb[lb == 3] = 2
                flip_lb[lb == 4] = 5
                flip_lb[lb == 5] = 4
                flip_lb[lb == 11] = 12
                flip_lb[lb == 12] = 11
                flip_lb[lb == 13] = 14
                flip_lb[lb == 14] = 13
            flip_lb = Image.fromarray(flip_lb)
            return dict(im=im.transpose(Image.FLIP_LEFT_RIGHT),
                        lb=flip_lb.transpose(Image.FLIP_LEFT_RIGHT))


class CutOut(object):
    def __init__(self, num_holes, max_h_size, max_w_size, p=0.5, *args, **kwargs):
        self.p = p
        self.num_holes = num_holes
        self.max_h_size = max_h_size
        self.max_w_size = max_w_size

    def __call__(self, im_lb):
        if random.random() > self.p:
            return im_lb
        else:
            im = np.array(im_lb['im'])
            lb = np.array(im_lb['lb'])
            holes = self.get_params_dependent_on_targets(im)

            return dict(im=Image.fromarray(F.cutout(im, holes, fill_value=0)),
                        lb=Image.fromarray(F.cutout(lb, holes, fill_value=0)))

    def get_params_dependent_on_targets(self, img):
        height, width = img.shape[:2]

        holes = []
        for _n in range(self.num_holes):
            y = random.randint(0, height)
            x = random.randint(0, width)

            y1 = np.clip(y - self.max_h_size // 2, 0, height)
            y2 = np.clip(y1 + self.max_h_size, 0, height)
            x1 = np.clip(x - self.max_w_size // 2, 0, width)
            x2 = np.clip(x1 + self.max_w_size, 0, width)
            holes.append((x1, y1, x2, y2))

        return holes


class ElasticTransform(object):
    def __init__(self, alpha, sigma, alpha_affine, border_mode, mask_value, approximate, p=0.5, *args, **kwargs):
        self.alpha = alpha
        self.sigma = sigma
        self.alpha_affine = alpha_affine
        self.border_mode = border_mode
        self.mask_value = mask_value
        self.approximate = approximate
        self.p = p

    def __call__(self, im_lb):
        if random.random() > self.p:
            return im_lb
        else:
            im = np.array(im_lb['im'])
            lb = np.array(im_lb['lb'])
            return dict(
                im=Image.fromarray(F.elastic_transform(
                    im,
                    self.alpha,
                    self.sigma,
                    self.alpha_affine,
                    cv2.INTER_NEAREST,
                    self.border_mode,
                    self.mask_value,
                    np.random.RandomState(1212),
                    self.approximate,
                )),
                lb=Image.fromarray(F.elastic_transform(
                    lb,
                    self.alpha,
                    self.sigma,
                    self.alpha_affine,
                    cv2.INTER_NEAREST,
                    self.border_mode,
                    self.mask_value,
                    np.random.RandomState(1212),
                    self.approximate,
                ))
            )


class RandomScale(object):
    def __init__(self, scales=(1, ), *args, **kwargs):
        self.scales = scales

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        W, H = im.size
        scale = random.choice(self.scales)
        w, h = int(W * scale), int(H * scale)
        return dict(im=im.resize((w, h), Image.BILINEAR),
                    lb=lb.resize((w, h), Image.NEAREST),
                    )


class ColorJitter(object):
    def __init__(self, brightness=None, contrast=None, saturation=None, *args, **kwargs):
        if not brightness is None and brightness > 0:
            self.brightness = [max(1-brightness, 0), 1+brightness]
        if not contrast is None and contrast > 0:
            self.contrast = [max(1-contrast, 0), 1+contrast]
        if not saturation is None and saturation > 0:
            self.saturation = [max(1-saturation, 0), 1+saturation]
        self.p = kwargs.pop('p')

    def __call__(self, im_lb):
        if random.random() > self.p:
            return im_lb
        im = im_lb['im']
        lb = im_lb['lb']
        r_brightness = random.uniform(self.brightness[0], self.brightness[1])
        r_contrast = random.uniform(self.contrast[0], self.contrast[1])
        r_saturation = random.uniform(self.saturation[0], self.saturation[1])
        im = ImageEnhance.Brightness(im).enhance(r_brightness)
        im = ImageEnhance.Contrast(im).enhance(r_contrast)
        im = ImageEnhance.Color(im).enhance(r_saturation)
        return dict(im=im,
                    lb=lb)


class MotionBlur(object):
    def __init__(self, p=1.0, *args, **kwargs):
        self.p = p
        self.blur_limit = (3, 7)

    def __call__(self, im_lb):
        if random.random() > self.p:
            return im_lb
        else:
            im = np.array(im_lb['im'])
            lb = im_lb['lb']
            kernel = self.get_params()
            im = F.motion_blur(im, kernel=kernel)

            return dict(im=Image.fromarray(im),
                        lb=lb)

    def get_params(self):
        ksize = random.choice(
            np.arange(self.blur_limit[0], self.blur_limit[1] + 1, 2))
        if ksize <= 2:
            raise ValueError("ksize must be > 2. Got: {}".format(ksize))
        kernel = np.zeros((ksize, ksize), dtype=np.uint8)
        xs, xe = random.randint(0, ksize - 1), random.randint(0, ksize - 1)
        if xs == xe:
            ys, ye = random.sample(range(ksize), 2)
        else:
            ys, ye = random.randint(0, ksize - 1), random.randint(0, ksize - 1)
        cv2.line(kernel, (xs, ys), (xe, ye), 1, thickness=1)

        # Normalize kernel
        kernel = kernel.astype(np.float32) / np.sum(kernel)
        return kernel


class MultiScale(object):
    def __init__(self, scales):
        self.scales = scales

    def __call__(self, img):
        W, H = img.size
        sizes = [(int(W*ratio), int(H*ratio)) for ratio in self.scales]
        imgs = []
        [imgs.append(img.resize(size, Image.BILINEAR)) for size in sizes]
        return imgs


class Compose(object):
    def __init__(self, do_list):
        self.do_list = do_list

    def __call__(self, im_lb):
        for comp in self.do_list:
            im_lb = comp(im_lb)
        return im_lb


class Rotate(object):

    def __init__(
        self,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_REFLECT_101,
        value=None,
        mask_value=None,
        p=0.5,
    ):
        super(Rotate, self).__init__()
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value
        self.angle = random.choice([90, 180, 270])
        self.p = p

    def __call__(self, im_lb):
        if random.random() > self.p:
            return im_lb

        im = im_lb['im']
        lb = im_lb['lb']

        im = np.array(im)
        lb = np.array(lb)
        im_re = F.rotate(im, self.angle, interpolation=cv2.INTER_LINEAR)
        lb_re = F.rotate(lb, self.angle, interpolation=cv2.INTER_NEAREST)
        return dict(im=Image.fromarray(im_re), lb=Image.fromarray(lb_re))


class RandomBrightnessContrast(object):
    """Randomly change brightness and contrast of the input image.

    Args:
        brightness_limit ((float, float) or float): factor range for changing brightness.
            If limit is a single float, the range will be (-limit, limit). Default: (-0.2, 0.2).
        contrast_limit ((float, float) or float): factor range for changing contrast.
            If limit is a single float, the range will be (-limit, limit). Default: (-0.2, 0.2).
        brightness_by_max (Boolean): If True adjust contrast by image dtype maximum,
            else adjust contrast by image mean.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=False, p=0.5):
        super(RandomBrightnessContrast, self).__init__()
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        self.brightness_by_max = brightness_by_max
        self.alpha = 1.0,
        self.beta = 0.0,
        self.p = p

    def __call__(self, im_lb):

        if random.random() > self.p:
            return im_lb

        im = im_lb['im']
        print(type(im))

        lb = im_lb['lb']
        im = np.array(im, dtype=np.float32)

        lb_re = np.array(lb)
        im_re = F.brightness_contrast_adjust(im, self.alpha, self.beta)

        return dict(im=Image.fromarray(im_re), lb=Image.fromarray(lb_re))
