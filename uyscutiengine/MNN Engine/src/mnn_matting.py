#!/usr/bin/python
# -*- encoding: utf-8 -*-
from .mnn_model import MNNModel
from .mnn_utils import *
from collections import deque
import numpy as np
import cv2
import time
import os


class MNNMatting(MNNModel):
    def __init__(self, mnn_path, optical_flow=False):
        super(MNNMatting, self).__init__()
        self.num_class = 0
        self.mnn_name = os.path.basename(mnn_path)
        ########### MNN Cfg ##############
        (
            self.net_mnn,
            self.session,
            self.input_tensor,
            self.c_scale,
            self.h_scale,
            self.w_scale,
        ) = self._load_model(mnn_path)
        self.deplay_frame = deque(maxlen=2)
        self.prev_frame = None
        # self.bgr = cv2.imread('/Users/markson/Downloads/Wallpaper/dan-freeman-7Zb7kUyQg1E-unsplash.jpg')
        # self.bgr = cv2.resize(self.bgr, (1280, 720))
        self.bgr = np.zeros((720, 1280, 3)) + (255, 255, 255)
        # self.bgr = np.zeros((720, 1280, 3)) + (162, 250, 139)

        # prev_mat
        self.prev_alpha = None
        # optical flow
        self.optical_flow = optical_flow
        self.disflow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        self.prev_gray = np.zeros((self.h_scale, self.w_scale), np.uint8)
        self.prev_fused = np.zeros((self.h_scale, self.w_scale), np.float32)
        self.is_init = True
        self.count = 0

        self.abs_cout = []

    def img_inference(self, frame, save_video=False, show_image=False):
        (fgr, pha) = self.predict(frame, mean=0, std=1., border_mode=cv2.BORDER_REFLECT)
        fake_img = None
        # optical flow matting fused
        if self.optical_flow:
            start = time.time()
            cur_gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), (self.w_scale, self.h_scale))
            pha = optic_flow_process(cur_gray, pha.squeeze(2), self.prev_gray,
                                     self.prev_fused, self.disflow, self.is_init)
            self.prev_fused = pha
            self.prev_gray = cur_gray
            self.is_init = False
            print("optical flow: {:.4}ms".format((time.time() - start)*1000))
        # if self.prev_frame is not None:
        #     fake_img = self._plot(self.prev_frame, (fgr, pha))
        # self.prev_frame = frame
        fake_img, pha = self._plot(frame, (fgr, pha), show_image)
        return fake_img, pha

    def cam_inference(self, video=0, dst_path='', show_image=False):
        cost = []
        save_video = bool(video and dst_path)
        cap = cv2.VideoCapture(video)
        ############# video ###############
        if save_video:
            if dst_path.endswith('.mp4'):
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
                out_video = cv2.VideoWriter(dst_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (720, 1280), True)
            else:
                os.makedirs(dst_path, exist_ok=True)
        ############# camera ###############
        while True:
            ret, frame = cap.read()
            if not ret or self.count == 1000:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, self.bgr.shape[:2][::-1])
            ####################################
            fake_img, pha = self.img_inference(frame, save_video=save_video, show_image=show_image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
            print('avg cost: {:.2f}ms'.format(np.mean(self.cost)*1000))
            if dst_path:
                if dst_path.endswith('.mp4'):
                    out_video.write(fake_img)
                    # out_video.write((pha * 255.).astype(np.uint8))
                    self.count += 1
                else:
                    cv2.imwrite(os.path.join(dst_path, '%04d.png') % self.count,
                                (pha * 255.).astype(np.uint8), [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                    self.count += 1
        cv2.destroyAllWindows()
        cap.release()
        if dst_path:
            out_video.release()

    def _preprocess(self, frame, mean=0, std=1, border_mode=cv2.BORDER_CONSTANT):
        '''
        Preproces, default input is normalized to [0,1].
        '''
        image = frame.copy()
        h, w, c = image.shape
        image = zero_resize_padding(image, self.h_scale, self.w_scale,
                                    ratio=(16, 9),
                                    border_mode=border_mode,
                                    rotate=True)
        print('padding size: {} | ratio: {:.3f}'.format(image.shape, max(image.shape[:2])/min(image.shape[:2])))
        image = self._to_tensor(image, mean, std)

        if self.c_scale == 4:
            if isinstance(self.prev_alpha, np.ndarray):
                print(self.count, 'use prev alpha')
                prev_alpha = self.prev_alpha.transpose(2, 0, 1)[np.newaxis, ...]
                image = np.concatenate([image, prev_alpha], axis=1)
            else:
                image = np.concatenate([image, np.zeros((1, 1, self.h_scale, self.w_scale), dtype=np.float32)], axis=1)
        self.debug_show(image)
        return image

    def _postprocess(self, output_tensor):
        # fgr alpha
        try:
            assert output_tensor['foreground'].getShape() == (1, 3, self.h_scale, self.w_scale)
            fgr = self._get_mnn_output(output_tensor['foreground'])
            assert output_tensor['alpha'].getShape() == (1, 1, self.h_scale, self.w_scale)
            pha = self._get_mnn_output(output_tensor['alpha'])
        except:
            for k, v in output_tensor.items():
                if v.getShape() == (1, 3, self.h_scale, self.w_scale):
                    fgr = self._get_mnn_output(v)
                elif v.getShape() == (1, 1, self.h_scale, self.w_scale):
                    pha = self._get_mnn_output(v)
        fgr = fgr.squeeze(0).transpose((1, 2, 0))[:, :, ::-1]
        pha = pha.squeeze(0).transpose((1, 2, 0))
        pha = self.entropy_filter(pha)
        self.prev_alpha = pha
        self.deplay_frame.append(pha)
        # pha = self.one_frame_delay(pha)
        print('output shape', pha.shape)
        return (fgr, pha)

    def momentun_filter(self, pha):
        # def sigmoid(x):
        #     return 1 / (1 + np.exp((-x+300)/100))
        def sigmoid(x):
            return 1 / (1 + np.exp((-x+0.5) * 5))
        if self.prev_alpha is None:
            return pha
        m = sigmoid(abs(pha - self.prev_alpha))
        cv2.imshow('m', m)
        print(m.min(), m.max())
        pha = self.prev_alpha * (1-m) + pha * m
        return pha.clip(0, 1)

    def entropy_filter(self, pha):
        if self.prev_alpha is None:
            return pha
        entropy = abs(pha*2 - 1)
        pha = self.prev_alpha * (1-entropy) + pha * entropy
        return pha

    def one_frame_delay(self, cur_frame):
        '''One Frame Delay policy from MODNet'''
        if len(self.deplay_frame) < 2:
            return cur_frame
        prev_frame = self.deplay_frame[0]
        frame = self.deplay_frame[1]
        out_frame = np.where(np.logical_or(np.abs(frame-prev_frame) > 0.1, np.abs(frame-cur_frame) > 0.1),
                             ((cur_frame+prev_frame) / 2),
                             frame)
        return out_frame

    def _plot(self, frame, post_out, show_image=False):
        '''
        0: 'background',
        1: 'foreground',
        '''
        fgr, alpha = recover_zero_padding(frame, post_out, ratio=(16, 9))
        fgr = (cv2.resize(fgr, (frame.shape[1], frame.shape[0])) * 255).astype(np.uint8)
        alpha = cv2.resize(alpha, (frame.shape[1], frame.shape[0]))
        if show_image:
            alpha = cv2.merge((alpha, alpha, alpha))
            cv2.imshow('alpha', alpha)
            cv2.imshow('fgr', fgr)
            fake_img = (frame[:, :, ::-1] * alpha + self.bgr * (1-alpha)).astype(np.uint8)
            cv2.imshow(self.mnn_name, fake_img)
        return fake_img, alpha
