#!/usr/bin/python
# -*- encoding: utf-8 -*-
from .mnn_model import MNNModel
from .mnn_utils import *
from .mnn_multimodel import MultiModel
from collections import deque
import MNN
import numpy as np
import cv2


class MNNParsing(MNNModel):
    def __init__(self,
                 mnn_path,
                 num_class,
                 optical_flow=False,
                 **kwargs):
        super(MNNParsing, self).__init__()
        self.num_class = num_class
        ########### MNN Cfg ##############
        (
            self.net_mnn,
            self.session,
            self.input_tensor,
            self.c_scale,
            self.h_scale,
            self.w_scale,
        ) = self._load_model(mnn_path)
        self.ratio = kwargs.pop('ratio', (16, 9))
        self.rotate = kwargs.pop('rotate', True)
        self.prev_mask = None
        # optical flow
        self.optical_flow = optical_flow
        self.disflow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        self.prev_gray = np.zeros((self.h_scale, self.w_scale), np.uint8)
        self.prev_fused = np.zeros((self.h_scale, self.w_scale), np.float32)
        self.is_init = True

    def img_inference(self, frame, show_image=False):
        post_outs = self.predict(frame, mean=0.5, std=0.5)
        # optical flow segmentation fused
        if self.optical_flow:
            cur_gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), (self.w_scale, self.h_scale))
            post_out = optic_flow_process(cur_gray, post_outs, self.prev_gray,
                                          self.prev_fused, self.disflow, self.is_init)
            self.prev_fused = post_out
            self.prev_gray = cur_gray
            self.is_init = False
        if show_image:
            parsing = recover_zero_padding(frame, post_outs, ratio=(16, 9))
            parsing = cv2.resize(parsing, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            parsing = draw_mask(frame, parsing, show_image=show_image)
            return parsing
        else:
            return post_outs

    def _preprocess(self, frame, mean=0.5, std=0.5, border_mode=cv2.BORDER_CONSTANT):
        '''
        Preproces, default input is normalized to [0,1].
        '''
        image = frame.copy()
        h, w, c = image.shape
        image = zero_resize_padding(frame, self.h_scale, self.w_scale,
                                    ratio=self.ratio,
                                    border_mode=border_mode,
                                    rotate=self.rotate)
        print('padding size: {} | ratio: {:.3f}'.format(image.shape, max(image.shape[:2])/min(image.shape[:2])))
        image = self._to_tensor(image, mean, std)

        if self.c_scale == 4:
            if isinstance(self.prev_mask, np.ndarray):
                # prev_mask = self._to_tensor(self.prev_mask, mean, std)
                prev_mask = self.prev_mask.transpose(2, 0, 1)[np.newaxis, ...]
            else:
                prev_mask = self._to_tensor(np.zeros((self.h_scale, self.w_scale, 1)), mean, std)
            print(image.shape, prev_mask.shape)
            image = np.concatenate([image, prev_mask], axis=1)
        self.debug_show(image)
        return image

    def _postprocess(self, output_tensor):
        try:
            mat_np = self._get_mnn_output(output_tensor['masks'])
        except:
            for k, v in output_tensor.items():
                if v.getShape() == (1, 2, self.h_scale, self.w_scale):
                    mat_np = self._get_mnn_output(v)
        last_fuse = mat_np.squeeze(0).transpose(1, 2, 0)
        self.prev_mask = last_fuse.argmax(2)[..., np.newaxis]
        # cv2.imshow('mask_prob', cv2.resize(mask_remove_padding(
        #     np.zeros([720, 1280, 3]), last_fuse[:, :, 1]), (1280, 720)))
        last_fuse = last_fuse.argmax(2)
        print('output shape', last_fuse.shape, np.unique(last_fuse))
        self.std.append(last_fuse)
        print('>>> stability', np.array(self.std).std(axis=0).mean())
        return last_fuse

    @ staticmethod
    def debug_show(image):
        image = (image.squeeze(0).transpose((1, 2, 0))[:, :, ::-1] + 1) * 127.5
        cv2.imshow('debug show', cv2.resize(image.astype(np.uint8), None, fx=2, fy=2))
