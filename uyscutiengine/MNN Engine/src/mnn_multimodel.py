#!/usr/bin/python
# -*- encoding: utf-8 -*-
import MNN
import numpy as np
import cv2
from PIL import Image
from time import time
from copy import deepcopy
from .mnn_model import MNNModel
from .mnn_utils import *


class MultiModel(MNNModel):
    def __init__(self,
                 *pipelines,
                 crop_ratio,
                 crop_shift,
                 rotate=True):
        super(MultiModel, self).__init__()
        self.pipelines = pipelines
        self.crop_ratio = crop_ratio
        self.crop_shift = crop_shift
        self.rotate = rotate
        self.boxes_landmarks_flags = []
        self.input_image = None

    def img_inference(self, frame, show_image=False):
        start = time()
        for pipeline in self.pipelines:
            if isinstance(frame, list):
                frames = frame.copy()
                outs_list = []
                for frame in frames:
                    outs_list.append(pipeline.img_inference(frame, show_image=show_image))
                result = self._postprocess(frames, outs_list, pipeline.h_scale, pipeline.w_scale)
            else:
                self.input_image = frame
                post_outs = pipeline.predict(frame, pipeline.mean, pipeline.std)
                outs = pipeline.post_out(frame, post_outs)
                if 'boxes' in outs or 'landmarks' in outs:
                    boxes = outs['boxes']
                    landmarks = outs['landmarks']
                    out_frames = []
                    resized_box_list = []
                    flag_list = []
                    print('detect num:', len(boxes))
                    for box, landmark in zip(boxes, landmarks):
                        crop_img, flag, *resized_box = crop_detected_frame(frame,
                                                                           box,
                                                                           landmark,
                                                                           ratio=self.crop_ratio,
                                                                           shift=self.crop_shift,
                                                                           rotate=self.rotate)
                        out_frames.append(crop_img)
                        resized_box_list.append(*resized_box)
                        flag_list.append(flag)
                    self.boxes_landmarks_flags = {
                        'boxes': resized_box_list,
                        'landmarks': landmarks,
                        'flags': flag_list,
                    }
                frame = out_frames
        self.cost.append(time()-start)
        return result

    def _postprocess(self, frames, outs_list, h_scale, w_scale):
        '''
        1. remove zero-padding
        2. recover mask to entire image
        3. recover box and landmark to entire image
        '''
        result = np.zeros_like(self.input_image)[:, :, 0]
        boxes = self.boxes_landmarks_flags['boxes']
        landmarks = self.boxes_landmarks_flags['landmarks']
        flags = self.boxes_landmarks_flags['flags']
        mode = ''
        draw_target = []
        for frame, box, landmark, flag, outs in zip(frames, boxes, landmarks, flags, outs_list):
            if isinstance(outs, np.ndarray):
                mode = 'parsing'
                draw_target.append((outs, box))
            elif isinstance(outs, float):
                mode = 'regression'
                draw_target.append((outs, box))
            elif isinstance(outs, tuple):
                if isinstance(outs[0], np.ndarray):
                    mode = 'landmark'
                    draw_target.append((outs, frame, flag, box))
                else:
                    mode = 'cls'
                    draw_target.append((outs, box))
        result = draw_multimodel_func(self.input_image, draw_target, mode, self.rotate, True, h_scale, w_scale)
        return result
