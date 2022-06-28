#!/usr/bin/python
# -*- encoding: utf-8 -*-
import os
import sys
sys.path.append('../')  # NOQA: 402
import time
import cv2
import numpy as np
import MNN
from collections import deque
from PIL import Image
from src.one_euro_filter import OneEuroFilter
from src.mnn_blaze_detector import BlazeMNNDetector
from src.mnn_model import MNNModel
from src.mnn_utils import *
from src.mnn_multimodel import MultiModel
from src.mnn_classification import MNNClassification


class HandGestureRecognition(MNNClassification):
    def __init__(self, mnn_path):
        super(HandGestureRecognition, self).__init__(mnn_path)
        self.classes = {
            0: 'others',
            1: 'call',
            2: 'dislike',
            3: 'fist',
            4: 'five',
            5: 'four',
            6: 'gun',
            7: 'heart',
            8: 'hold_face',
            9: 'like',
            10: 'ok',
            11: 'one',
            12: 'salute',
            13: 'three',
            14: 'yeah',
        }

    def _postprocess(self, output_tensor):
        logit = self._get_mnn_output(output_tensor['output']).squeeze()
        index = np.argmax(logit)
        score = softmax(logit)[index]
        label = self.classes[index]
        # label, score = self._multiframe_process(np.argmax(score), score, mode='threshold')
        return 'cls', score, label


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="hand gesture recognition")
    parser.add_argument(
        "--mnn_path",
        default="../models/mnn/hgr/mbv3-0909.mnn")
    args = parser.parse_args()
    md = MultiModel(BlazeMNNDetector('palm', True),
                    HandGestureRecognition(args.mnn_path),
                    crop_ratio=2.5,
                    crop_shift=0.5,
                    rotate=False)
    md.cam_inference()
    # md.img_inference(img_path='../datasets/video/gesture_new', save_video=True)
