#!/usr/bin/python
# -*- encoding: utf-8 -*-
import os
import sys
import cv2
import numpy as np
from PIL import Image
import time
sys.path.append('../')  # NOQA: 402
from src.mnn_parsing import MNNParsing
from src.mnn_utils import *


def video_inference(portrait_parsing):
    src = '../../../MusicMVGen/ouyangfeng.mp4'
    cap = cv2.VideoCapture(src)
    ret = True
    count = 0
    while ret:
        ret, frame_bgr = cap.read()
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        out = portrait_parsing.img_inference(frame)
        cv2.imshow('demo', out)
        cv2.waitKey(5)
        cv2.imwrite('/Users/markson/WorkSpace/MusicMVGen/gen_v/%04d.png' % count, out)
        cv2.imwrite('/Users/markson/WorkSpace/MusicMVGen/gen_v/%04d.jpg' % count, frame_bgr)
        count += 1
    cv2.destroyAllWindows()
    cv2.release()


class PortraitParsing(MNNParsing):
    def __init__(self,
                 mnn_path,
                 num_class,
                 optical_flow=False):
        super(PortraitParsing, self).__init__(mnn_path=mnn_path,
                                              num_class=num_class,
                                              optical_flow=optical_flow)
        self.bgr = cv2.imread('/Users/markson/Downloads/Wallpaper/dan-freeman-7Zb7kUyQg1E-unsplash.jpg')
        self.bgr = cv2.resize(self.bgr, (1280, 720))

    def img_inference(self, frame):
        post_outs = self.predict(frame, mean=0, std=1.0)
        # optical flow segmentation fused
        if self.optical_flow:
            cur_gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), (self.w_scale, self.h_scale))
            post_out = optic_flow_process(cur_gray, post_outs, self.prev_gray,
                                          self.prev_fused, self.disflow, self.is_init)
            self.prev_fused = post_out
            self.prev_gray = cur_gray
            self.is_init = False
        parsing = draw_mask(frame, post_outs)
        return parsing


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Portrait Parsing")
    parser.add_argument(
        "--mnn_path",
        default="../models/mnn/mbv3-unet-seg-portrait-benchmark.mnn")
    args = parser.parse_args()
    face_parsing = PortraitParsing(mnn_path=args.mnn_path,
                                   num_class=1,
                                   optical_flow=False,)
    face_parsing.cam_inference()
