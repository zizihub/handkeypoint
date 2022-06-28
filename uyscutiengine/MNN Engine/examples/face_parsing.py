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
from src.mnn_blaze_detector import BlazeMNNDetector
from src.mnn_multimodel import MultiModel
from src.mnn_utils import *


def video_inference(face_parsing):
    src = '../../../MusicMVGen/ouyangfeng.mp4'
    cap = cv2.VideoCapture(src)
    ret = True
    count = 0
    while ret:
        ret, frame_bgr = cap.read()
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        out = face_parsing.img_inference(frame)
        cv2.imshow('demo', out)
        cv2.waitKey(5)
        cv2.imwrite('/Users/markson/WorkSpace/MusicMVGen/gen_v/%04d.png' % count, out)
        cv2.imwrite('/Users/markson/WorkSpace/MusicMVGen/gen_v/%04d.jpg' % count, frame_bgr)
        count += 1
    cv2.destroyAllWindows()
    cv2.release()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="face parsing")
    parser.add_argument(
        "--mnn_path",
        default="../models/mnn/face-7-1101.mnn")
    parser.add_argument(
        "--num_class",
        type=int,
        default=7)
    args = parser.parse_args()
    face_parsing = MultiModel(BlazeMNNDetector(mode='face', filtering=False),
                              MNNParsing(mnn_path=args.mnn_path,
                                         num_class=args.num_class,
                                         ratio=(1, 1),
                                         rotate=False),
                              crop_ratio=2.0,
                              crop_shift=0.0,
                              rotate=False,)
    face_parsing.cam_inference()
