#!/usr/bin/python
# -*- encoding: utf-8 -*-
import os
import sys
import cv2
import numpy as np
from PIL import Image
import time
sys.path.append('../')  # NOQA: 402
from src import MNNLandmarkRegressor, BlazeMNNDetector, MultiModel
from src.mnn_utils import *


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="hand pose estimation")
    parser.add_argument(
        "--mnn_path",
        default=None)
    parser.add_argument(
        "--mode",
        default="lite")
    args = parser.parse_args()
    hand_landmark = MultiModel(BlazeMNNDetector(mode='palm',
                                                filtering=False),
                               MNNLandmarkRegressor(mnn_path=args.mnn_path,
                                                    mode=args.mode,
                                                    filtering=True),
                               crop_ratio=3.0,
                               crop_shift=0.5,
                               rotate=True)
    hand_landmark.cam_inference()
    # hand_landmark.cam_inference(video='/Users/markson/WorkSpace/uyscutiengine/MNN Engine/datasets/video/handpose_demo_video.mp4',
    #                             dst_path='/Users/markson/WorkSpace/uyscutiengine/MNN Engine/datasets/video/handpose_demo_mbv3.mp4')
