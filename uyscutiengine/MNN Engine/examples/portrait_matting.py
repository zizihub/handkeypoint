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
from src.mnn_matting import MNNMatting


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Portrait Matting")
    parser.add_argument(
        "--mnn_path",
        default="../models/mnn/ddr-mat-portrait-1104-192_fp16.mnn")
    parser.add_argument(
        "--video",
        default=False,
        action="store_true")
    args = parser.parse_args()
    portrait_matting = MNNMatting(mnn_path=args.mnn_path, optical_flow=False)
    if not args.video:
        portrait_matting.cam_inference()
    else:
        portrait_matting.cam_inference('../datasets/video/hair_seg/hair.mp4',
                                       '../datasets/video/ddr23slim-kd-momentum-mask.mp4')
