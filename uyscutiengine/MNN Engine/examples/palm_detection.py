#!/usr/bin/python
# -*- encoding: utf-8 -*-
import os
import sys
sys.path.append('../')  # NOQA: 402
from src.mnn_blaze_detector import BlazeMNNDetector
from src.mnn_yolox_detector import YOLOXDetector


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="hand pose estimation")
    parser.add_argument(
        "--mnn_path",
        default=None)
    parser.add_argument(
        "--mode",
        default=None)
    parser.add_argument(
        "--video",
        default=False,
        action="store_true")
    args = parser.parse_args()
    if args.mode:
        md = BlazeMNNDetector('palm', filtering=True)
    else:
        md = YOLOXDetector(mnn_path=args.mnn_path, filtering=True)
    # md.cam_inference(show_image=True)
    #### test ####
    import cv2
    frame = cv2.imread('../scripts/input.jpg')
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    md.img_inference(frame, show_image=True)
