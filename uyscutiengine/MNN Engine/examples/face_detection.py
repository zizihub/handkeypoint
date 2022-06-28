#!/usr/bin/python
# -*- encoding: utf-8 -*-
import os
import sys
sys.path.append('../')  # NOQA: 402
from src.mnn_blaze_detector import BlazeMNNDetector


if __name__ == '__main__':
    md = BlazeMNNDetector('face_fullrange', filtering=False)
    md.cam_inference()
