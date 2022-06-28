#!/usr/bin/python
# -*- encoding: utf-8 -*-
import sys
sys.path.append('../')  # NOQA: 402
from src.mnn_regression import MNNRegression
from src.mnn_blaze_detector import BlazeMNNDetector
from src.mnn_multimodel import MultiModel
from src.mnn_utils import *


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Puff Regression")
    parser.add_argument(
        "--mnn_path",
        default="../models/mnn/xuziyan/puff_regression.mnn")
    args = parser.parse_args()
    puff_regression = MultiModel(BlazeMNNDetector(mode='face', filtering=False),
                                 MNNRegression(mnn_path=args.mnn_path),
                                 crop_ratio=1.3,
                                 crop_shift=0.0)
    puff_regression.cam_inference()
