#!/usr/bin/python
# -*- encoding: utf-8 -*-
import sys
sys.path.append('../')  # NOQA: 402
from src.mnn_parsing import MNNParsing
from src.mnn_utils import *


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="sky parsing")
    parser.add_argument(
        "--mnn_path",
        default="../models/mnn/xuziyan/sky_parsing.mnn")
    args = parser.parse_args()
    hair_parsing = MNNParsing(
        mnn_path=args.mnn_path,
        num_class=1,
    )
    hair_parsing.cam_inference()
