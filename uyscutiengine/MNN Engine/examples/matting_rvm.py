import os
import sys
import cv2
import numpy as np
from PIL import Image
import shutil
import time
sys.path.append('../')  # NOQA: 402
from src.mnn_rvm import RVMModel


class HumanChecker:
    def __init__(self, rvm):
        self.model = rvm

    def check_human(self, root, f):
        src_path = os.path.join(root, f)
        # dst_path = src_path.replace('indoor', 'indoor_no_human')
        img = cv2.imread(src_path)
        if img is None:
            return
        _, alpha = self.model.predict(img, mean=0, std=1)
        score = alpha.sum()
        if score > 1000:
            print(src_path, score)
            os.remove(src_path)

    def mp_check_human(self):
        src = '/Users/markson/WorkSpace/indoor_no_human'
        for root, dirs, files in os.walk(src):
            if dirs:
                continue
            print('checking:', root)
            # os.makedirs(root.replace('indoor', 'indoor_no_human'), exist_ok=True)
            for f in files:
                self.check_human(root, f,)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Portrait Parsing")
    parser.add_argument(
        "--mnn_path",
        default="../models/mnn/portrait_mat/rvm_mobilenetv3_fp32-480-480.mnn")
    parser.add_argument(
        "--video",
        default=False,
        action="store_true")
    args = parser.parse_args()
    rvm = RVMModel(mnn_path=args.mnn_path)
    # HC = HumanChecker(rvm)
    # HC.mp_check_human()
    if not args.video:
        rvm.cam_inference()
    else:
        rvm.cam_inference('../datasets/video/portrait_matting_hard.mp4',
                          '../datasets/video/portrait_matting_hard_rvm')
