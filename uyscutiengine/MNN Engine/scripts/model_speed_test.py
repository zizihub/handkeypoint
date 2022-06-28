
import sys
import numpy as np
sys.path.append('..')  # NOQA: 402
from src.mnn_model import MNNModel  # NOQA: 402


class MNNSpeedTest(MNNModel):
    def __init__(self, mnn_path):
        super(MNNSpeedTest, self).__init__()
        ########### MNN Cfg ##############
        (
            self.net_mnn,
            self.session,
            self.input_tensor,
            self.c_scale,
            self.w_scale,
            self.h_scale
        ) = self._load_model(mnn_path)

    def img_inference(self, frame, show_image=True):
        image = self._preprocess(frame)
        out, cost_once = self._mnn_inference(image)
        self.cost.append(cost_once)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="model speed test")
    parser.add_argument(
        "--mnn_path",
        default=None)
    args = parser.parse_args()
    speedtest = MNNSpeedTest(args.mnn_path)
    speedtest.cam_inference()
