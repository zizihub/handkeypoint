#!/usr/bin/python
# -*- encoding: utf-8 -*-
from .mnn_model import MNNModel
from .mnn_blaze_detector import BlazeMNNDetector
from .mnn_utils import *
from collections import deque
import MNN
import numpy as np
import cv2


class MNNRegression(MNNModel):
    def __init__(self, mnn_path):
        super(MNNRegression, self).__init__()
        ########### MNN Cfg ##############
        (
            self.net_mnn,
            self.session,
            self.input_tensor,
            self.c_scale,
            self.w_scale,
            self.h_scale,
        ) = self._load_model(mnn_path)
        self.prev_mask = None

    def img_inference(self, frame, show_image=False):
        out = self.predict(frame, mean=0.5, std=0.5)
        return float(out.squeeze())

    def txt_inference(self, img_path):
        cost = []
        img = cv2.imread(img_path)
        img = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
        list_text = open('./puff_badcase.txt', 'a+')
        post_out = self.img_inference(img)
        lines = img_path+' '+str(post_out[0][0])+'\n'
        list_text.write(lines)

    def video_inference(self):
        cost = []
        src = '/Users/orange/Desktop/infer_data/xzy.mp4'
        cap = cv2.VideoCapture(src)
        ret = True
        count = 0
        while ret:
            ret, frame_bgr = cap.read()
            frame = cv2.cvtColor(cv2.flip(frame_bgr, 1), cv2.COLOR_BGR2RGB)

            post_out = self.img_inference(frame)
            vis_img = self._plot(frame, post_out)

            # cv2.imshow('demo', out)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
            print('avg cost: {:.2f}ms'.format(np.mean(cost)*1000))
            cv2.imwrite('/Users/orange/Desktop/infer_data/video_xzy/%04d.png' % count, vis_img)
            # cv2.imwrite('/Users/orange/Desktop/infer_data/video_infer/%04d.jpg' % count, frame_bgr)
            count += 1
        cv2.destroyAllWindows()
        cv2.release()

    def _plot(self, frame, post_outs):
        img = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2BGR)
        img = cv2.putText(img, str(post_outs), ((0, 0), (0, 0)), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2, 2)
        cv2.imshow('demo', img)
        return img

    def _postprocess(self, out):
        out = self._get_mnn_output(out['ranking'])
        return out
