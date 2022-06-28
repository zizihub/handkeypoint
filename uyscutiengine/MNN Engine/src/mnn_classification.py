#!/usr/bin/python
# -*- encoding: utf-8 -*-
from .mnn_model import MNNModel
from .mnn_utils import *
import numpy as np
from collections import deque


class MNNClassification(MNNModel):
    def __init__(self, mnn_path):
        super(MNNClassification, self).__init__()
        ########### MNN Cfg ##############
        (
            self.net_mnn,
            self.session,
            self.input_tensor,
            self.c_scale,
            self.w_scale,
            self.h_scale,
        ) = self._load_model(mnn_path)
        self.label_window = deque(maxlen=10)
        self.score_window = deque(maxlen=10)

    def img_inference(self, frame, show_image=False):
        out = self.predict(frame, mean=0.5, std=0.5)
        return out

    def _postprocess(self, output_tensor):
        pass

    def _multiframe_process(self, label_idx, score, mode='voting'):
        if mode == 'threshold':
            if score < 0.95 and len(self.label_window):
                vote_idx = self.label_window[-1]
            else:
                vote_idx = label_idx
                self.label_window.append(label_idx)
            print(self.label_window)
            return self.classes[vote_idx], round(score, 4)
        else:
            self.label_window.append(label_idx)
            # print(self.label_window)
            self.score_window.append(score)
            x = len(self.classes)
            vote_label = np.zeros(x)
            avg_scores = np.zeros(x)
            for i in range(len(self.label_window)):
                vote_label[self.label_window[i]] += 1
                avg_scores[self.label_window[i]] += self.score_window[i]

            vote_label[vote_label == 0] = 1
            avg_scores /= vote_label
            if mode == 'voting':
                vote_idx = np.argmax(vote_label)
                vote_idx = vote_idx if vote_label[vote_idx] >= 8 else 0
            elif mode == 'weighted':
                vote_idx = np.argmax(avg_scores)
            elif mode == 'threshold':
                vote_idx = self.label_window[-1]

            print(avg_scores, self.classes[vote_idx])
            return self.classes[vote_idx], round(avg_scores[vote_idx], 4)
