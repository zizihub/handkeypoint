#!/usr/bin/python
# -*- encoding: utf-8 -*-
import MNN
import numpy as np
import os
import cv2
from time import time
from .mnn_utils import *
from collections import deque
from abc import ABCMeta, abstractmethod

__all__ = ['MNNModel']


class MNNModel(metaclass=ABCMeta):
    def __init__(self):
        super(MNNModel, self).__init__()
        self.cost = deque(maxlen=100)
        self.std = deque(maxlen=5)
        # with open('../scripts/face_hand_landmark.txt', 'r') as f:
        #     files = f.readlines()
        # self.faces, self.hands = [], []
        # for line in files:
        #     face, hand = line.split(';')[1:]
        #     self.faces.append(np.array(face.split(' ')))
        #     self.hands.append(np.array(hand.split(' ')))

    def _load_model(self, mnn_path, input_name='input'):
        net_mnn = MNN.Interpreter(mnn_path)
        session = net_mnn.createSession()
        input_tensor = net_mnn.getSessionInput(session, input_name)
        c_scale = input_tensor.getShape()[1]
        h_scale = input_tensor.getShape()[2]
        w_scale = input_tensor.getShape()[3]
        print(f"{input_tensor.getShape()=}")
        return net_mnn, session, input_tensor, c_scale, w_scale, h_scale

    def cam_inference(self, video=0, show_image=False, dst_path=''):
        save_video = bool(video and dst_path)
        cap = cv2.VideoCapture(video)
        count = 0
        ############# video ###############
        if save_video:
            if dst_path.endswith('.mp4'):
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
                out_video = cv2.VideoWriter(dst_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (1680, 720), True)
            else:
                os.makedirs(dst_path, exist_ok=True)
        while True:
            ret, frame = cap.read()
            frame_rgb = cv2.cvtColor(frame[:, :, :], cv2.COLOR_BGR2RGB)

            # # draw
            # if len(self.hands[count]) != 1:
            #     hand_line = np.array(self.hands[count]).reshape(21, 3)
            #     for i in range(21):
            #         x, y, _ = hand_line[i]
            #         cv2.circle(frame_rgb, (int(x), int(y)), 5, (255, 0, 0), 5, 1)
            # print(self.faces[count])
            # face_line = np.array(self.faces[count]).reshape(68, 3)
            # for i in range(68):
            #     x, y, _ = face_line[i]
            #     cv2.circle(frame_rgb, (int(x), int(y)), 5, (0, 255, 0), 5, 1)

            out = self.img_inference(frame_rgb, show_image=show_image)
            # with open('hand_landmark.txt', 'a') as f:
            #     if out == None:
            #         f.write(f'frame{count}' + '\n')
            #     else:
            #         f.write(f'frame{count} ' +
            #                 ' '.join(map(str, map(lambda x: round(x, 5), out[0][0].flatten().tolist()))) + '\n')

            count += 1
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
            print('>>> {} sumavg cost: {:.2f}ms\n'.format(self.__class__, np.mean(self.cost)*1000))
            if dst_path:
                out_video.write(out[1] if out != None else frame)
        cv2.destroyAllWindows()
        cap.release()
        if dst_path:
            out_video.release()

    @ abstractmethod
    def img_inference(self):
        """single image inference"""
        raise NotImplementedError

    def predict(self, img, mean, std, border_mode=cv2.BORDER_CONSTANT):
        ####################################
        image = self._preprocess(img, mean=mean, std=std, border_mode=border_mode)
        out, cost_once = self._mnn_inference(image)
        post_out = self._postprocess(out)
        self.cost.append(cost_once)
        print('>>> {} avg cost: {:.2f}ms'.format(self.__class__, np.mean(self.cost)*1000))
        return post_out

    def load_anchors(self, path):
        self._anchors = np.load(path).astype(np.float32)
        assert(self._anchors.ndim == 2)
        assert(self._anchors.shape[0] == self.num_anchors)
        assert(self._anchors.shape[1] == 4)

    def _to_tensor(self, image, mean, std):
        # change numpy data type as np.float32 to match tensor's format
        image = (image / 255.0).astype(np.float32)
        # cv2 read shape is NHWC, Tensor's need is NCHW,transpose it
        image = image.transpose((2, 0, 1))

        # normalize
        mean = np.array(mean)
        std = np.array(std)
        if mean.ndim == 1:
            mean = mean.reshape(-1, 1, 1)
        if std.ndim == 1:
            std = std.reshape(-1, 1, 1)
        image = (image - mean) / std

        return image[np.newaxis, ...]

    def _preprocess(self, frame, mean=0, std=1, ratio=(1, 1), border_mode=cv2.BORDER_CONSTANT, rotate=False, letter_box=True):
        '''
        Preproces, default input is normalized to [0,1].
        '''
        image = frame.copy()
        image = zero_resize_padding(frame, self.h_scale, self.w_scale, ratio=ratio,
                                    border_mode=border_mode, rotate=rotate, letter_box=letter_box)
        # image = cv2.resize(image, (int(self.w_scale), int(self.h_scale)))
        cv2.imshow('input', image)
        image = self._to_tensor(image, mean, std)
        return image

    def _postprocess(self, out):
        """customize postprocess"""
        raise NotImplementedError

    def _mnn_inference(self, image):
        start = time()
        tmp_input = MNN.Tensor((1, self.c_scale, int(self.h_scale), int(self.w_scale)),
                               MNN.Halide_Type_Float,
                               image.astype(np.float32),
                               MNN.Tensor_DimensionType_Caffe)
        self.input_tensor.copyFrom(tmp_input)
        self.net_mnn.runSession(self.session)
        output_tensor = self.net_mnn.getSessionOutputAll(self.session)
        cost_once = time() - start
        return output_tensor, cost_once

    def _get_mnn_output(self, tensor):
        holder_shape = tensor.getShape()
        place_holder = MNN.Tensor(holder_shape,
                                  MNN.Halide_Type_Float,
                                  np.ones(holder_shape).astype(np.float32),
                                  MNN.Tensor_DimensionType_Caffe)
        tensor.copyToHostTensor(place_holder)
        return place_holder.getNumpyData().copy()

    def debug_show(self, image):
        image = (image.squeeze(0).transpose((1, 2, 0)) * self.mean + self.std)[:, :, ::-1] * 255
        cv2.imshow('debug show', cv2.resize(image.astype(np.uint8), None, fx=2, fy=2))
