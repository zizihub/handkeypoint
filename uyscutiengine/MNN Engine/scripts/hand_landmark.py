#!/usr/bin/python
# -*- encoding: utf-8 -*-
import numpy as np
import time
from time import time
import MNN
import os
import cv2
from collections import deque
from abc import ABCMeta, abstractmethod


def zero_resize_padding(img, h_scale, w_scale, ratio=(1, 1), border_mode=cv2.BORDER_CONSTANT, rotate=False):
    """Zero Resize Padding

    Args:
        img (np.ndarray): input image
        h_scale (int): image resize height for model
        w_scale (int): image resize width for model
        ratio (tuple, optional): ratio of input image in (height, width). Defaults to (1, 1).
        border_mode ([type], optional): mode for zero padding. Defaults to cv2.BORDER_CONSTANT.
        rotate (bool, optional): using dynamic portrait rotate for portrait mode. Defaults to False.

    Returns:
        image: output zero padding resized image
    """
    # zero padding
    h, w, _ = img.shape
    if w > h:
        ratio = ratio[::-1]
    tb = int(ratio[0]/ratio[1]*w) - h
    lr = int(ratio[1]/ratio[0]*h) - w
    if tb >= lr:
        tb //= 2
        image = cv2.copyMakeBorder(img, abs(tb), abs(tb), 0, 0, border_mode)
    else:
        lr //= 2
        image = cv2.copyMakeBorder(img, 0, 0, abs(lr), abs(lr), border_mode)

    if w > h and rotate:
        image = cv2.resize(image, (int(h_scale), int(w_scale)))
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    else:
        image = cv2.resize(image, (int(w_scale), int(h_scale)))

    return image


class MNNModel(metaclass=ABCMeta):
    def __init__(self):
        super(MNNModel, self).__init__()
        self.cost = deque(maxlen=100)
        self.std = deque(maxlen=5)

    def _load_model(self, mnn_path, input_name='input'):
        net_mnn = MNN.Interpreter(mnn_path)
        session = net_mnn.createSession()
        input_tensor = net_mnn.getSessionInput(session, input_name)
        c_scale = input_tensor.getShape()[1]
        w_scale = input_tensor.getShape()[2]
        h_scale = input_tensor.getShape()[3]
        return net_mnn, session, input_tensor, c_scale, w_scale, h_scale

    def cam_inference(self, video=0, show_image=True, dst_path=''):
        save_video = bool(video and dst_path)
        cap = cv2.VideoCapture(video)
        ############# video ###############
        if save_video:
            if dst_path.endswith('.mp4'):
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
                out_video = cv2.VideoWriter(dst_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (2000, 720), True)
            else:
                os.makedirs(dst_path, exist_ok=True)
        while True:
            ret, frame = cap.read()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out = self.img_inference(frame_rgb, show_image=show_image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
            print('>>> {} sumavg cost: {:.2f}ms\n'.format(self.__class__, np.mean(self.cost)*1000))
            if dst_path:
                out_video.write(out[1] if out != None else frame)
        cv2.destroyAllWindows()
        cap.release()
        if dst_path:
            out_video.release()

    @abstractmethod
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

    def _preprocess(self, frame, mean=0, std=1, border_mode=cv2.BORDER_CONSTANT):
        '''
        Preproces, default input is normalized to [0,1].
        '''
        image = frame.copy()
        image = zero_resize_padding(frame, self.h_scale, self.w_scale, border_mode=border_mode)
        # image = cv2.resize(image, (int(self.w_scale), int(self.h_scale)))
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


class MNNLandmarkRegressor(MNNModel):
    def __init__(self, mnn_path=None, mode='full'):
        super(MNNLandmarkRegressor, self).__init__()
        self.landmark_key = 'Identity'
        self.confidence = 'Identity_1'
        self.handedness = 'Identity_2'
        if mnn_path == None and mode == 'full':
            mnn_path = f'{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/models/Mediapipe/hand_landmark_full_fp16.mnn'
            input_name = 'input_1'
        elif mnn_path == None and mode == 'lite':
            mnn_path = f'{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/models/Mediapipe/hand_landmark_lite_fp16.mnn'
            input_name = 'input_1'

        self.mode = mode
        ########### MNN Cfg ##############
        (
            self.net_mnn,
            self.session,
            self.input_tensor,
            self.c_scale,
            self.w_scale,
            self.h_scale,
        ) = self._load_model(mnn_path, input_name=input_name)
        self.cost = []

    def img_inference(self, frame, show_image=False):
        out = self.predict(frame, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], border_mode=cv2.BORDER_REFLECT)
        return out

    def _postprocess(self, output_tensor):
        # ------ mediapipe ------
        # landmark (21, 3)
        landmark = self._get_mnn_output(output_tensor[self.landmark_key]).T.reshape((21, 3)).astype(np.float32)
        # confidence
        confidence = self._get_mnn_output(output_tensor[self.confidence]).astype(np.float32).squeeze()
        # handedness
        handedness = self._get_mnn_output(output_tensor[self.handedness]).astype(np.float32).squeeze()
        return (landmark, confidence, handedness)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="hand landmark")
    parser.add_argument(
        "--mnn_path",
        default="../models/Mediapipe/hand_landmark_lite_fp16.mnn")
    args = parser.parse_args()
    hair_parsing = MNNLandmarkRegressor(
        mnn_path=None,
        mode='lite',
    )
    hair_parsing.cam_inference(show_image=True)
