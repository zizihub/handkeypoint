#!/usr/bin/python
# -*- encoding: utf-8 -*-
import MNN
import numpy as np
import cv2
from time import time
from collections import deque
from abc import ABCMeta, abstractmethod


# ------------------------------------------------------------
#                          utils
# ------------------------------------------------------------


def draw_mask(frame, parsing, show_image=False):
    # Colors for all 20 parts
    part_colors = [
        [255, 0, 0], [255, 85, 0], [105, 128, 112],
        [85, 96, 225], [255, 0, 170],
        [0, 255, 0], [85, 0, 255], [170, 255, 0],
        [255, 255, 255], [0, 255, 170],
        [0, 0, 255], [85, 0, 255], [170, 0, 255],
        [0, 85, 255], [0, 170, 255],
        [255, 255, 0], [255, 255, 85], [255, 255, 170],
        [255, 0, 255], [255, 85, 255], [255, 170, 255],
        [0, 255, 255], [85, 255, 255], [170, 255, 255]
    ]
    img = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2BGR)
    vis_parsing = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8) + 255
    num_of_class = np.max(parsing)
    for pi in range(1, num_of_class + 1):
        # if pi != 5 and pi != 6 and pi != 7:
        #     continue
        index = np.where(parsing == pi)
        vis_parsing[index[0], index[1], :] = part_colors[pi]
    print(vis_parsing.shape, frame.shape)
    img = cv2.addWeighted(img, 0.75, vis_parsing, 0.25, 0)
    if show_image:
        cv2.imshow('Demo', img)
        cv2.imshow('blank', vis_parsing)
    return parsing


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


def recover_zero_padding(frame, post_out, ratio=(1, 1)):
    '''
    mask remove zero padding
    '''
    h, w, _ = frame.shape

    def recover_single(out):
        # rotated
        if w > h:
            out = cv2.rotate(out, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # rotated and height:width > 16:9
            h_o, w_o = out.shape[:2]
            if w / h > ratio[0]/ratio[1]:
                ll = int(h_o - h * w_o/w)
                return out[ll//2:-(ll-ll//2), :]
            # rotated and height:width <= 16:9
            elif w / h < ratio[0]/ratio[1]:
                ll = int(w_o - w * h_o/h)
                return out[:, ll//2:-(ll-ll//2)]
            else:
                return out
        else:
            h_o, w_o = out.shape[:2]
            # height:width > 16:9
            if h / w > ratio[0]/ratio[1]:
                ll = int(w_o - w * h_o/h)
                return out[:, ll//2:-(ll-ll//2)]
            # height:width <= 16:9
            elif h / w < ratio[0]/ratio[1]:
                ll = int(h_o - h * w_o/w)
                return out[ll//2:-(ll-ll//2), :]
            else:
                return out
    if isinstance(post_out, (list, tuple)):
        result = []
        for out in post_out:
            result.append(recover_single(out))
    else:
        result = recover_single(post_out)
    return result


# ------------------------------------------------------------
#                          Base Model Class
# ------------------------------------------------------------


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

    def cam_inference(self, video=0, show_image=True):
        cap = cv2.VideoCapture(video)
        while True:
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.img_inference(frame, show_image=show_image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
            print('>>> {} sumavg cost: {:.2f}ms\n'.format(self.__class__, np.mean(self.cost)*1000))
        cv2.destroyAllWindows()

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
        image = image / 255.0
        image = (image - mean) / std
        # cv2 read shape is NHWC, Tensor's need is NCHW,transpose it
        image = image.transpose((2, 0, 1))
        # change numpy data type as np.float32 to match tensor's format
        image = image.astype(np.float32)
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


# ------------------------------------------------------------
#                          Segmentation Model Class
# ------------------------------------------------------------


class MNNParsing(MNNModel):
    def __init__(self,
                 mnn_path,
                 num_class,
                 optical_flow=False,
                 **kwargs):
        super(MNNParsing, self).__init__()
        self.num_class = num_class
        ########### MNN Cfg ##############
        (
            self.net_mnn,
            self.session,
            self.input_tensor,
            self.c_scale,
            self.h_scale,
            self.w_scale,
        ) = self._load_model(mnn_path)
        self.ratio = kwargs.pop('ratio', (16, 9))
        self.rotate = kwargs.pop('rotate', True)
        self.prev_mask = None
        # optical flow
        self.optical_flow = optical_flow
        self.disflow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        self.prev_gray = np.zeros((self.h_scale, self.w_scale), np.uint8)
        self.prev_fused = np.zeros((self.h_scale, self.w_scale), np.float32)
        self.is_init = True

    def img_inference(self, frame, show_image=False):
        post_outs = self.predict(frame, mean=0.5, std=0.5)
        # optical flow segmentation fused
        if show_image:
            parsing = recover_zero_padding(frame, post_outs, ratio=(16, 9))
            parsing = cv2.resize(parsing, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            parsing = draw_mask(frame, parsing, show_image=show_image)
            return parsing
        else:
            return post_outs

    def _preprocess(self, frame, mean=0.5, std=0.5, border_mode=cv2.BORDER_CONSTANT):
        '''
        Preproces, default input is normalized to [0,1].
        '''
        image = frame.copy()
        h, w, c = image.shape
        image = zero_resize_padding(frame, self.h_scale, self.w_scale,
                                    ratio=self.ratio,
                                    border_mode=border_mode,
                                    rotate=self.rotate)
        print('padding size: {} | ratio: {:.3f}'.format(image.shape, max(image.shape[:2])/min(image.shape[:2])))
        image = self._to_tensor(image, mean, std)

        if self.c_scale == 4:
            if isinstance(self.prev_mask, np.ndarray):
                # prev_mask = self._to_tensor(self.prev_mask, mean, std)
                prev_mask = self.prev_mask.transpose(2, 0, 1)[np.newaxis, ...]
            else:
                prev_mask = self._to_tensor(np.zeros((self.h_scale, self.w_scale, 1)), mean, std)
            print(image.shape, prev_mask.shape)
            image = np.concatenate([image, prev_mask], axis=1)
        self.debug_show(image)
        return image

    def _postprocess(self, output_tensor):
        try:
            mat_np = self._get_mnn_output(output_tensor['masks'])
        except:
            for k, v in output_tensor.items():
                if v.getShape() == (1, 2, self.h_scale, self.w_scale):
                    mat_np = self._get_mnn_output(v)
        last_fuse = mat_np.squeeze(0).transpose(1, 2, 0)
        self.prev_mask = last_fuse.argmax(2)[..., np.newaxis]
        # cv2.imshow('mask_prob', cv2.resize(mask_remove_padding(
        #     np.zeros([720, 1280, 3]), last_fuse[:, :, 1]), (1280, 720)))
        last_fuse = last_fuse.argmax(2)
        print('output shape', last_fuse.shape, np.unique(last_fuse))
        self.std.append(last_fuse)
        print('>>> stability', np.array(self.std).std(axis=0).mean())
        return last_fuse

    @ staticmethod
    def debug_show(image):
        image = (image.squeeze(0).transpose((1, 2, 0))[:, :, ::-1] + 1) * 127.5
        cv2.imshow('debug show', cv2.resize(image.astype(np.uint8), None, fx=2, fy=2))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="hair parsing")
    parser.add_argument(
        "--mnn_path",
        default="../models/mnn/hairseg-ddr23slim-sz512_fp16.mnn")
    args = parser.parse_args()
    hair_parsing = MNNParsing(
        mnn_path=args.mnn_path,
        num_class=1,
    )
    hair_parsing.cam_inference(show_image=True)
