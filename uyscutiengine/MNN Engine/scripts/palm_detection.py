#!/usr/bin/python
# -*- encoding: utf-8 -*-
import MNN
import numpy as np
import cv2
import os
from time import time
from collections import deque
from abc import ABCMeta, abstractmethod

__all__ = ['MNNModel']


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


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


def get_iou(base_box, compare_box):
    x1_insection = np.maximum(base_box[0] - base_box[2] / 2,
                              compare_box[:, 0] - compare_box[:, 2] / 2)
    y1_insection = np.maximum(base_box[1] - base_box[3] / 2,
                              compare_box[:, 1] - compare_box[:, 3] / 2)
    x2_insection = np.minimum(base_box[0] + base_box[2] / 2,
                              compare_box[:, 0] + compare_box[:, 2] / 2)
    y2_insection = np.minimum(base_box[1] + base_box[3] / 2,
                              compare_box[:, 1] + compare_box[:, 3] / 2)
    width_insection = np.maximum(0, x2_insection - x1_insection)
    height_insection = np.maximum(0, y2_insection - y1_insection)
    area_insection = width_insection * height_insection
    area_union = base_box[2] * base_box[3] + compare_box[:, 2] * compare_box[:, 3] - area_insection
    iou = area_insection / area_union
    return iou


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
        w_scale = input_tensor.getShape()[2]
        h_scale = input_tensor.getShape()[3]
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

    def _preprocess(self, frame, mean=0, std=1, border_mode=cv2.BORDER_CONSTANT):
        '''
        Preproces, default input is normalized to [0,1].
        '''
        image = frame.copy()
        image = zero_resize_padding(frame, self.h_scale, self.w_scale, border_mode=border_mode)
        cv2.imshow('input', image)
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


class BlazeMNNDetector(MNNModel):
    def __init__(self, mode):
        super(BlazeMNNDetector, self).__init__()
        self.num_classes = 1
        self.x_scale = 128.0
        self.y_scale = 128.0
        self.num_coords = 18
        self.classificators = 'classificators'
        self.regressors = 'regressors'
        self.mnn_path = f'{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/models/Mediapipe/palm_detection_fp16.mnn'
        self.num_anchors = 896
        self.load_anchors(f'{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/models/anchors.npy')

        self.score_clipping_thresh = 100.0
        self.min_suppression_threshold = 0.3
        self.min_score_thresh = 0.75
        print('load {} model'.format(mode))
        ########### MNN Cfg ##############
        (
            self.net_mnn,
            self.session,
            self.input_tensor,
            self.c_scale,
            self.w_scale,
            self.h_scale
        ) = self._load_model(self.mnn_path)
        self.mean = 0.5
        self.std = 0.5

    def load_anchors(self, path):
        self._anchors = np.load(path).astype(np.float32)
        assert(self._anchors.ndim == 2)
        assert(self._anchors.shape[0] == self.num_anchors)
        assert(self._anchors.shape[1] == 4)

    def img_inference(self, frame, show_image=True):
        post_outs = self.predict(frame, mean=self.mean, std=self.std)
        outs = self.post_out(frame, post_outs)
        print(outs)
        self._plot_detections(frame, outs, show_image=show_image)
        return outs

    def _postprocess(self, output_tensor):
        # classificators
        raw_score_ndarray = self._get_mnn_output(output_tensor[self.classificators])
        # regressors
        raw_box_ndarray = self._get_mnn_output(output_tensor[self.regressors])
        # filter raw output
        detections = self._ndarray_to_detections(raw_box_ndarray, raw_score_ndarray)
        filtered_detections = []
        for i in range(len(detections)):
            # do NMS
            boxes = self._weighted_nms(detections[i])
            boxes = np.stack(boxes) if len(boxes) > 0 else np.zeros((0, self.num_anchors))
            filtered_detections.append(boxes)
        return filtered_detections[0]

    def _ndarray_to_detections(self, raw_box_ndarray, raw_score_ndarray):
        '''
        decode raw output into standard detections
        '''
        assert raw_box_ndarray.ndim == 3
        assert raw_box_ndarray.shape[1] == self.num_anchors
        assert raw_box_ndarray.shape[2] == self.num_coords
        assert raw_score_ndarray.ndim == 3
        assert raw_score_ndarray.shape[1] == self.num_anchors
        assert raw_score_ndarray.shape[2] == self.num_classes
        assert raw_box_ndarray.shape[0] == raw_score_ndarray.shape[0]

        # decode raw regressors and select boxes which are over score-threshold
        detections_boxes = self._decode_boxes(raw_box_ndarray)
        thresh = self.score_clipping_thresh
        raw_score_ndarray = raw_score_ndarray.clip(-thresh, thresh)
        detections_scores = sigmoid(raw_score_ndarray).squeeze(-1)
        mask = detections_scores > self.min_score_thresh
        # refactor output format into Nx5 shape (x, y, w, h, scores)
        output_detections = []
        for i in range(raw_box_ndarray.shape[0]):
            boxes = detections_boxes[i, mask[i]]
            scores = detections_scores[i, mask[i]][..., None]
            output_detections.append(np.concatenate([boxes, scores], axis=-1))
        return output_detections

    def _weighted_nms(self, detections):
        '''
        weighted NMS: Inception Single Shot MultiBox Detector for object detection
        paper url: https://ieeexplore.ieee.org/document/8026312
        '''
        if len(detections) == 0:
            return []
        output_detections = []
        remaining = np.argsort(detections[:, self.num_coords])[::-1]
        while len(remaining) > 0:
            detection = detections[remaining[0]]
            # Compute the overlap between the first box and the other
            # remaining boxes. (Note that the other_boxes also include
            # the first_box.)
            first_box = detection[:4]
            other_boxes = detections[remaining, :4]
            ious = get_iou(first_box, other_boxes)
            # jump out of loop
            if ious.sum() == 0:
                return []
            # If two detections don't overlap enough, they are considered
            # to be from different faces.
            mask = ious > self.min_suppression_threshold
            overlapping = remaining[mask]
            remaining = remaining[~mask]

            # Take an average of the coordinates from the overlapping
            # detections, weighted by their confidence scores.
            weighted_detection = detection.copy()
            if len(overlapping) > 1:
                coordinates = detections[overlapping, :self.num_coords]
                scores = detections[overlapping, self.num_coords:self.num_coords+1]
                total_score = scores.sum()
                weighted = (coordinates * scores).sum(axis=0) / total_score
                weighted_detection[:self.num_coords] = weighted
                weighted_detection[self.num_coords] = total_score / len(overlapping)

            output_detections.append(weighted_detection)
        return output_detections

    def _decode_boxes(self, raw_boxes):
        '''
        Converts the predictions into actual coordinates using
        the anchor boxes. Processes the entire batch at once.
        '''
        boxes = np.zeros(raw_boxes.shape)

        x_center = raw_boxes[..., 0] / self.x_scale + self._anchors[:, 0]
        y_center = raw_boxes[..., 1] / self.y_scale + self._anchors[:, 1]

        w = raw_boxes[..., 2] / self.w_scale
        h = raw_boxes[..., 3] / self.h_scale

        boxes[..., 0] = x_center
        boxes[..., 1] = y_center
        boxes[..., 2] = w
        boxes[..., 3] = h

        for k in range((self.num_coords-4)//2):
            offset = 4 + k*2
            keypoint_x = raw_boxes[..., offset] / self.x_scale + self._anchors[:, 0]
            keypoint_y = raw_boxes[..., offset + 1] / self.y_scale + self._anchors[:, 1]
            boxes[..., offset] = keypoint_x
            boxes[..., offset + 1] = keypoint_y

        return boxes

    def _recover_box(self, boxes, frame, padding=True):
        '''
        remove zero padding
        '''
        height = frame.shape[0]
        width = frame.shape[1]
        pad_size = [width, height]
        if padding:
            pad_size = [max(height, width)] * 2
        boxes[..., 0] = boxes[..., 0] * pad_size[0] - (pad_size[0] - width) / 2
        boxes[..., 1] = boxes[..., 1] * pad_size[1] - (pad_size[1] - height) / 2
        boxes[..., 2] = boxes[..., 2] * pad_size[0]
        boxes[..., 3] = boxes[..., 3] * pad_size[1]
        for k in range((self.num_coords-4)//2):
            offset = 4 + k*2
            keypoint_x = boxes[..., offset] * pad_size[0] - (pad_size[0] - width) / 2
            keypoint_y = boxes[..., offset+1] * pad_size[1] - (pad_size[1] - height) / 2
            boxes[..., offset] = keypoint_x
            boxes[..., offset+1] = keypoint_y

        return boxes

    def post_out(self, img, detections):
        detections = self._recover_box(detections, img)
        out_boxes = []
        out_landmarks = []
        for i in range(detections.shape[0]):
            ymin = detections[i, 1] - detections[i, 3] // 2
            xmin = detections[i, 0] - detections[i, 2] // 2
            ymax = detections[i, 1] + detections[i, 3] // 2
            xmax = detections[i, 0] + detections[i, 2] // 2
            out_boxes.append([xmin, ymin, xmax, ymax])
            out_landmarks.append(detections[i, 4:-1])
        return {'boxes': out_boxes, 'landmarks': out_landmarks}

    def _plot_detections(self, ori_img, outs, with_keypoints=True, show_image=False):
        img = cv2.cvtColor(ori_img.copy(), cv2.COLOR_RGB2BGR)
        boxes, landmarks = outs['boxes'], outs['landmarks']
        print('Found %d object(s)' % len(boxes))
        for box, landmark in zip(boxes, landmarks):
            xmin, ymin, xmax, ymax = box
            img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 4)
            if with_keypoints:
                for k in range((self.num_coords-4)//2):
                    kp_x = landmark[k*2]
                    kp_y = landmark[k*2 + 1]
                    ori_img = cv2.circle(img, (int(kp_x), int(kp_y)), 1, (0, 255, 0), 4)
                    ori_img = cv2.putText(img, str(k), (int(kp_x), int(kp_y)), 0, 1, (0, 0, 0), 1)
        if show_image:
            cv2.imshow('blaze_detector', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == '__main__':
    md = BlazeMNNDetector('palm')
    # md.cam_inference(show_image=True)
    frame = cv2.imread('./input.jpg')
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    md.img_inference(frame, True)
