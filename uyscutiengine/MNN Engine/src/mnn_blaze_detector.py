#!/usr/bin/python
# -*- encoding: utf-8 -*-
from .mnn_model import MNNModel
from .mnn_utils import *
from .one_euro_filter import OneEuroFilter
import MNN
import numpy as np
import cv2
import os
import time


class BlazeMNNDetector(MNNModel):
    def __init__(self, mnn_path=None, mode=None, filtering=True):
        super(BlazeMNNDetector, self).__init__()
        self.num_classes = 1
        if mode == 'face':
            self.num_coords = 16
            self.classificators = 'classificators'
            self.regressors = 'regressors'
            self.mnn_path = f'{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/models/Mediapipe/face_detection_short_range.mnn'
            self.num_anchors = 896
            self.load_anchors(
                f'{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/models/anchors/anchor_face_shortrange.npy')
        elif mode == 'face_fullrange':
            self.num_coords = 16
            self.classificators = 'reshaped_classifier_face_4'
            self.regressors = 'reshaped_regressor_face_4'
            self.mnn_path = f'{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/models/Mediapipe/face_detection_full_range.mnn'
            self.num_anchors = 2304
            self.load_anchors(
                f'{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/models/anchors/anchor_face_fullrange.npy')

        elif mode == 'palm':
            self.num_coords = 18
            self.classificators = 'classificators'
            self.regressors = 'regressors'
            self.mnn_path = f'{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/models/Mediapipe/palm_detection_fp16.mnn'
            self.num_anchors = 896
            self.load_anchors(
                f'{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/models/anchors/anchor_face_shortrange.npy')
        elif mode == 'palm_shortrange':
            self.num_coords = 18
            self.classificators = 'Identity_1'
            self.regressors = 'Identity'
            self.mnn_path = f'{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/models/Mediapipe/palm_detection_fp16.mnn'
            self.num_anchors = 896
            self.load_anchors(
                f'{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/models/anchors/anchor_palm_fullrange.npy')
        elif mode == 'palm_fullrange':
            self.num_coords = 18
            self.classificators = 'Identity_1'
            self.regressors = 'Identity'
            self.mnn_path = f'{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/models/Mediapipe/palm_detection_fp16.mnn'
            self.num_anchors = 896
            self.load_anchors(
                f'{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/models/anchors/anchor_palm_fullrange.npy')
        else:
            self.mnn_path = mnn_path

        self.score_clipping_thresh = 100.0
        self.min_suppression_threshold = 0.3
        self.min_score_thresh = 0.75
        print('load {} model'.format(mode))
        ########### One Euro #############
        if filtering:
            self.oe_filter = OneEuroFilter(time.time(), np.zeros(self.num_coords+1),
                                           min_cutoff=1.0,
                                           beta=0.1,
                                           d_cutoff=1.0)
        else:
            self.oe_filter = None
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
        # anchor mode [xc,yc,w,h]
        self._anchors = np.load(path).astype(np.float32)
        assert(self._anchors.ndim == 2)
        assert(self._anchors.shape[0] == self.num_anchors)
        assert(self._anchors.shape[1] == 4)

    def img_inference(self, frame, show_image=True):
        post_outs = self.predict(frame, mean=self.mean, std=self.std)
        print(len(post_outs))
        outs = self.post_out(frame, post_outs)
        vis_img = self._plot_detections(frame, outs, show_image=show_image)
        return outs, vis_img

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

    def _weighted_nms(self, detections, mode='xywh'):
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
            ious = get_iou(first_box, other_boxes, mode=mode)
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

        x_center = raw_boxes[..., 0] / self.w_scale + self._anchors[:, 0]
        y_center = raw_boxes[..., 1] / self.h_scale + self._anchors[:, 1]

        w = raw_boxes[..., 2] / self.w_scale
        h = raw_boxes[..., 3] / self.h_scale

        boxes[..., 0] = x_center
        boxes[..., 1] = y_center
        boxes[..., 2] = w
        boxes[..., 3] = h

        for k in range((self.num_coords-4)//2):
            offset = 4 + k*2
            keypoint_x = raw_boxes[..., offset] / self.w_scale + self._anchors[:, 0]
            keypoint_y = raw_boxes[..., offset + 1] / self.h_scale + self._anchors[:, 1]
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
        out_scores = []
        for i in range(detections.shape[0]):
            # one-euro filter
            if self.oe_filter:
                detections[i] = self.oe_filter(time.time(), detections[i])
            xmin = detections[i, 0] - detections[i, 2] // 2
            ymin = detections[i, 1] - detections[i, 3] // 2
            xmax = detections[i, 0] + detections[i, 2] // 2
            ymax = detections[i, 1] + detections[i, 3] // 2
            out_boxes.append([xmin, ymin, xmax, ymax])
            out_landmarks.append(detections[i, 4:-1])
            out_scores.append(str(detections[i, -1]))
        return {'boxes': out_boxes, 'landmarks': out_landmarks, 'scores': out_scores}

    def _plot_detections(self, ori_img, outs, with_keypoints=True, show_image=False):
        img = cv2.cvtColor(ori_img.copy(), cv2.COLOR_RGB2BGR)
        boxes, landmarks, scores = outs['boxes'], outs['landmarks'], outs['scores']
        print('Found %d object(s)' % len(boxes))
        for box, landmark, score in zip(boxes, landmarks, scores):
            xmin, ymin, xmax, ymax = np.array(box).astype(np.int32)
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 4)
            # roughly estimate the proper font size
            text_size, text_baseline = cv2.getTextSize(score,
                                                       cv2.FONT_HERSHEY_DUPLEX,
                                                       0.5, 1)
            text_x1 = int(box[0])
            text_y1 = int(max(0, box[1] - text_size[1] - text_baseline))
            text_x2 = int(box[0] + text_size[0])
            text_y2 = int(text_y1 + text_size[1] + text_baseline)
            cv2.rectangle(img, (text_x1, text_y1), (text_x2, text_y2), (255, 0, 0), cv2.FILLED)
            cv2.putText(img, score, (text_x1, text_y2 - text_baseline),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
            if with_keypoints:
                for k in range((self.num_coords-4)//2):
                    kp_x = landmark[k*2]
                    kp_y = landmark[k*2 + 1]
                    cv2.circle(img, (int(kp_x), int(kp_y)), 1, (0, 255, 0), 4)
                    cv2.putText(img, str(k), (int(kp_x), int(kp_y)), 0, 1, (0, 0, 0), 1)
        if show_image:
            cv2.imshow('blaze_detector', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        return img
