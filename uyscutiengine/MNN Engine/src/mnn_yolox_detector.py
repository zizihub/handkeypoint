#!/usr/bin/python
# -*- encoding: utf-8 -*-
from matplotlib.transforms import Bbox
from .mnn_blaze_detector import BlazeMNNDetector
from .mnn_utils import *
from .one_euro_filter import OneEuroFilter
import MNN
import numpy as np
import cv2
import os
import time


class YOLOXDetector(BlazeMNNDetector):
    def __init__(self, mnn_path, filtering=True):
        super(YOLOXDetector, self).__init__(mnn_path=mnn_path, filtering=filtering)
        self.num_classes = 1
        self.num_coords = 18
        self.classificators = 'classificators'
        self.regressors = 'regressors'
        self.priors = 'flatten_priors'
        self.min_suppression_threshold = 0.30
        self.min_score_thresh = 0.50

        self.ratio = (1, 1)
        # self.ratio = (15, 9)
        self.rotate = False
        self.letterbox = True
        self.border_mode = cv2.BORDER_REPLICATE

    def _preprocess(self, frame, mean=0, std=1, border_mode=cv2.BORDER_CONSTANT):
        '''
        Preproces, default input is normalized to [0,1].
        '''
        image = frame.copy()
        image = zero_resize_padding(image, self.h_scale, self.w_scale,
                                    ratio=self.ratio,
                                    border_mode=self.border_mode,
                                    rotate=self.rotate,
                                    letter_box=self.letterbox)
        print('padding size: {} | ratio: {:.3f}'.format(image.shape, max(image.shape[:2])/min(image.shape[:2])))
        image = self._to_tensor(image, mean, std)
        self.debug_show(image)
        return image

    def _postprocess(self, output_tensor):
        # classificators
        raw_score_ndarray = self._get_mnn_output(output_tensor[self.classificators])
        # regressors
        raw_box_ndarray = self._get_mnn_output(output_tensor[self.regressors])
        # priors
        raw_prior_ndarray = self._get_mnn_output(output_tensor[self.priors])
        self.num_anchors = raw_prior_ndarray.shape[0]
        # filter raw output
        detections = self._ndarray_to_detections(raw_box_ndarray, raw_score_ndarray, raw_prior_ndarray)
        filtered_detections = []
        for i in range(len(detections)):
            # do NMS
            boxes = self._weighted_nms(detections[i], mode='xywh')
            boxes = np.stack(boxes) if len(boxes) > 0 else np.zeros((0, self.num_anchors))
            filtered_detections.append(boxes)
        return filtered_detections[0]

    def _ndarray_to_detections(self, raw_box_ndarray, detections_scores, raw_prior_ndarray):
        '''
        decode raw output into standard detections
        '''
        # decode raw regressors and select boxes which are over score-threshold
        detections_boxes = self._decode_boxes(raw_prior_ndarray, raw_box_ndarray)
        # detections_scores = detections_scores.reshape(1, -1)
        mask = detections_scores > self.min_score_thresh
        # refactor output format into Nx5 shape (x, y, w, h, scores)
        output_detections = []
        for i in range(raw_box_ndarray.shape[0]):
            boxes = detections_boxes[i, mask[i]]
            scores = detections_scores[i, mask[i]][..., None]
            output_detections.append(np.concatenate([boxes, scores], axis=-1))
        return output_detections

    def _decode_boxes(self, priors, bbox_preds):
        bbox_preds = bbox_preds.clip(-10, 10)
        xys = (bbox_preds[..., :2] * priors[:, 2:]) + priors[:, :2]
        whs = np.exp(bbox_preds[..., 2:4]) * priors[:, 2:]
        decoded_bboxes = np.concatenate([xys, whs], axis=2)
        keypoints = (bbox_preds[..., 4:] * np.tile(priors[:, 2:], (1, 7))) + np.tile(priors[:, :2], (1, 7))
        decoded_bboxes = np.concatenate([decoded_bboxes, keypoints], axis=2)
        return decoded_bboxes

    def _recover_box(self, boxes, frame):
        '''
        remove zero padding
        '''
        height = frame.shape[0]
        width = frame.shape[1]
        pad_scale = max([self.w_scale, self.h_scale])
        pad_size = max([width, height])
        boxes[..., :18] = boxes[..., :18] / pad_scale * pad_size
        if self.letterbox:
            offset_width = (pad_size - width) / 2
            offset_height = (pad_size - height) / 2
            boxes[..., 0] -= offset_width
            boxes[..., 1] -= offset_height
            boxes[..., 4:18:2] -= offset_width
            boxes[..., 5:18:2] -= offset_height
        if self.rotate:
            # if rotate90
            temp_boxes = boxes.copy()
            center = (height // 2, height // 2)
            rotation_matrix = self._get_rotation_matrix(center, -90)
            # bbox rotate90 recover
            num_bboxes = len(temp_boxes)
            xs = temp_boxes[:, 0].reshape(num_bboxes * 1)
            ys = temp_boxes[:, 1].reshape(num_bboxes * 1)
            ones = np.ones_like(xs)
            points = np.vstack([xs, ys, ones])
            warp_points = rotation_matrix @ points
            warp_points = warp_points[:2] / warp_points[2]
            xs = warp_points[0].reshape(num_bboxes, 1)
            ys = warp_points[1].reshape(num_bboxes, 1)
            boxes[:, :2] = np.concatenate([xs, ys], axis=1)
            # point rotate90 recover
            keypoints = temp_boxes[:, 4:18].copy().reshape(-1, 7, 2)
            warp_keypoints = np.zeros_like(keypoints)
            for i in range(keypoints.shape[0]):
                for j in range(keypoints.shape[1]):
                    warp_keypoints[i, j, 0:2] = self.affine_transform(keypoints[i, j, 0:2], rotation_matrix)
            boxes[:, 4:18] = warp_keypoints.reshape(-1, 14)
        return boxes

    @staticmethod
    def affine_transform(pt, trans_mat):
        """Apply an affine transformation to the points.

        Args:
            pt (np.ndarray): a 2 dimensional point to be transformed
            trans_mat (np.ndarray): 2x3 matrix of an affine transform

        Returns:
            np.ndarray: Transformed points.
        """
        assert len(pt) == 2
        new_pt = np.array(trans_mat) @ np.array([pt[0], pt[1], 1.])

        return new_pt[:2]

    @staticmethod
    def _get_rotation_matrix(center, rotate_degrees):
        radian = math.radians(rotate_degrees)
        x, y = center
        cos_theta = np.cos(radian)
        sin_theta = np.sin(radian)
        rotation_matrix = np.array(
            [[cos_theta, -sin_theta, -x*cos_theta + y*sin_theta + x],
             [sin_theta, cos_theta, -x*sin_theta - y*cos_theta + y],
             [0., 0., 1.]],
            dtype=np.float32)
        return rotation_matrix
