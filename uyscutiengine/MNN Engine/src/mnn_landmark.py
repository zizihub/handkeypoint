#!/usr/bin/python
# -*- encoding: utf-8 -*-
from .mnn_model import MNNModel
from .one_euro_filter import OneEuroFilter
from .mnn_utils import *
import numpy as np
import cv2
import time
import os
# import matplotlib.pyplot as plt


class MNNLandmarkRegressor(MNNModel):
    def __init__(self, mnn_path=None, mode='full', filtering=True):
        super(MNNLandmarkRegressor, self).__init__()
        self.landmark_key = 'Identity'
        self.confidence = 'Identity_1'
        self.handedness = 'Identity_2'
        self.mean = 0
        self.std = 1.
        if mnn_path == None and mode == 'full':
            mnn_path = f'{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/models/Mediapipe/hand_landmark_full_fp16.mnn'
            input_name = 'input_1'
        elif mnn_path == None and mode == 'lite':
            mnn_path = f'{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/models/Mediapipe/hand_landmark_lite_fp16.mnn'
            input_name = 'input_1'
        elif mnn_path == None and mode == 'handpose_x':
            mnn_path = f'{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/models/mnn/handpose_x.mnn'
            input_name = 'input_1'
            self.mean = [0.5, 0.5, 0.5]
            self.std = [0.5, 0.5, 0.5]
        elif mode == 'xuziyan':
            input_name = 'input.1'
            self.landmark_key = '3143'
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
        else:
            mode = 'others'
            input_name = 'input'
            self.landmark_key = 'joints_3d'
            self.handedness = 'hand_type'
            if 'peclr' in mnn_path or '3dhandpose' in mnn_path:
                self.mean = [0.5, 0.5, 0.5]
                self.std = [0.5, 0.5, 0.5]
            else:
                self.mean = [0.485, 0.456, 0.406]
                self.std = [0.229, 0.224, 0.225]

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
        if filtering:
            self.oe_filter = OneEuroFilter(time.time(), np.zeros((21, 3)),
                                           min_cutoff=1.0,
                                           beta=0.1,
                                           d_cutoff=1.0)
        else:
            self.oe_filter = None
        self.cost = []
        #### plot #########
        # self.vibrations = [0]
        # self.timeseries = [time.time()]
        # self.prev_landmarks = np.zeros((21, 3))
        # plt.ion()

    def img_inference(self, frame, show_image=False):
        out = self.predict(frame, mean=self.mean, std=self.std, border_mode=cv2.BORDER_REFLECT)
        return out

    def _get_hand_image(self, frame, palm_boxes, palm_landmarks):
        img = frame.copy()
        hand_images = []
        for i in range(len(palm_boxes)):
            box = palm_boxes[i]
            landmarks = palm_landmarks[i]
            temp_img, flag, expanded_box = crop_detected_frame(img, box, landmarks)
            if temp_img.shape[0] == 0 or temp_img.shape[1] == 0:
                continue
            hand_images.append((temp_img, flag, expanded_box))
        return hand_images

    def _postprocess(self, output_tensor):
        if self.mode == 'handpose_x':
            # ------ regression ------
            landmark = self._get_mnn_output(output_tensor[self.landmark_key]).T.reshape(
                (21, 2)).astype(np.float32) * self.w_scale
            return (landmark,)
        elif self.mode in ['full', 'lite']:
            # ------ mediapipe ------
            # landmark (21, 3)
            landmark = self._get_mnn_output(output_tensor[self.landmark_key]).T.reshape((21, 3)).astype(np.float32)
            # confidence
            confidence = self._get_mnn_output(output_tensor[self.confidence]).astype(np.float32).squeeze()
            # handedness
            handedness = self._get_mnn_output(output_tensor[self.handedness]).astype(np.float32).squeeze()
            return (landmark, confidence, handedness)
        else:
            if self.mode == 'xuziyan':
                # ------ heatmaps ------
                # landmark
                heatmaps = self._get_mnn_output(output_tensor[self.landmark_key]).astype(np.float32)
                landmark, _ = self._get_max_preds(heatmaps)
                # DARK
                # landmark = self.post_dark_udp(landmark, heatmaps)
                landmark = self.transform_preds(landmark.squeeze(0), heatmaps.shape[2:], use_udp=True)
                return (landmark, )
            else:
                # ------ I2L3DHead ------
                landmark = self._get_mnn_output(output_tensor[self.landmark_key]).reshape((21, 3)).astype(np.float32)
                keypoints_3d = np.zeros((landmark.shape[0], 3), dtype=np.float32)
                keypoints_3d[:, 0] = landmark[:, 0] / 56 * self.w_scale
                keypoints_3d[:, 1] = landmark[:, 1] / 56 * self.h_scale
                keypoints_3d[:, 2] = (landmark[:, 2] / 56 - 0.5) * 400
                handedness = self._get_mnn_output(output_tensor[self.handedness]).astype(np.float32).squeeze()
                return (keypoints_3d, 1.0, handedness[0])

    def _get_max_preds(self, heatmaps):
        """Get keypoint predictions from score maps.

        Note:
            batch_size: N
            num_keypoints: K
            heatmap height: H
            heatmap width: W

        Args:python
            heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.

        Returns:
            tuple: A tuple containing aggregated results.

            - preds (np.ndarray[N, K, 2]): Predicted keypoint location.
            - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
        """
        assert isinstance(heatmaps,
                          np.ndarray), ('heatmaps should be numpy.ndarray')
        assert heatmaps.ndim == 4, 'batch_images should be 4-ndim'

        N, K, _, W = heatmaps.shape
        heatmaps_reshaped = heatmaps.reshape((N, K, -1))
        idx = np.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
        maxvals = np.amax(heatmaps_reshaped, 2).reshape((N, K, 1))

        preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
        preds[:, :, 0] = preds[:, :, 0] % W
        preds[:, :, 1] = preds[:, :, 1] // W

        preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
        return preds, maxvals

    def transform_preds(self, coords, output_size, use_udp=True):
        """Get final keypoint predictions from heatmaps and apply scaling and
        translation to map them back to the image.

        Note:
            num_keypoints: K

        Args:
            coords (np.ndarray[K, ndims]):

                * If ndims=2, corrds are predicted keypoint location.
                * If ndims=4, corrds are composed of (x, y, scores, tags)
                * If ndims=5, corrds are composed of (x, y, scores, tags,
                flipped_tags)

            center (np.ndarray[2, ]): Center of the bounding box (x, y).
            scale (np.ndarray[2, ]): Scale of the bounding box
                wrt [width, height].
            output_size (np.ndarray[2, ] | list(2,)): Size of the
                destination heatmaps.
            use_udp (bool): Use unbiased data processing

        Returns:
            np.ndarray: Predicted coordinates in the images.
        """
        scale = np.array([self.h_scale, self.w_scale])
        center = np.array([self.h_scale/2, self.w_scale/2])
        assert coords.shape[1] in (2, 4, 5)
        assert len(output_size) == 2

        if use_udp:
            scale_x = scale[0] / (output_size[0] - 1.0)
            scale_y = scale[1] / (output_size[1] - 1.0)
        else:
            scale_x = scale[0] / output_size[0]
            scale_y = scale[1] / output_size[1]

        target_coords = np.ones_like(coords)
        target_coords[:, 0] = coords[:, 0] * scale_x + center[0] - scale[0] * 0.5
        target_coords[:, 1] = coords[:, 1] * scale_y + center[1] - scale[1] * 0.5

        return target_coords

    @staticmethod
    def post_dark_udp(coords, batch_heatmaps, kernel=3):
        """DARK post-pocessing. Implemented by udp. Paper ref: Huang et al. The
        Devil is in the Details: Delving into Unbiased Data Processing for Human
        Pose Estimation (CVPR 2020). Zhang et al. Distribution-Aware Coordinate
        Representation for Human Pose Estimation (CVPR 2020).

        Note:
            - batch size: B
            - num keypoints: K
            - num persons: N
            - height of heatmaps: H
            - width of heatmaps: W

            B=1 for bottom_up paradigm where all persons share the same heatmap.
            B=N for top_down paradigm where each person has its own heatmaps.

        Args:
            coords (np.ndarray[N, K, 2]): Initial coordinates of human pose.
            batch_heatmaps (np.ndarray[B, K, H, W]): batch_heatmaps
            kernel (int): Gaussian kernel size (K) for modulation.

        Returns:
            np.ndarray([N, K, 2]): Refined coordinates.
        """
        if not isinstance(batch_heatmaps, np.ndarray):
            batch_heatmaps = batch_heatmaps.cpu().numpy()
        B, K, H, W = batch_heatmaps.shape
        N = coords.shape[0]
        assert (B == 1 or B == N)
        for heatmaps in batch_heatmaps:
            for heatmap in heatmaps:
                cv2.GaussianBlur(heatmap, (kernel, kernel), 0, heatmap)
        np.clip(batch_heatmaps, 0.001, 50, batch_heatmaps)
        np.log(batch_heatmaps, batch_heatmaps)

        batch_heatmaps_pad = np.pad(
            batch_heatmaps, ((0, 0), (0, 0), (1, 1), (1, 1)),
            mode='edge').flatten()

        index = coords[..., 0] + 1 + (coords[..., 1] + 1) * (W + 2)
        index += (W + 2) * (H + 2) * np.arange(0, B * K).reshape(-1, K)
        index = index.astype(int).reshape(-1, 1)
        i_ = batch_heatmaps_pad[index]
        ix1 = batch_heatmaps_pad[index + 1]
        iy1 = batch_heatmaps_pad[index + W + 2]
        ix1y1 = batch_heatmaps_pad[index + W + 3]
        ix1_y1_ = batch_heatmaps_pad[index - W - 3]
        ix1_ = batch_heatmaps_pad[index - 1]
        iy1_ = batch_heatmaps_pad[index - 2 - W]

        dx = 0.5 * (ix1 - ix1_)
        dy = 0.5 * (iy1 - iy1_)
        derivative = np.concatenate([dx, dy], axis=1)
        derivative = derivative.reshape(N, K, 2, 1)
        dxx = ix1 - 2 * i_ + ix1_
        dyy = iy1 - 2 * i_ + iy1_
        dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
        hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1)
        hessian = hessian.reshape(N, K, 2, 2)
        hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
        coords -= np.einsum('ijmn,ijnk->ijmk', hessian, derivative).squeeze()
        return coords
