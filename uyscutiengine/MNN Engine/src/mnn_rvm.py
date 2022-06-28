#!/usr/bin/python
# -*- encoding: utf-8 -*-
import MNN
import numpy as np
import cv2
import os
import numpy as np
from .mnn_utils import *
from .mnn_model import MNNModel


class RVMModel(MNNModel):
    def __init__(self, mnn_path):
        super(RVMModel, self).__init__()
        (
            self.net_mnn,
            self.session,
            self.input_tensor,
            self.c_scale,
            self.w_scale,
            self.h_scale,
        ) = self._load_model(mnn_path, input_name='src')
        self.bgr = np.zeros((720, 1280, 3)) + (162, 250, 139)
        self._mnn_initial()

    def _mnn_initial(self):
        self.net_mnn.resizeTensor(self.input_tensor, (1, 3, self.h_scale, self.w_scale))
        self.r1i = self.net_mnn.getSessionInput(self.session, "r1i")
        self.r2i = self.net_mnn.getSessionInput(self.session, "r2i")
        self.r3i = self.net_mnn.getSessionInput(self.session, "r3i")
        self.r4i = self.net_mnn.getSessionInput(self.session, "r4i")

        r1_h, r1_w = self.h_scale//2, self.w_scale//2
        r2_h, r2_w = self.h_scale//4, self.w_scale//4
        r3_h, r3_w = self.h_scale//8, self.w_scale//8
        r4_h, r4_w = self.h_scale//16, self.w_scale//16

        self.net_mnn.resizeTensor(self.r1i, (1, 16, r1_h, r1_w))
        self.net_mnn.resizeTensor(self.r2i, (1, 20, r2_h, r2_w))
        self.net_mnn.resizeTensor(self.r3i, (1, 40, r3_h, r3_w))
        self.net_mnn.resizeTensor(self.r4i, (1, 64, r4_h, r4_w))
        self.net_mnn.resizeSession(self.session)

        tmp_r1i = MNN.Tensor((1, 16, r1_h, r1_w),
                             MNN.Halide_Type_Float,
                             np.zeros((1, 16, r1_h, r1_w), dtype=np.float32),
                             MNN.Tensor_DimensionType_Caffe,)
        tmp_r2i = MNN.Tensor((1, 20, r2_h, r2_w),
                             MNN.Halide_Type_Float,
                             np.zeros((1, 20, r2_h, r2_w), dtype=np.float32),
                             MNN.Tensor_DimensionType_Caffe,)
        tmp_r3i = MNN.Tensor((1, 40, r3_h, r3_w),
                             MNN.Halide_Type_Float,
                             np.zeros((1, 40, r3_h, r3_w), dtype=np.float32),
                             MNN.Tensor_DimensionType_Caffe,)
        tmp_r4i = MNN.Tensor((1, 64, r4_h, r4_w),
                             MNN.Halide_Type_Float,
                             np.zeros((1, 64, r4_h, r4_w), dtype=np.float32),
                             MNN.Tensor_DimensionType_Caffe,)

        self.r1i.copyFrom(tmp_r1i)
        self.r2i.copyFrom(tmp_r2i)
        self.r3i.copyFrom(tmp_r3i)
        self.r4i.copyFrom(tmp_r4i)

    def img_inference(self, frame):
        post_out = self.predict(frame, mean=0, std=1.)
        alpha = self._plot(frame, post_out)
        return alpha

    def cam_inference(self, video=0, dst_path=None):
        cap = cv2.VideoCapture(video)
        count = 0
        if dst_path:
            os.makedirs(dst_path, exist_ok=True)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            alpha = self.img_inference(frame)
            # cv2.imwrite(os.path.join(dst_path, '%04d.png' % count), alpha, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            count += 1
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
            print('avg cost: {:.2f}ms'.format(np.mean(self.cost)*1000))
        cv2.destroyAllWindows()

    def _postprocess(self, out):
        fgr = self._get_mnn_output(out['fgr'])
        pha = self._get_mnn_output(out['pha'])
        self.r1i.copyFrom(out['r1o'])
        self.r2i.copyFrom(out['r2o'])
        self.r3i.copyFrom(out['r3o'])
        self.r4i.copyFrom(out['r4o'])

        fgr = fgr.squeeze(0).transpose((1, 2, 0))[:, :, ::-1]
        pha = pha.squeeze(0).squeeze(0)
        return (fgr, pha)

    def _plot(self, frame, post_out, show_image=False):
        '''
        0: 'background',
        1: 'foreground',·
        '''
        fgr, alpha = self._recover_mask(frame, post_out)

        if show_image:
            fgr = cv2.resize(fgr*255., (frame.shape[1], frame.shape[0])).astype(np.uint8)
            alpha = cv2.resize(alpha, (frame.shape[1], frame.shape[0])).astype(np.float32)
            alpha = cv2.merge((alpha, alpha, alpha))
            fake_img = (frame[:, :, ::-1] * alpha + self.bgr * (1-alpha)).astype(np.uint8)
            cv2.imshow('img_with_bgr', fake_img)
            cv2.imshow('alpha', alpha)
        return alpha*255.

    def _recover_mask(self, frame, post_out):
        '''
        mask remove zero padding
        '''
        h, w, _ = frame.shape
        fgr, alpha = post_out
        alpha = cv2.resize(alpha, (max(h, w), max(h, w)), interpolation=cv2.INTER_LINEAR)
        fgr = cv2.resize(fgr, (max(h, w), max(h, w)), interpolation=cv2.INTER_LINEAR)
        ll = (h - w) // 2
        if ll > 0:
            return fgr[:, ll:-ll], alpha[:, ll:-ll]
        elif ll < 0:
            ll = -ll
            return fgr[ll:-ll, :], alpha[ll:-ll, :]
        else:
            return fgr, alpha
