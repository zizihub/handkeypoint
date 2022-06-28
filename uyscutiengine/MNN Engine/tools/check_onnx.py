import os
import sys
import onnxruntime
import onnx
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
sys.path.append('../../../classification-engine')  # NOQA: 402
sys.path.append('../../../segmentation-engine')  # NOQA: 402
sys.path.append('../../../regression-engine')  # NOQA: 402
sys.path.append('../../../pose-engine')  # NOQA: 402
sys.path.append('..')   # NOQA: 402
import cls_engine
import seg_engine
import reg_engine
import pose_engine
from src.mnn_landmark import MNNLandmarkRegressor


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


class ONNXModel():
    def __init__(self, onnx_path):
        """
        :param onnx_path:
        """
        self.onnx_session = onnxruntime.InferenceSession(onnx_path)
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)
        print("input_name:{}".format(self.input_name))
        print("output_name:{}".format(self.output_name))

    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, image_numpy):
        """
        input_feed={self.input_name: image_numpy}
        :param input_name:
        :param image_numpy:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_numpy
        return input_feed

    def forward(self, image_numpy):
        '''
        # image_numpy = image.transpose(2, 0, 1)
        # image_numpy = image_numpy[np.newaxis, :]
        # onnx_session.run([output_name], {input_name: x})
        # :param image_numpy:
        # :return:
        '''
        # 输入数据的类型必须与模型一致,以下三种写法都是可以的
        # scores, boxes = self.onnx_session.run(None, {self.input_name: image_numpy})
        # scores, boxes = self.onnx_session.run(self.output_name, input_feed={self.input_name: iimage_numpy})
        input_feed = self.get_input_feed(self.input_name, image_numpy)
        ouputs = self.onnx_session.run(self.output_name, input_feed=input_feed)
        return ouputs


if __name__ == '__main__':
    frame = cv2.cvtColor(cv2.imread('../datasets/left_hand.png'), cv2.COLOR_BGR2RGB)
    image = transforms.functional.normalize(transforms.functional.to_tensor(frame), mean=[
                                            0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).unsqueeze(0)
    # torch
    device = "cpu"
    model = torch.load('../checkpoints/mbv3-mix-0321.pt', map_location=device)
    model.eval()
    outputs_torch = model.forward(image)
    print(outputs_torch)
    # onnx
    onnxmodel = ONNXModel('../models/onnx/mbv3-mix-0321_sim.onnx')
    image_numpy = to_numpy(image)
    outputs_onnx = onnxmodel.forward(image_numpy)
    print(outputs_onnx)
    # mnn
    mnnmodel = MNNLandmarkRegressor(mnn_path='../models/mnn/mbv3-mix-0321.mnn')
    outputs_mnn = mnnmodel._mnn_inference(image_numpy)
    print(mnnmodel._get_mnn_output(outputs_mnn[0]['joints_3d']), mnnmodel._get_mnn_output(outputs_mnn[0]['hand_type']))
