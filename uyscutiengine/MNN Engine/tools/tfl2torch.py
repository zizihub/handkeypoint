from tflite import Model
import numpy as np
import sys
sys.path.append('../models')  # NOQA: 402
from hair_seg_net import HairSegNet
import torch
from collections import OrderedDict


class TFLite2Torch:
    def __init__(self, model_path='../checkpoints/hair_segmentation.tflite'):
        self.model_path = model_path

    def check_tfl(self):
        data = open(self.model_path, "rb").read()
        self.model = Model.GetRootAsModel(data, 0)
        self.subgraph = self.model.Subgraphs(0)
        self.subgraph.Name()
        for i in range(0, self.subgraph.TensorsLength()):
            tensor = self.subgraph.Tensors(i)
            print("%3d %30s %d %2d %s" % (i, tensor.Name(), tensor.Type(), tensor.Buffer(),
                                          self._get_shape(self.subgraph.Tensors(i))))

    def get_weights(self):
        self.tensor_dict = {(self.subgraph.Tensors(i).Name().decode("utf8")): i
                            for i in range(self.subgraph.TensorsLength())}
        parameters = self._get_parameters(self.subgraph)
        print('length of parameters', len(parameters))

        W = self._get_weights("conv2d/Kernel")
        b = self._get_weights("conv2d/Bias")
        print('Weights.shape: %s |  Bias.shape: %s' % (W.shape, b.shape))

    def get_probable_names(self):
        probable_names = []
        for i in range(0, self.subgraph.TensorsLength()):
            tensor = self.subgraph.Tensors(i)
            if tensor.Buffer() > 0 and (tensor.Type() == 0 or tensor.Type() == 1):
                probable_names.append(tensor.Name().decode("utf-8"))
        return probable_names

    def build_state_dict(self, net, convert):
        """
        Copy the weights into the layers.

        Note that the ordering of the weights is different between PyTorch and TFLite, so we need to transpose them.

        Convolution weights:

        TFLite:  (out_channels, kernel_height, kernel_width, in_channels)
        PyTorch: (out_channels, in_channels, kernel_height, kernel_width)

        Depthwise convolution weights:

        TFLite:  (1, kernel_height, kernel_width, channels)
        PyTorch: (channels, 1, kernel_height, kernel_width)
        """
        new_state_dict = OrderedDict()

        for dst, src in convert.items():
            W = self._get_weights(src)
            print("%-50s %-30s %-20s %s" % (dst, src, W.shape, net.state_dict()[dst].shape))

            if W.ndim == 4:
                if W.shape[0] == 1 or 'conv_trans' in dst:
                    W = W.transpose((3, 0, 1, 2))  # depthwise conv or ConvTranspose2D
                else:
                    W = W.transpose((0, 3, 1, 2))  # regular conv
            elif W.ndim == 3:
                if 'prelu' in dst or 'stem' in dst:
                    W = W.squeeze()                # prelu
            new_state_dict[dst] = torch.from_numpy(W)
        return new_state_dict

    @staticmethod
    def get_convert(net, probable_names):
        convert = {}
        i = 0
        assert len(net.state_dict().items()) == len(
            probable_names), f"net length {len(net.state_dict().items())} doesn't match with probable_name length {len(probable_names)}"
        for name, params in net.state_dict().items():
            print(name, probable_names[i])
            convert[name] = probable_names[i]
            i += 1
        return convert

    @staticmethod
    def _get_shape(tensor):
        return [tensor.Shape(i) for i in range(tensor.ShapeLength())]

    @staticmethod
    def _get_parameters(graph):
        parameters = {}
        for i in range(graph.TensorsLength()):
            tensor = graph.Tensors(i)
            if tensor.Buffer() > 0:
                name = tensor.Name().decode("utf8")
                parameters[name] = tensor.Buffer()
        return parameters

    def _get_weights(self, tensor_name):
        i = self.tensor_dict[tensor_name]
        tensor = self.subgraph.Tensors(i)
        buffer = tensor.Buffer()
        shape = self._get_shape(tensor)
        assert(tensor.Type() == 0 or tensor.Type() == 1)  # FLOAT32

        W = self.model.Buffers(buffer).DataAsNumpy()
        if tensor.Type() == 0:
            W = W.view(dtype=np.float32)
        elif tensor.Type() == 1:
            W = W.view(dtype=np.float16)
        W = W.reshape(shape)
        return W


if __name__ == "__main__":
    tool = TFLite2Torch()
    tool.check_tfl()
    tool.get_weights()
    prob_name = tool.get_probable_names()
    net = HairSegNet()
    convert = tool.get_convert(net, prob_name)
    static = tool.build_state_dict(net, convert)
    net.load_state_dict(static, strict=True)

    x = torch.randn((1, 4, 512, 512))
    out = net(x)
    print(out.shape)

    torch.save(static, '../checkpoints/hair_segmentation.pth')
    