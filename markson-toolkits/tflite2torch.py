import numpy as np
from collections import OrderedDict
from tflite import Model
import torch
import sys
sys.path.append('../HandPoseEstimation/BlazeFace-PyTorch')
from blazepalm import BlazePalm  # noqa: E402


class TFLiteConverter(object):
    def __init__(self):
        super(TFLiteConverter, self).__init__()

    def _check_env(self):
        print("PyTorch version:", torch.__version__)
        print("CUDA version:", torch.version.cuda)
        print("cuDNN version:", torch.backends.cudnn.version())

    def get_device(self):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def load_model(self, tfl_model_path):
        model_tfl_data = open(tfl_model_path, "rb").read()
        self.model = Model.GetRootAsModel(model_tfl_data, 0)
        self.subgraph = self.model.Subgraphs(0)
        print(self.subgraph.Name())
        self._print_graph(self.subgraph)

    def convert(self, torch_model_path):
        self.tensor_dict = {(self.subgraph.Tensors(i).Name().decode("utf8")): i for i in range(self.subgraph.TensorsLength())}
        front_parameters = self._get_parameters(self.subgraph)
        print(len(front_parameters))
        net = BlazePalm(large=False)
        probable_names = self._get_probable_names(self.subgraph)
        print(probable_names[:5])
        convert_net = self._get_convert(net, probable_names)
        state_dict = self._build_state_dict(self.model, self.subgraph, self.tensor_dict, net, convert_net)
        net.load_state_dict(state_dict, strict=True)
        torch.save(net.state_dict(), torch_model_path)
        print('torch model save: {}'.format(torch_model_path))

    def _get_shape(self, tensor):
        return [tensor.Shape(i) for i in range(tensor.ShapeLength())]

    def _print_graph(self, graph):
        for i in range(0, graph.TensorsLength()):
            tensor = graph.Tensors(i)
            print("%3d %30s %d %2d %s" % (i, tensor.Name(), tensor.Type(), tensor.Buffer(),
                                          self._get_shape(graph.Tensors(i))))

    def _get_parameters(self, graph):
        parameters = {}
        for i in range(graph.TensorsLength()):
            tensor = graph.Tensors(i)
            if tensor.Buffer() > 0:
                name = tensor.Name().decode("utf8")
                parameters[name] = tensor.Buffer()
        return parameters

    def _get_weights(self, model, graph, tensor_dict, tensor_name):
        i = tensor_dict[tensor_name]
        tensor = graph.Tensors(i)
        buffer = tensor.Buffer()
        shape = self._get_shape(tensor)
        assert(tensor.Type() == 0 or tensor.Type() == 1)  # FLOAT32

        W = model.Buffers(buffer).DataAsNumpy()
        if tensor.Type() == 0:
            W = W.view(dtype=np.float32)
        elif tensor.Type() == 1:
            W = W.view(dtype=np.float16)
        W = W.reshape(shape)
        return W

    def _get_probable_names(self, graph):
        probable_names = []
        for i in range(0, graph.TensorsLength()):
            tensor = graph.Tensors(i)
            if tensor.Buffer() > 0 and (tensor.Type() == 0 or tensor.Type() == 1):
                probable_names.append(tensor.Name().decode("utf-8"))
        return probable_names

    def _get_convert(self, net, probable_names):
        convert = {}
        i = 0
        for name, params in net.state_dict().items():
            convert[name] = probable_names[i]
            i += 1
        return convert

    def _build_state_dict(self, model, graph, tensor_dict, net, convert):
        new_state_dict = OrderedDict()

        for dst, src in convert.items():
            W = self._get_weights(model, graph, tensor_dict, src)

            if 'p_re_lu' in src:
                W = W.squeeze()

            print('%30s: %-20s | %40s: %-5s' % (src, W.shape, dst, net.state_dict()[dst].shape))

            if W.ndim == 4:
                if W.shape[0] == 1:
                    W = W.transpose((3, 0, 1, 2))  # depthwise conv
                else:
                    W = W.transpose((0, 3, 1, 2))  # regular conv

            new_state_dict[dst] = torch.from_numpy(W)
        return new_state_dict


if __name__ == '__main__':
    tlc = TFLiteConverter()
    tlc.load_model('../TFLiteDetection/palm_detection.tflite')
    tlc.convert('../HandPoseEstimation/blazepalm.pth')
