import os
import sys
sys.path.append('..')  # noqa: E402
from reg_engine.config import get_cfg, get_outname
from reg_engine.models import EfficientNet
from reg_engine import build_model
from PIL import Image
from torchvision import transforms
import torch


def setup(checkpoint=''):
    '''
    Create configs and perform basic setups.
    '''
    if checkpoint:
        my_cfg = '{}/myconfig.yaml'.format(os.path.dirname(checkpoint))
    else:
        my_cfg = '../myconfig.yaml'
    cfg = get_cfg()
    cfg.merge_from_file(my_cfg)
    output_name = get_outname(cfg)
    cfg.merge_from_list(['OUTPUT_NAME', output_name])
    cfg.merge_from_file(
        f'../log/{cfg.TASK}/{cfg.DATE}/{cfg.OUTPUT_NAME}/myconfig.yaml')
    cfg.merge_from_list(['DEPLOY', True])
    cfg.freeze()
    return cfg


def model_transform(pt_save, test_tensor, save_pt=True):
    cfg = setup()
    print('Loading >>>>>>>>>>> ', cfg.OUTPUT_NAME)
    model = build_model(cfg)
    if isinstance(model.backbone, EfficientNet):
        model.backbone.set_swish(memory_efficient=False)
    model.load_state_dict(torch.load(
        f'../log/{cfg.TASK}/{cfg.DATE}/{cfg.OUTPUT_NAME}/{cfg.OUTPUT_NAME}_best.pth', map_location='cpu')['net'])
    # ! model.eval() is essential
    model.eval()
    test_torch_model(model, test_tensor)
    traced_model = torch.jit.trace(model, test_tensor)
    if save_pt:
        traced_model.save(pt_save)
        test_jit_model(pt_save, test_tensor)
        print(pt_save)


def test_jit_model(pt_save, test_tensor):
    traced_model = torch.jit.load(pt_save, map_location='cpu')
    print('jit model output:', torch.softmax(
        traced_model.forward(test_tensor), dim=1))


def test_torch_model(model, test_tensor):
    print('torch model output:', torch.softmax(
        model.forward(test_tensor), dim=1))


def model_convert(args):
    pt_save = args.save_name
    transform = transforms.Compose(
        [transforms.Resize((128, 128)),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], inplace=True)]
    )
    test_image = Image.open(
        '/data2/zhangziwei/datasets/HGR-dataset/homemade/test/five/five-03B5m.jpg').convert('RGB')
    test_tensor = transform(test_image).unsqueeze(0)
    test_tensor = torch.cat([test_tensor]*4)
    print(test_tensor.shape)
    model_transform(pt_save, test_tensor, True)


def torch_save_pt(args):
    checkpoint = args.checkpoint
    pt_save = args.save_name
    test_tensor = torch.randn(5, 3, 128, 128)
    cfg = setup(checkpoint)
    model = build_model(cfg)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu')['net'])
    if 'RepVGG' in cfg.MODEL.NAME:
        from reg_engine.models.repvgg import repvgg_model_convert
        model = repvgg_model_convert(model)
        print('>>> RepVGG model converted...')
    model.eval()
    print('pth model output', model.forward(test_tensor))
    torch.save(model, pt_save)
    print('>>>>>> model save', pt_save)
    model_pt = torch.load(pt_save, map_location='cpu')
    print('pt model output', model_pt.forward(test_tensor))


def torch_save_onnx(args):
    checkpoint = args.checkpoint
    onnx_save = args.save_name
    test_tensor = torch.randn(1, 3, 128, 128)
    cfg = setup(checkpoint)
    model = build_model(cfg)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu')['net'])
    if 'RepVGG' in cfg.MODEL.NAME:
        from reg_engine.models.repvgg import repvgg_model_convert
        model = repvgg_model_convert(model)
        print('>>> RepVGG model converted...')
    model.eval()
    torch.onnx.export(model,
                      test_tensor,
                      onnx_save,
                      verbose=True,
                      input_names=["input"],
                      output_names=["output"],
                      opset_version=11)  # interpolation issue refer to:https://github.com/onnx/tutorials/issues/137)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="model convert")
    parser.add_argument(
        "--checkpoint",
        default="../log/HandGestureRecognition/alexnet_rw_BasicHead_large_sz128x128_kdl_0729-r50/alexnet_rw_BasicHead_large_sz128x128_kdl_0729-r50_best.pth")
    parser.add_argument(
        "--save_name",
        default="../checkpoint/alex-hgr.pt")
    args = parser.parse_args()
    torch_save_pt(args)
    # torch_save_onnx(args)
