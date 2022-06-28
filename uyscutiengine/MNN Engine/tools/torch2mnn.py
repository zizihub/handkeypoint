import torch
import argparse
import sys
import os
import os.path as osp
import onnx
sys.path.append('../../../classification-engine')  # NOQA: 402
sys.path.append('../../../segmentation-engine')  # NOQA: 402
sys.path.append('../../../regression-engine')  # NOQA: 402
sys.path.append('../../../pose-engine')  # NOQA: 402
import cls_engine
import seg_engine
import reg_engine
import pose_engine


def convert(model, args):
    # start process
    model_name = osp.splitext(osp.basename(args.torch_model))[0]
    input_size = tuple(args.size)
    print("=====> initialize configuration...")
    print("Model Base Name: {}".format(model_name))
    onnx_model_path = "../models/onnx"
    mnn_model_path = "../models/mnn"
    os.makedirs(onnx_model_path, exist_ok=True)
    os.makedirs(mnn_model_path, exist_ok=True)

    onnx_model = osp.join(onnx_model_path, model_name+"_sim.onnx")
    print("Model ONNX Name: {}".format(onnx_model))
    if args.fp16:
        fp16 = "--fp16 fp16"
        mnn_model = osp.join(mnn_model_path, model_name+"_fp16.mnn")
        mnn_quant_model = osp.join(mnn_model_path, model_name+"_quant_fp16.mnn")
    else:
        fp16 = ""
        mnn_model = osp.join(mnn_model_path, model_name+".mnn")
        mnn_quant_model = osp.join(mnn_model_path, model_name+"_quant.mnn")
    print("Model MNN Name: {}".format(mnn_model))
    print("Model MNN Quant Name: {}".format(mnn_quant_model))
    print("Model Input Shape: {}".format(input_size))
    print("=====> load pytorch checkpoint...")

    print("=====> convert pytorch model to onnx...")
    dummy_input = torch.randn(input_size)
    output = model.eval()(dummy_input)
    if isinstance(output, dict):
        output_names = list(output.keys())
    else:
        output_names = ["output"]
    torch.onnx.export(
        model,
        dummy_input,
        onnx_model,
        verbose=True,
        input_names=["input"],
        output_names=output_names,
        opset_version=11,)       # interpolation issue refer to:https://github.com/onnx/tutorials/issues/137

    if 1:
        print("=====> check onnx model...")
        model = onnx.load(onnx_model)
        onnx.checker.check_model(model)
        os.system("python -m onnxsim {} {}".format(onnx_model, onnx_model))
        print("=====> onnx model simplify Ok!")

    os.system("mnnconvert -f ONNX --modelFile {} --MNNModel {} --bizCode biz {}".format(onnx_model, mnn_model, fp16))

    if 0:
        print("=====> mnn model quant...")
        os.system("mnnquant {} {} quant_cfg.json".format(mnn_model, mnn_quant_model))

    print("=====> MNN model Ok!")


def from_pt(args):
    # load model
    device = "cpu"
    model = torch.load(args.torch_model, map_location=device)
    model.eval()
    print(type(model))
    convert(model, args)


def from_pth(args):
    model = HairSegNet()
    static = torch.load(args.torch_model, map_location='cpu')
    model.load_state_dict(static, strict=True)
    model.eval()
    convert(model, args)


if __name__ == "__main__":
    # argumentsParse
    parser = argparse.ArgumentParser(description="pytorch2onnx")
    parser.add_argument(
        "--torch_model",
        default="../checkpoints/repvgg-a0-hgr.pt")
    parser.add_argument("--size", nargs="+", type=int, help="input size")
    parser.add_argument("--fp16", default=False, action="store_true", help="mnn float-point 16")
    args = parser.parse_args()
    from_pt(args)
    # from_pth(args)
