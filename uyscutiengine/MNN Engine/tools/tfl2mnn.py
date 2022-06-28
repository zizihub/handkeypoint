import tflite2onnx
import os
import argparse
import os.path as osp


def main():
    # argumentsParse
    parser = argparse.ArgumentParser(description="pytorch2onnx")
    parser.add_argument(
        "--tfl_model",
        default="../model/tflite/hand_landmark_full.tflite")
    parser.add_argument("--fp16", default=False, action="store_true", help="mnn float-point 16")
    args = parser.parse_args()
    onnx_model_path = "../models/onnx"
    mnn_model_path = "../models/mnn"
    os.makedirs(onnx_model_path, exist_ok=True)
    os.makedirs(mnn_model_path, exist_ok=True)
    tflite_path = args.tfl_model
    model_name = osp.splitext(osp.basename(args.tfl_model))[0]
    onnx_path = os.path.join(onnx_model_path, '{}.onnx'.format(model_name))
    mnn_path = os.path.join(mnn_model_path, '{}.mnn'.format(model_name))
    if args.fp16:
        fp16 = '--fp16'
    else:
        fp16 = ''

    # tflite to onnx
    tflite2onnx.convert(tflite_path, onnx_path)

    # # mnn simplified
    os.system(f'python -m onnxsim {onnx_path} {onnx_path}')

    # onnx to mnn
    os.system(f'mnnconvert -f ONNX --modelFile {onnx_path} --MNNModel {mnn_path} --bizCode biz {fp16}')


if __name__ == '__main__':
    main()
