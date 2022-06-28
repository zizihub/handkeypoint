import tflite2onnx


def main():
    tflite_path = '../HandPoseEstimation/palm_detection.tflite'
    onnx_path = '../HandPoseEstimation/palm_detection.onnx'
    tflite2onnx.convert(tflite_path, onnx_path)


if __name__ == '__main__':
    main()
