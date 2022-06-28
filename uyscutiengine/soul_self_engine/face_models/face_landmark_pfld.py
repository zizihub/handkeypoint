import numpy as np
import MNN
import cv2
import argparse
import os

import sys 
sys.path.append("../")

from utils.image_process import pad_and_resize, draw_landmark

class FaceLandmark(object):
    def __init__(self):
        super(FaceLandmark, self).__init__()
        model_path = "./models/face_landmark.mnn"
        self.input_size = 112
        self.output_num = 106 * 2
        self.interpreter = MNN.Interpreter(model_path)
        self.session = self.interpreter.createSession()
        self.input_tensor = self.interpreter.getSessionInput(self.session)

         
    def preprocess(self, input_image):
        image = pad_and_resize(input_image, [self.input_size, self.input_size], padding= True, norm_flag = 1)
        tmp_input = MNN.Tensor((1, 3, self.input_size, self.input_size), MNN.Halide_Type_Float, image, MNN.Tensor_DimensionType_Caffe)
        return tmp_input


    def get_model_output(self):
        tensor_scores = self.interpreter.getSessionOutput(self.session)
        scores_output = MNN.Tensor((1, self.output_num, 1), MNN.Halide_Type_Float, np.ones([1, self.output_num, 1]).astype(np.float32), MNN.Tensor_DimensionType_Caffe)
        tensor_scores.copyToHostTensor(scores_output)
        scores_output = np.array(scores_output.getData()).flatten()
        
        return scores_output

    def post_process(self, model_output, input_image):
        landmarks = model_output.reshape((-1, 2))
        for i in range(landmarks.shape[0]):
            landmarks[i][0] = int(landmarks[i][0] * input_image.shape[1])
            landmarks[i][1] = int(landmarks[i][1] * input_image.shape[0])
        return landmarks


    def get_face_landmark(self, input_image):
        model_input = self.preprocess(input_image)
        self.input_tensor.copyFrom(model_input)
        self.interpreter.runSession(self.session)
        model_output = self.get_model_output()
        landmarks = self.post_process(model_output, input_image)
        return landmarks


def parse_args():
    parser = argparse.ArgumentParser(description='face_landmark')
    parser.add_argument('--input_dir', default="../../face++/test_imgs/face", type=str, help="input dir for images")
    parser.add_argument('--save_dir', default="./test_output/face_landmark", type=str, help="output_dir to save datas")

    global args
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)


def main():
    parse_args()
    face_landmark = FaceLandmark()

    for file in os.listdir(args.input_dir):
        if not file.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        img_path = os.path.join(args.input_dir, file)
        draw_path = os.path.join(args.save_dir, file)
        img = cv2.imread(img_path)
        draw_img = img.copy()
        landmarks = face_landmark.get_face_landmark(img)
        # print(face_boxes)
        draw_img = draw_landmark(draw_img, landmarks)
        cv2.imwrite(draw_path, draw_img)


if __name__ == "__main__":
    main()