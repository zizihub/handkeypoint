import numpy as np
import MNN
import cv2
import argparse
import os

import sys 
sys.path.append("../")

from utils.image_process import pad_and_resize, draw_direction

class FaceDirection(object):
    def __init__(self):
        super(FaceDirection, self).__init__()
        model_path = "./models/face_pose.mnn"
        self.input_size = 64
        self.class_num = 10
        self.interpreter = MNN.Interpreter(model_path)
        self.class_dict = {0 : 0, 1 : 0, 2: 90, 3 : 90, 4 : 180, 5 : 180, 6 : 180, 7 : 270, 8 : 270, 9 : 0}
        self.session = self.interpreter.createSession()
        self.input_tensor = self.interpreter.getSessionInput(self.session)

         
    def preprocess(self, input_image):
        image = pad_and_resize(input_image, [self.input_size, self.input_size], padding= True)
        tmp_input = MNN.Tensor((1, 3, self.input_size, self.input_size), MNN.Halide_Type_Float, image, MNN.Tensor_DimensionType_Caffe)
        return tmp_input


    def get_model_output(self):
        tensor_scores = self.interpreter.getSessionOutput(self.session)
        scores_output = MNN.Tensor((1, self.class_num, 1), MNN.Halide_Type_Float, np.ones([1, self.class_num, 1]).astype(np.float32), MNN.Tensor_DimensionType_Caffe)
        tensor_scores.copyToHostTensor(scores_output)
        scores_output = np.array(scores_output.getData()).flatten()
        
        return scores_output

    def post_process(self, scores):
        pre_order = np.argsort(-scores)[0]
        out_angle = self.class_dict[pre_order]
        return out_angle

    def get_face_direction(self, input_image):
        model_input = self.preprocess(input_image)
        self.input_tensor.copyFrom(model_input)
        self.interpreter.runSession(self.session)
        scores = self.get_model_output()
        direciton = self.post_process(scores)
        return direciton



def parse_args():
    parser = argparse.ArgumentParser(description='face_direction')
    parser.add_argument('--input_dir', default="../../face++/test_imgs/face", type=str, help="input dir for images")
    parser.add_argument('--save_dir', default="./test_output/direction", type=str, help="output_dir to save datas")

    global args
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)


def main():
    parse_args()
    face_direction = FaceDirection()
    # img = cv2.imread("./3.jpeg")
    # face_boxes = face_detector.get_face_boxes(img)
    # print(face_boxes)

    for file in os.listdir(args.input_dir):
        if not file.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        img_path = os.path.join(args.input_dir, file)
        draw_path = os.path.join(args.save_dir, file)
        img = cv2.imread(img_path)
        draw_img = img.copy()
        direciton = face_direction.get_face_direction(img)
        # print(face_boxes)
        draw_img = draw_direction(draw_img, direciton)
        cv2.imwrite(draw_path, draw_img)


if __name__ == "__main__":
    main()