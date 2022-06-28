import numpy as np
import MNN
import cv2
import argparse
import os

import sys 
sys.path.append("../")
# print(sys.path) 
# import utils
from utils.image_process import pad_and_resize, draw_boxes
from utils.utils import sigmoid, weighted_non_max_suppression


class BlazeFace(object):
    def __init__(self):
        super(BlazeFace, self).__init__()
        model_path = "./models/lite_blazeface.mnn"
        anchors_path = "./models/anchors.npy"
        anchors = np.load(anchors_path)
        self.anchors = anchors.flatten()
        self.output_num = 896
        self.input_size = 128
        self.score_clipping_thresh = 100.0
        self.min_score_thresh = 0.8
        self.iou_threshold = 0.4
        self.interpreter = MNN.Interpreter(model_path)
        self.session = self.interpreter.createSession()
        self.input_tensor = self.interpreter.getSessionInput(self.session)

         
    def preprocess(self, input_image):
        image = pad_and_resize(input_image, [self.input_size, self.input_size], padding= True)
        # print(image)
        tmp_input = MNN.Tensor((1, 3, self.input_size, self.input_size), MNN.Halide_Type_Float, image, MNN.Tensor_DimensionType_Caffe)
        return tmp_input


    def get_model_output(self):
        output_tensor_name0 = "scores"
        output_tensor_name1 = "boxes"

        tensor_scores = self.interpreter.getSessionOutput(self.session, output_tensor_name0)
        tensor_boxes = self.interpreter.getSessionOutput(self.session, output_tensor_name1)

        scores_output = MNN.Tensor((1, self.output_num, 1), MNN.Halide_Type_Float, np.ones([1, self.output_num, 1]).astype(np.float32), MNN.Tensor_DimensionType_Caffe)
        tensor_scores.copyToHostTensor(scores_output)
        boxes_output = MNN.Tensor((1, self.output_num, 16), MNN.Halide_Type_Float, np.ones([1, self.output_num, 16]).astype(np.float32), MNN.Tensor_DimensionType_Caffe)
        tensor_boxes.copyToHostTensor(boxes_output)
        scores_output = np.array(scores_output.getData()).flatten()
        boxes_output = np.array(boxes_output.getData()).flatten()

        return scores_output, boxes_output

    def take_score(self, elem):
        return elem[-1]

    def select_box(self, face_boxes, face_scores):
        select_boxes = []
        for i in range(len(face_scores)):
            if face_scores[i] > self.min_score_thresh:
                face_box = face_boxes[i]
                face_box.append(face_scores[i])
                select_boxes.append(face_box)

        if len(select_boxes) <= 0:
            return []
        select_boxes.sort(key=self.take_score, reverse=True)
        select_boxes = weighted_non_max_suppression(select_boxes, self.iou_threshold)
        return select_boxes

    def find_ori_img_box(self, selected_boxes, input_img, rotation = 0):
        image = input_img
        if rotation != 0:
            image = np.rot90(image, k=rotation)
        new_len = max(image.shape[0], image.shape[1])
        result_box = []
        for tmp in selected_boxes:
            new_value = []
            for value in tmp:
                new_value.append(int(value * new_len))

            if new_len == image.shape[0]:
                padding_size = int((new_len - image.shape[1])/2)
                for i in range(0, len(new_value) - 1, 2):
                    new_value[i] = int(new_value[i] - padding_size)

            else:
                padding_size = int((new_len - image.shape[0]) / 2)
                for i in range(0, len(new_value) - 1, 2):
                    new_value[i + 1] = int(new_value[i + 1] - padding_size)

            result_box.append(new_value)

        return result_box


    def decode_face_box(self, boxes):
        new_boxes = []
        step = 16
        input_size = self.input_size
        anchors = self.anchors
        step_anchor = 4
        for i in range(self.output_num):
            x_c = boxes[step * i + 0] / input_size * anchors[step_anchor * i + 2] + anchors[step_anchor * i + 0]
            y_c = boxes[step * i + 1] / input_size * anchors[step_anchor * i + 3] + anchors[step_anchor * i + 1]

            w = boxes[step * i + 2] / input_size * anchors[step_anchor * i + 2]
            h = boxes[step * i + 3] / input_size * anchors[step_anchor * i + 3]

            ymin = y_c - h / 2.  # ymin
            xmin = x_c - w / 2.  # xmin
            ymax = y_c + h / 2.  # ymax
            xmax = x_c + w / 2.  # xmax

            new_box = [xmin, ymin, xmax, ymax]
            for k in range(6):
                offset = 4 + k * 2
                keypoint_x = boxes[step * i + offset] / input_size * anchors[step_anchor * i + 2] + anchors[step_anchor * i + 0]
                keypoint_y = boxes[step * i + offset + 1] / input_size * anchors[step_anchor * i + 3] + anchors[step_anchor * i + 1]
                new_box.append(keypoint_x)
                new_box.append(keypoint_y)

            new_boxes.append(new_box)
        return new_boxes


    def post_process(self, boxes, scores, input_image):
        face_boxes = self.decode_face_box(boxes)
        face_scores = np.where(scores > self.score_clipping_thresh, self.score_clipping_thresh, scores)
        face_scores = np.where(face_scores < self.score_clipping_thresh * (-1), self.score_clipping_thresh * (-1), scores)
        face_scores = sigmoid(face_scores)
        selected_boxes = self.select_box(face_boxes, face_scores)
        result_boxes = self.find_ori_img_box(selected_boxes, input_image)
        return result_boxes

    def get_face_boxes(self, input_image):
        model_input = self.preprocess(input_image)
        self.input_tensor.copyFrom(model_input)
        self.interpreter.runSession(self.session)
        scores, boxes = self.get_model_output()
        selected_boxes = self.post_process(boxes, scores, input_image)
        return selected_boxes



def parse_args():
    parser = argparse.ArgumentParser(description='face_detector')
    parser.add_argument('--input_dir', default="../../face++/test_imgs/face", type=str, help="input dir for images")
    parser.add_argument('--save_dir', default="./test_output/box", type=str, help="output_dir to save datas")

    global args
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)


def main():
    parse_args()
    face_detector = BlazeFace()
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
        face_boxes = face_detector.get_face_boxes(img)
        # print(face_boxes)
        draw_img = draw_boxes(draw_img, face_boxes)
        cv2.imwrite(draw_path, draw_img)


if __name__ == "__main__":
    main()