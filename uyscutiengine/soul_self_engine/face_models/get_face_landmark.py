import os
from face_detector_blazeface import BlazeFace
# from face_direction_classify import FaceDirection
from face_pose_points import FaceDirection
from face_landmark_pfld import FaceLandmark
from mouth_landmark_v1 import MouthLandmark
from eye_landmark_v1 import EyeLandmark
import cv2
import argparse
import shutil
import numpy as np

import sys 
sys.path.append("../")
from utils.image_process import crop_box_img, draw_direction, draw_boxes, draw_landmark, get_base_box

left_eye_index = [43, 48, 49, 51, 50, 46, 47, 45, 44, 35, 41, 40, 42, 39, 75, 37, 33, 36, 38, 34]
right_eye_index = [101, 105, 104, 103, 102, 97, 98, 99, 100, 93, 96, 94, 95, 89, 81, 90, 87, 91, 92, 88]
mouth_index = [52, 64, 63, 71, 67, 68, 61, 58, 59, 53, 56, 55, 65, 66, 62, 70, 69, 57, 60, 54]

class FaceLandmarkAll(object):
    def __init__(self):
        super(FaceLandmarkAll, self).__init__()
        self.face_detector = BlazeFace()
        self.face_direction = FaceDirection()
        self.face_landmark = FaceLandmark()
        self.mouth_landmark = MouthLandmark()
        self.eye_landmark = EyeLandmark()

    def get_all_landmark(self, img):
        ori_img = img.copy()
        face_boxes = self.face_detector.get_face_boxes(img)
        if len(face_boxes) < 1:
            return []
        output = []
        for box in face_boxes:
            crop_img, crop_box = crop_box_img(img, box, len_scale = 1.05, use_dia = True)
            crop_img_direciton, _ = crop_box_img(img, box, len_scale = 1.1, use_dia = False)
            direciton = self.face_direction.get_face_direction(crop_img_direciton)
            if int(direciton) != 0:
                crop_img = rot_input_image(crop_img, direciton)
            landmark = self.face_landmark.get_face_landmark(crop_img)
            landmark = find_to_ori_image(landmark, direciton, crop_img, crop_box)
            landmark = update_by_small_model(landmark, ori_img, self.mouth_landmark, self.eye_landmark, direciton)
            tmp_dict = {}
            tmp_dict["box"] = box
            tmp_dict["landmark"] = landmark
            tmp_dict["direction"] = direciton
            output.append(tmp_dict)
        return output


def parse_args():
    parser = argparse.ArgumentParser(description='face_detector')
    parser.add_argument('--input_dir', default="../../face++/test_imgs/face", type=str, help="input dir for images")
    parser.add_argument('--save_dir', default="./test_output/face_landmark", type=str, help="output_dir to save datas")

    global args
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)


def rot_input_image(image, rot_angle):
    if rot_angle == 90:
        image = np.rot90(image, k = 3)
    if rot_angle == 180:
        image = np.rot90(image, k = 2)
    if rot_angle == 270:
        image = np.rot90(image, k = 1)
    return image

def find_to_ori_image(landmarks,rot_angle, crop_img, crop_base_box):
    landmarks = landmarks.reshape((-1, 2))

    if rot_angle == 90:
        for tmp in landmarks:
            tmp[0], tmp[1] = tmp[1], crop_img.shape[1] - 1 - tmp[0]
    if rot_angle == 270:
        for tmp in landmarks:
            tmp[0], tmp[1] = crop_img.shape[0] - 1 - tmp[1], tmp[0]

    if rot_angle == 180:
        for tmp in landmarks:
            tmp[0], tmp[1] = crop_img.shape[1] - 1 - tmp[0], crop_img.shape[0] - 1 - tmp[1]

    for tmp in landmarks:
        tmp[0] = tmp[0] + crop_base_box[0]
        tmp[1] = tmp[1] + crop_base_box[1]
    
    return landmarks

def get_small_model_result(landmark, area_index, area_model, direciton, draw_img, scale, flip=False):
    area_index = np.array(area_index, dtype=int)
    area_landmark = landmark[area_index, :]
    base_box = get_base_box(area_landmark)
    crop_img, crop_box = crop_box_img(draw_img, base_box, len_scale=scale)
    if int(direciton) != 0:
        crop_img = rot_input_image(crop_img, direciton)

    if flip:
        area_landmark = area_model.get_landmark(crop_img, True)
    else:
        area_landmark = area_model.get_landmark(crop_img)

    area_landmark = find_to_ori_image(area_landmark, direciton, crop_img, crop_box)
    landmark[area_index, :] = area_landmark
    return landmark


def update_by_small_model(landmark, draw_img, mouth_model, eye_model, direciton):
    landmark = get_small_model_result(landmark, mouth_index, mouth_model, direciton, draw_img, 1.5)
    landmark = get_small_model_result(landmark, left_eye_index, eye_model, direciton, draw_img, 1.2)
    landmark = get_small_model_result(landmark, right_eye_index, eye_model, direciton, draw_img, 1.2, True)
    return landmark

def main():
    parse_args()
    face_landmark_all = FaceLandmarkAll()

    for file in os.listdir(args.input_dir):
        if not file.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(args.input_dir, file)
        draw_path = os.path.join(args.save_dir, file)
        img = cv2.imread(img_path)
        draw_img = img.copy()
        output = face_landmark_all.get_all_landmark(img)
        
        face_boxes = []
        for tmp_dict in output:
            box = tmp_dict["box"]
            face_boxes.append(box)
            landmark = tmp_dict["landmark"]
            direciton = tmp_dict["direction"]
            draw_img = draw_direction(draw_img, direciton, box)
            draw_img = draw_landmark(draw_img, landmark)
        draw_img = draw_boxes(draw_img, face_boxes)
        cv2.imwrite(draw_path, draw_img)

if __name__ == "__main__":
    main()