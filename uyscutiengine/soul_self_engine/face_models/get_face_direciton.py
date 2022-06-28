import os
# from soul_self_engine.face_models import face_direction_classify 
from face_detector_blazeface import BlazeFace
from face_direction_classify import FaceDirection
import cv2
import argparse
import shutil

import sys 
sys.path.append("../")
from utils.image_process import crop_box_img, draw_direction, draw_boxes

def parse_args():
    parser = argparse.ArgumentParser(description='face_detector')
    # parser.add_argument('--input_dir', default="../../face++/test_imgs/face", type=str, help="input dir for images")
    parser.add_argument('--input_dir', default="/Volumes/Elements/data/face_landmark_106/longmao_face/soul人脸106点数据-20201116/images", type=str, help="input dir for images")
    parser.add_argument('--input_dir_list', default=["/Volumes/Elements/data/face_landmark_106/longmao_face/soul人脸106点数据-20201116/images",\
                                                     "/Volumes/Elements/data/face_landmark_106/longmao_face/soul人脸106点数据-20201118/images",\
                                                     "/Volumes/Elements/data/face_landmark_106/longmao_face/soul人脸106点数据-20201125/images",\
                                                     "/Volumes/Elements/data/face_landmark_106/longmao_face/soul人脸106点数据-20201204/images",\
                                                     "/Volumes/Elements/data/face_landmark_106/longmao_face/soul人脸106点数据-20201207/images",\
                                                     "/Volumes/Elements/data/face_landmark_106/longmao_face/soul人脸106点数据-20201208/images"], type=str, help="input dir for images")

    parser.add_argument('--save_dir', default="./test_output/direciton", type=str, help="output_dir to save datas")
    parser.add_argument('--save_dir2', default="./test_output/rot_shape", type=str, help="output_dir to save datas")

    global args
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.save_dir2, exist_ok=True)


def main():
    parse_args()
    face_detector = BlazeFace()
    face_direction = FaceDirection()
    dirty_file_1 = open("./dirty_img_list.txt", "w")
    dirty_file_2 = open("./rot_shape_img_list.txt", "w")
    # for file in os.listdir(args.input_dir):
    for input_dir in args.input_dir_list:
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if not file.lower().endswith((".jpg", ".png", ".jpeg")):
                    continue
                img_path = os.path.join(root, file)
                draw_path = os.path.join(args.save_dir, file)
                img = cv2.imread(img_path)
                draw_img = img.copy()
                face_boxes = face_detector.get_face_boxes(img)
                for box in face_boxes:
                    crop_img, crop_box = crop_box_img(img, box, len_scale = 1.05, use_dia = True)
                    direciton = face_direction.get_face_direction(crop_img)
                    if int(direciton) != 0:
                        shutil.copy(img_path, args.save_dir)
                        print("rot img ", img_path)
                        dirty_file_1.write(img_path)
                        dirty_file_1.write("\n")
                        continue
                    if draw_img.shape[0] < draw_img.shape[1]:
                        shutil.copy(img_path, args.save_dir2)
                        print("shape wrong ", img_path)
                        dirty_file_2.write(img_path)
                        dirty_file_2.write("\n")
                #     draw_img = draw_direction(draw_img, direciton, box)
                    
                # draw_img = draw_boxes(draw_img, face_boxes)
                # cv2.imwrite(draw_path, draw_img)
        dirty_file_1.flush()
        dirty_file_2.flush()
    dirty_file_1.close()
    dirty_file_2.close()

if __name__ == "__main__":
    main()