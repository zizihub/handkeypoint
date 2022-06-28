# -*- coding: utf-8 -*-
import time
import os
import cv2
import requests
import json
import argparse
from util_facepp import file_name, get_base_name

def get_pose_attribute(pose_attribute):
    pitch = pose_attribute["pitch_angle"]
    roll = pose_attribute["roll_angle"]
    yaw = pose_attribute["yaw_angle"]
    pitch_flag = False
    yaw_flag = False
    if abs(pitch) > 30:
        pitch_flag = True
    if abs(yaw) > 30:
        yaw_flag = True
    return pitch_flag, yaw_flag, [pitch, yaw, roll]

def get_single_eye_status(single_eye):
    eye_open_flag = False
    eye_close_flag = False
    if single_eye["no_glass_eye_open"] > 85 or single_eye["normal_glass_eye_open"] > 85:
        eye_open_flag = True
    if single_eye["no_glass_eye_close"] > 85 or single_eye["normal_glass_eye_close"] > 85:
        eye_close_flag = True
    return eye_close_flag, eye_open_flag


def get_eye_attribute(eye_attribute):
    left_eye = eye_attribute["left_eye_status"]
    right_eye = eye_attribute["right_eye_status"]
    left_close_flag, left_open_flag = get_single_eye_status(left_eye)
    right_close_flag, right_open_flag = get_single_eye_status(right_eye)
    return left_close_flag or right_close_flag, left_open_flag and right_open_flag


def get_mouth_attribute(mouth_attribute):
    mouth_open_flag = False
    mouth_close_flag = False
    if mouth_attribute["close"] > 85:
        mouth_close_flag = True
    if mouth_attribute["open"] > 85:
        mouth_open_flag = True
    return mouth_open_flag, mouth_close_flag


def get_emotion_attribute(emotion_attribute):
    anger_flag = False
    surprise_flag = False
    if emotion_attribute["anger"] > 85:
        anger_flag = True
    if emotion_attribute["surprise"] > 85:
        surprise_flag = True
    return anger_flag, surprise_flag


def write_data_file(save_path, name, data_list):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    txt_name = name + ".txt"
    txt_name = os.path.join(save_path, txt_name)
    out_file = open(txt_name, "w")
    for data in data_list:
        out_file.write(data[0])
        for tmp in data[1]:
            out_file.write(" ")
            out_file.write(str(tmp))
        for tmp in data[2]:
            out_file.write(" ")
            out_file.write(str(tmp))
        for tmp in data[3]:
            out_file.write(" ")
            out_file.write(str(tmp))
        out_file.write("\n")


def visable_label(img_draw, save_path, out_name, base_name):
    # for i in range(len(flag_list)):
    #     if flag_list[i]:
    #         path = os.path.join(save_path, out_name[i])
    #         save_name = os.path.join(path, base_name)
    #         cv2.imwrite(save_name, img_draw)
    path = os.path.join(save_path, out_name)
    save_name = os.path.join(path, base_name)
    cv2.imwrite(save_name, img_draw)


def getFacailLandmarksFromFacepp():
    requests.adapters.DEFAULT_RETRIES = 5
    url = 'https://api-cn.faceplusplus.com/facepp/v3/detect'
    input_path = args.input_dir 
    save_path = args.save_dir
    L = file_name(input_path)
    L.sort(key=len)
    out_name = args.attribute_class
    # ["big_pitch", "big_yaw", "eye_close", "eye_open", "mouth_open", "mouth_close", "anger", "surprise", "all"]
    out_list = [[]] * len(out_name)
    for tmp in out_name:
        tmp_path = os.path.join(save_path, tmp)
        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)

    for index, image_name in enumerate(L):
        # if index < 35000:
        #     continue
        image_path = image_name
        img_file = {'image_file': open(image_path, 'rb')}
        payload = {
                   # 'api_key':'FFIU24spbxVMdy2IZHuCQZP_a-zcqsZE',
                   # 'api_secret':'jDZY475CNLNOwJXNsPiuSp8XXOBRQmbQ',
                   'api_key': '02jqumPOP3yW8ajEJ9sNPOoTcSywiNja',
                   'api_secret': '7SNWpDy1WZBz-lWO-ZpU-vqF0mSgfZud',
                    'return_landmark': 2,
                    'return_attributes': "headpose,emotion,eyestatus,mouthstatus,facequality"}
        s = requests.session()
        s.keep_alive = False
        r = requests.post(url, files=img_file, data=payload)
        if r:
            data = json.loads(r.text)
            if args.visible_flag:
                image = cv2.imread(image_path)
                img_draw = image.copy()
            
            base_name = get_base_name(image_path)

            if 'faces' not in data.keys() or len(data['faces']) == 0:
                print("no face in ", image_path)
            else:
                # assert len(data['faces']) == 1
                for i in range(len(data['faces'])):
                    face = data['faces'][i]
                    width = face['face_rectangle']['width']
                    top = face['face_rectangle']['top']
                    height = face['face_rectangle']['height']
                    left = face['face_rectangle']['left']
                    box = [left, top, width, height]
                    landmark = []
                    for j in face['landmark']:
                        point = face['landmark'][j]
                        x = point['x']
                        y = point['y']
                        landmark.append(x)
                        landmark.append(y)
                        if args.visible_flag:
                            cv2.circle(img_draw, (x, y), 2, (0, 255, 0), -1)
                    if args.visible_flag:
                        cv2.rectangle(img_draw, (left, top),(left+width, top+height), (0, 255, 0), 1)
                    attribute = face['attributes']
                    flag_dict = {}
                    flag_dict["all"] = True
                    flag_dict["big_pitch"], flag_dict["big_yaw"], angles = get_pose_attribute(attribute['headpose'])
                    flag_dict["eye_close"], flag_dict["eye_open"] = get_eye_attribute(attribute['eyestatus'])
                    flag_dict["mouth_open"], flag_dict["mouth_close"] = get_mouth_attribute(attribute['mouthstatus'])
                    flag_dict["anger"], flag_dict["surprise"] = get_emotion_attribute(attribute['emotion'])
 
                    data_inf = [image_path, box, landmark, angles]

                    for index, tmp_attribute in enumerate(out_name):
                        if flag_dict[tmp_attribute]:
                            out_list[index].append(data_inf)

                    if args.visible_flag:
                        visable_label(img_draw, save_path, "all", base_name)

        if (index+1) % 100 == 0 or (index+1) == len(L):
            print('%d done', index+1)

        if (index+1) % 5000 == 0:
            write_data_file(save_path, out_name[i] + "_" + str(index + 1), out_list[i])
            out_list = [[]] * len(out_name)
            time.sleep(10)

    for i in range(len(out_name)):
        write_data_file(save_path, out_name[i] + "_" + "last", out_list[i])

def main():
    parse_args()
    getFacailLandmarksFromFacepp()

def parse_args():
    parser = argparse.ArgumentParser(description='facepp_face_landmark')
    parser.add_argument('--input_dir', default="./test_imgs/face", type=str, help="input dir for images")
    parser.add_argument('--save_dir', default="./test_output/face", type=str, help="output_dir to save datas")
    parser.add_argument('--attribute_class', default=["big_yaw", "all"], type=list, \
            help="different Attributions to save, choices include: \
            'big_pitch', 'big_yaw', 'eye_close', 'eye_open', 'mouth_open', \
            'mouth_close', 'anger', 'surprise', 'all'")
    parser.add_argument('--visible_flag', default=False, type=bool, help="save visible imgs")

    global args
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)


if __name__ == '__main__':
    main()