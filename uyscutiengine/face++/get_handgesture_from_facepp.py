# -*- coding: utf-8 -*-
import time
import os
import cv2
import requests
import json
import numpy as np
import argparse
from util_facepp import file_name, get_base_name, pre_process_video


gesture_dict = {"unknown": 0,
                "heart_a": 1,
                "heart_b": 2,
                "heart_c": 3,
                "heart_d": 4,
                "ok": 5,
                "hand_open": 6,
                "thumb_up": 7,
                "thumb_down": 8,
                "rock": 9,
                "namaste": 10,
                "palm_up": 11,
                "fist": 12,
                "index_finger_up": 13,
                "double_finger_up": 14,
                "victory": 15,
                "big_v": 16,
                "phonecall": 17,
                "beg": 18,
                "thanks": 19}


def write_data_file(save_path, name, box_list, gesture_list):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # box_txt_name = "box_" + name + ".txt"
    # txt_name = os.path.join(save_path, box_txt_name)
    # out_file = open(txt_name, "w")
    # for data in box_list:
    #     out_file.write(data[0])
    #     boxes = data[1]
    #     for box_data in boxes:
    #         box = box_data[0]
    #         gesture_flag = box_data[1]
    #         for tmp in box:
    #             out_file.write(" ")
    #             out_file.write(str(tmp))
    #         out_file.write(" ")
    #         out_file.write(str(gesture_flag))
    #     out_file.write("\n")
    # out_file.close()
    gesture_txt_name = "gesture_" + name + ".txt"
    txt_name = os.path.join(save_path, gesture_txt_name)
    out_file = open(txt_name, "w")
    for data in gesture_list:
        out_file.write(data[0])
        box = data[1]
        flag = data[2]
        for tmp in box:
            out_file.write(" ")
            out_file.write(str(tmp))
        out_file.write(" ")
        out_file.write(str(flag))
        out_file.write("\n")
    out_file.close()


def visable_gesture(data_inf_box, save_name):
    img_path = data_inf_box[0]
    img = cv2.imread(img_path)
    img_draw = img.copy()
    for data in data_inf_box[1]:
        box = data[0]
        gesture_flag = data[-1]

        cv2.rectangle(img_draw, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 1)
        font_scale = min(img.shape[0] / 15, img.shape[1] / 15)
        font_scale = min(font_scale, box[0] / 3)
        font_scale = min(font_scale, box[1] / 3) / 10
        cv2.putText(img_draw, gesture_flag, (box[0], box[1]), cv2.FONT_HERSHEY_COMPLEX, font_scale, (100, 200, 200), 2)

    cv2.imwrite(save_name, img_draw)


def process_input_data(input_list, save_path, url, payload, input_flag, ori_img_save_path = ""):
    boxes_list = []
    gesture_list = []
    # s = requests.session()
    # s.keep_alive = False
    for index, image_name in enumerate(input_list):
        image_path = image_name
        img_file = {'image_file': open(image_path, 'rb')}
        
        s = requests.session()
        s.keep_alive = False
        r = requests.post(url, files=img_file, data=payload)

        if r:
            data = json.loads(r.text)
            if 'hands' not in data.keys() or len(data['hands']) == 0:
                print("%s find no hands." % (image_path))
                if input_flag == 1:
                    os.remove(image_path)
            else:
                # assert len(data['faces']) == 1
                boxes = []
                for i in range(len(data['hands'])):
                    hand = data['hands'][i]
                    width = hand['hand_rectangle']['width']
                    top = hand['hand_rectangle']['top']
                    height = hand['hand_rectangle']['height']
                    left = hand['hand_rectangle']['left']
                    box = [left, top, width, height]

                    gesture_predict = hand["gesture"]
                    hand_gesture = max(gesture_predict.items(), key=lambda x: x[1])[0]
                    hand_gesture_flag = gesture_dict[hand_gesture]

                    boxes.append([box, hand_gesture_flag, hand_gesture])
                    data_inf_gesture = [image_path, box, hand_gesture_flag, hand_gesture]
                    gesture_list.append(data_inf_gesture)

                data_inf_box = [image_path, boxes]
                boxes_list.append(data_inf_box)
                save_img = os.path.join(save_path, os.path.basename(image_path))
                if args.visible_flag:
                    visable_gesture(data_inf_box, save_img)
                # time.sleep(1)

        if (index + 1) % 100 == 0 or (index + 1) == len(input_list):
            print('%d done'%(index + 1))

        if (index + 1) % 5000 == 0:
            write_data_file(save_path, "hand_gesture" + "_" + str(index + 1), boxes_list, gesture_list)
            boxes_list = []
            gesture_list = []
            time.sleep(10)

        write_data_file(save_path, "hand_gesture" + "_" + "last", boxes_list, gesture_list)


def getHandGestureFromFacepp():
    requests.adapters.DEFAULT_RETRIES = 5
    url = 'https://api-cn.faceplusplus.com/humanbodypp/v1/gesture'
    payload = {
        # 'api_key':'FFIU24spbxVMdy2IZHuCQZP_a-zcqsZE',
        # 'api_secret':'jDZY475CNLNOwJXNsPiuSp8XXOBRQmbQ',
        'api_key': '02jqumPOP3yW8ajEJ9sNPOoTcSywiNja',
        'api_secret': '7SNWpDy1WZBz-lWO-ZpU-vqF0mSgfZud',
        'return_gesture': 1
    }
    input_flag = args.input_flag

    if input_flag == 0: # 0 means "read from file"
        input_path = args.input_dir  # Dir for New Face Images
        save_path = args.save_dir
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        L = file_name(input_path)
        L.sort(key=len)
        process_input_data(L, save_path, url, payload, input_flag)

    if input_flag == 1: # 1 means "read from video file"
        input_path = args.input_dir
        save_path = args.save_dir
        ori_img_save_path = "./test_imgs/video_ori_imgs"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(ori_img_save_path):
            os.makedirs(ori_img_save_path)
        L = pre_process_video(input_path, ori_img_save_path)
        process_input_data(L, save_path, url, payload, input_flag)


def parse_args():
    parser = argparse.ArgumentParser(description='facepp_hand_gesture')
    parser.add_argument('--input_dir', default="./test_imgs/hand", type=str, help="input dir for images")
    parser.add_argument('--save_dir', default="./test_output/hand", type=str, help="output_dir to save datas")
    parser.add_argument('--input_flag', default=0, type=int, help="input format:0-image_dir, 1-video_path")
    parser.add_argument('--visible_flag', default=True, type=bool, help="save visible imgs")

    global args
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)


def main():
    parse_args()
    getHandGestureFromFacepp()


if __name__ == '__main__':
    main()