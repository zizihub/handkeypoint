import os
import cv2
import numpy as np
from PIL import Image

def file_name(file_dir):
    img_suffix = ['.jpg', '.jpeg', '.png']
    list1 = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in img_suffix:
                list1.append(os.path.join(root, file))
    return list1


def get_base_name(image_path):
    name = image_path.replace("/", "_")
    return name


def pre_process_video(input_path, ori_img_save_path):
    list_output = []
    index = 0
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.lower().endswith(('.mp4', '.mov', '.avi')):
                video_path = os.path.join(root, file)
                cap = cv2.VideoCapture(video_path)
                count = 0
                while(1):
                    ret, frame = cap.read()
                    if count % 10 == 0 and np.max(frame) is not None:
                        ori_img_name = os.path.join(ori_img_save_path, "index_" + str(index).zfill(8) + ".jpg")
                        cv2.imwrite(ori_img_name, frame)
                        list_output.append(ori_img_name)
                        index = index + 1
                    if count == 5000:
                        break
                    if cv2.waitKey(100) & 0xFF == ord('q'):
                        break
                    count = count + 1
                cap.release()
                cv2.destroyAllWindows()
    print("video frame list len is ", len(list_output))
    return list_output

def merge_output_img(img_list):
    img_size = img_list[0].size
    merge_img = Image.new('RGB', (len(img_list) * img_list[0].size[0], img_list[0].size[1]))
    count = 0
    for img in img_list:
        merge_img.paste(img, (count * img_size[0], 0))
        count = count + 1
    return merge_img