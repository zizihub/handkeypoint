import cv2
import numpy as np
import math

def padding_img(image):
    new_len = max(image.shape[0], image.shape[1])
    if new_len == image.shape[0]:
        padding_size_0 = int((new_len - image.shape[1])/2)
        padding_size_1 = new_len - image.shape[1] - padding_size_0
        image = cv2.copyMakeBorder(image, 0, 0, padding_size_0, padding_size_1, cv2.BORDER_CONSTANT)
    else:
        padding_size_0 = int((new_len - image.shape[0])/2)
        padding_size_1 = new_len - image.shape[0] - padding_size_0
        image = cv2.copyMakeBorder(image, padding_size_0, padding_size_1, 0, 0, cv2.BORDER_CONSTANT)
    return image


def pad_and_resize(img, resize_shape, padding = True, norm_flag = 0, change_order = True, rotation = 0, add_axis = True):
    """
    pad and resize images

    Parameters
    ----------
    img : input_data, cv2.imread image, format is BGR
    resize_shape : list [width, heigt]
    padding: pad image to square or not, add black border 
    norm_flag: 0 —> -1~1, 1 -> 0~1
    change_order： change img to CHW or not
    rotation: rot 90 times, if 0 not rotate
    add_axis: add first axis or not, add batch dimension
    Returns
    numpy array
    """
    
    if padding:
        img = padding_img(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_data = cv2.resize(img, (resize_shape[0], resize_shape[1]))
    if rotation != 0:
        img_data = np.rot90(img_data, k=rotation)
    img_data = img_data.astype(np.float32)
    if norm_flag == 0:
        img_data = (img_data - 127.5) / float(127.5)
    if norm_flag == 1:
        img_data = img_data / float(255.0)
    if change_order:
        img_data = img_data.transpose((2, 0, 1))
    if add_axis:
        img_data = img_data[np.newaxis, :, :, :]
    return img_data


def draw_boxes(img, boxes, color=(0, 0, 255)):
    line_thickness = min(img.shape[:2])
    line_thickness = int(line_thickness / 100)

    for box in boxes:
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, line_thickness)
    
    return img


def draw_direction(img, direciton, face_box = None, color=(200, 144, 0)):
    if face_box is None:
        face_box = [0, 0, img.shape[1] - 1, img.shape[0] - 1]
    
    pos = (face_box[0] + (face_box[2] - face_box[0]) // 10, face_box[1] + (face_box[3] - face_box[1]) // 10)
    fontScale = min(img.shape[:2]) / 800
    cv2.putText(img, str(direciton), pos, cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness=2)
    return img

def draw_landmark(img, landmarks, face_box = None, color=(0, 255, 0)):
    if face_box is None:
        face_box = [0, 0, img.shape[1] - 1, img.shape[0] - 1]
    radius = min(img.shape[:2])
    radius = int(radius / 100)
    for i in range(landmarks.shape[0]):
        p_x = int(landmarks[i][0] + face_box[0])
        p_y = int(landmarks[i][1] + face_box[1])
        cv2.circle(img, (p_x, p_y), radius, color, -1)
    return img


def crop_and_pad(img, ori_box):
    """
    crop image base crop_box
    :param img: ori image
    :param ori_box: crop box type:list len:4  [left, top, right, bottom]
    return crop image
    """
    padding_size = [0, 0, 0, 0]  # top, bottom, left, right
    crop_box = ori_box.copy()
    if crop_box[0] < 0:
        padding_size[2] = -crop_box[0]
        crop_box[0] = 0
    if crop_box[1] < 0:
        padding_size[0] = -crop_box[1]
        crop_box[1] = 0
    if crop_box[2] > img.shape[1]:
        padding_size[3] = crop_box[2] - img.shape[1]
        crop_box[2] = img.shape[1]
    if crop_box[3] >= img.shape[0]:
        padding_size[1] = crop_box[3] - img.shape[0]
        crop_box[3] = img.shape[0]
    # print("crop box ", crop_box)
    crop_img = img[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
    crop_img = cv2.copyMakeBorder(crop_img, padding_size[0], padding_size[1], padding_size[2], padding_size[3], cv2.BORDER_CONSTANT)
    return crop_img


def crop_box_img(img, face_box, len_scale = 1.1, use_dia = False):
    """
    box: lurd
    """
    x_min = face_box[0]
    x_max = face_box[2]
    y_min = face_box[1]
    y_max = face_box[3]

    if use_dia:
        new_len = int(math.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2) * len_scale)
    else:
        new_len = int(max(x_max - x_min, y_max - y_min) * len_scale)
    new_x_min = int((x_min + x_max) // 2 - new_len // 2)
    new_y_min = int((y_min + y_max) // 2 - new_len // 2)
    crop_box = [new_x_min, new_y_min, new_x_min + new_len, new_y_min + new_len]
    crop_img = crop_and_pad(img, crop_box)
    return crop_img, crop_box


def get_base_box(landmark):
    x_list = np.reshape(landmark, (-1, 2))[:, 0]
    y_list = np.reshape(landmark, (-1, 2))[:, 1]
    x_min = np.min(x_list)
    x_max = np.max(x_list)
    y_min = np.min(y_list)
    y_max = np.max(y_list)

    return [x_min, y_min, x_max, y_max]