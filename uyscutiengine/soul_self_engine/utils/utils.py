import numpy as np
import math

def sigmoid(x):
    y = 1.0 / (1.0 + np.exp(-x))
    return y


def get_iou(box0, box1):
    up = max(box0[1], box1[1])
    down = min(box0[3], box1[3])
    left = max(box0[0], box1[0])
    right = min(box0[2], box1[2])

    inner_height = max(0, down - up)
    inner_width = max(0, right - left)

    inner_area = inner_height * inner_width
    area0 = (box0[2] - box0[0]) * (box0[3] - box0[1])
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    iou = inner_area / (area0 + area1 - inner_area)
    return iou


def weighted_non_max_suppression(select_boxes, iou_threshold=0.3):
    output_boxes = []
    while len(select_boxes) > 0:
        tmp_boxes = [select_boxes[0]]
        tmp_ids = [0]
        new_selected_boxes = []
        for i in range(1, len(select_boxes)):
            tmp_iou = get_iou(select_boxes[0], select_boxes[i])
            if tmp_iou > iou_threshold:
                tmp_boxes.append(select_boxes[i])
                tmp_ids.append(i)
            else:
                new_selected_boxes.append(select_boxes[i])
        output_box = [0] * len(tmp_boxes[0])
        for tmp in tmp_boxes:
            for i in range(len(tmp) - 1):
                output_box[i] = output_box[i] + tmp[i] * tmp[-1]
            output_box[-1] = output_box[-1] + tmp[-1]
        output_box = output_box / output_box[-1]
        output_boxes.append(output_box)
        select_boxes = new_selected_boxes
    return output_boxes  


def azimuth_angle(x1, y1, x2, y2):
    angle = 0.0
    dx = x2 - x1
    dy = y2 - y1
    if x2 == x1:
        angle = math.pi / 2.0
        if y2 == y1:
            angle = 0.0
        elif y2 < y1:
            angle = 3.0 * math.pi / 2.0
    elif x2 > x1 and y2 > y1:
        angle = math.atan(dx / dy)
    elif x2 > x1 and y2 < y1:
        angle = math.pi / 2 + math.atan(-dy / dx)
    elif x2 < x1 and y2 < y1:
        angle = math.pi + math.atan(dx / dy)
    elif x2 < x1 and y2 > y1:
        angle = 3.0 * math.pi / 2.0 + math.atan(dy / -dx)
    return (angle * 180 / math.pi)