import cv2
import sys
import json
import os
from PIL import Image
import numpy as np
sys.path.append('..')  # NOQA:402
from src.mnn_blaze_detector import BlazeMNNDetector
import math
from copy import deepcopy


def normalized_l1(gt_landmarks, pred_landmarks, image_shape=(0, 0)):
    gt = []
    pred = []
    for x, y in gt_landmarks.values():
        gt.append([y, x])
    for x, y in pred_landmarks.values():
        pred.append([y, x])

    gt = np.array(gt)
    pred = np.array(pred)

    if not gt.shape == pred.shape:
        return 10000

    return np.mean(abs(gt-pred) / image_shape[:2]) * 100


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False):
    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (min(b1_x2, b2_x2) - max(b1_x1, b2_x1)) * (min(b1_y2, b2_y2) - max(b1_y1, b2_y1))

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter

    iou = inter / union  # iou
    if GIoU or DIoU or CIoU:
        cw = max(b1_x2, b2_x2) - min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = max(b1_y2, b2_y2) - min(b1_y1, b2_y1)  # convex height
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + 1e-16  # convex area
            return iou - (c_area - union) / c_area  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + 1e-16
            # centerpoint distance squared
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * np.arctan(w2 / h2) - np.arctan(w1 / h1) ** 2
                alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU

    return iou


def draw_utils(img, label_dic, gt=True):
    if gt:
        bbox_color = (0, 255, 0)  # green
        kpt_colors = (0, 0, 255)  # red
        xmin, ymin, xmax, ymax = label_dic['bbox']
    else:
        bbox_color = (255, 0, 0)  # blue
        kpt_colors = (0, 255, 255)  # yellow
        xmin, ymin, xmax, ymax = label_dic['pred_bbox']
    img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), bbox_color, 10, 10)
    for k, (x, y) in label_dic['landmark'].items():
        img = cv2.circle(img, (int(x), int(y)), 1, kpt_colors, 20, 20)
        # img = cv2.putText(img, str(k), (int(x), int(y)), 1, 10, (0, 0, 0), 20, 20)
    return img


def visualize(src_path):
    for root, dirs, files in os.walk(src_path):
        if dirs:
            continue
        for f in files:
            img_path = os.path.join(root, f)
            kpt_json_path = os.path.join(root.replace('image', 'pointsJson'), f.replace('jpg', 'json'))
            rect_json_path = os.path.join(root.replace('image', 'rectJson'), f.replace('jpg', 'json'))
            img = cv2.imread(img_path)
            label_dic = {}
            with open(kpt_json_path, 'r+') as f:
                label_dic.update(json.load(f)[0])
            with open(rect_json_path, 'r+') as f:
                label_dic.update(json.load(f)[0])
            label_dic['bbox'] = [label_dic['xmin'], label_dic['ymin'], label_dic['xmax'], label_dic['ymax']]
            print(img.shape)
            print(label_dic)

            img = draw_utils(img, label_dic)

            cv2.imshow('demo', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def inference(src_path):
    count = 0
    detector = BlazeMNNDetector('palm')
    l1_list = []
    iou_list = []
    giou_list = []
    diou_list = []
    ciou_list = []
    small_box = 0
    for root, dirs, files in os.walk(src_path):
        if dirs:
            continue
        for f in sorted(files):
            count += 1
            img_path = os.path.join(root, f)
            print(count, img_path)
            kpt_json_path = os.path.join(root.replace('image', 'points'), f.replace('jpg', 'json'))
            rect_json_path = os.path.join(root.replace('image', 'frames'), f.replace('jpg', 'json'))
            # change name
            if not os.path.exists(kpt_json_path):
                kpt_json_path = kpt_json_path.replace('.json', '_lm.json')
            if not os.path.exists(rect_json_path):
                rect_json_path = rect_json_path.replace('.json', '_det.json')
            label_dic = {}
            with open(kpt_json_path, 'r+') as f:
                js = json.load(f)
                label_dic.update(js[0] if isinstance(js, list) else js)
            with open(rect_json_path, 'r+') as f:
                js = json.load(f)
                label_dic.update(js[0] if isinstance(js, list) else js)
            label_dic['bbox'] = [label_dic['xmin'], label_dic['ymin'], label_dic['xmax'], label_dic['ymax']]
            print('gt bbox: {}, gt bbox area: {}'.format(
                label_dic['bbox'], (label_dic['bbox'][2]-label_dic['bbox'][0], label_dic['bbox'][3]-label_dic['bbox'][1])))
            if label_dic['bbox'][2]-label_dic['bbox'][0] < 150 or label_dic['bbox'][3]-label_dic['bbox'][1] < 150:
                small_box += 1
                continue
            print('landmarks:', label_dic['landmark'])
            gt_box = label_dic['bbox']
            gt_landmarks = deepcopy(label_dic['landmark'])
            img = cv2.imread(img_path)
            print(img.shape)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = draw_utils(img, label_dic)
            out, cost_palm = detector.predict(img_rgb, show_image=False)
            for box, landmarks in zip(out['boxes'], out['landmarks']):
                if 'pred_bbox' in label_dic.keys() and (box[2]-box[0])*(box[3]-box[1]) < (label_dic['pred_bbox'][2]-label_dic['pred_bbox'][0])*(label_dic['pred_bbox'][3]-label_dic['pred_bbox'][1]):
                    continue
                label_dic.update(pred_bbox=box)
                for i in range(0, len(landmarks), 2):
                    label_dic['landmark'].update(**{str(i//2): [landmarks[i], landmarks[i+1]]})

            if 'pred_bbox' in label_dic.keys():
                pred_box = label_dic['pred_bbox']
                img = draw_utils(img, label_dic, gt=False)
                print('pred bbox:', label_dic['pred_bbox'])
                print('landmarks:', label_dic['landmark'])

            l1_loss = normalized_l1(gt_landmarks, label_dic['landmark'], img.shape)
            iou = bbox_iou(pred_box, gt_box)
            giou = bbox_iou(pred_box, gt_box, GIoU=True)
            diou = bbox_iou(pred_box, gt_box, DIoU=True)
            ciou = bbox_iou(pred_box, gt_box, CIoU=True)

            print('Normalized L1 loss: {}'.format(l1_loss))
            print('DIoU: {} '.format(diou))
            print(small_box)
            print('='*10)
            if 'pred_bbox' in label_dic.keys() and iou > 0:
                iou_list.append(iou)
                giou_list.append(giou)
                diou_list.append(diou)
                ciou_list.append(ciou)
                l1_list.append(l1_loss)
            cv2.imshow('demo', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    print('Mean DIoU: {} | NormL1: {}'.format(np.mean(diou_list), np.mean(l1_list)))
    print('total num: {}'.format(count))


if __name__ == '__main__':
    src_path = '/Users/markson/Documents/data supplier/数据堂/image'
    inference(src_path)
