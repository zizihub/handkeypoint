import cv2
import os
import os.path as osp
import json
import numpy as np
from PIL import Image


np.set_printoptions(suppress=True)
bad_cases_json = '/Users/markson/WorkSpace/uyscutiengine/MNN Engine/datasets/HGR-dataset/version5/bad_case.json'
src_path = '/Users/markson/WorkSpace/uyscutiengine/MNN Engine/datasets/HGR-dataset/version5'
label_file = '/Users/markson/WorkSpace/uyscutiengine/MNN Engine/datasets/HGR-dataset/version5/label.json'


def bad_cases_2_images():
    with open(label_file, 'r') as f:
        classes = json.load(f)
    dst_path = osp.join(osp.dirname(src_path), 'bad case')
    os.makedirs(dst_path, exist_ok=True)
    with open(bad_cases_json, 'r+') as f:
        js = json.load(f)
    for k, v in js.items():
        white_pad = np.zeros([27*len(classes), 320, 3], dtype=np.uint8)+255
        img = cv2.imread(osp.join(src_path, 'test', k))
        if img is None:
            print('Unable to open {}'.format(k))
            continue
        img = cv2.resize(img, (320, 320))
        cv2.putText(white_pad, 'Label: '+str(osp.dirname(k)), (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, 0)
        if len(np.array(v).shape) > 1:
            # multi-head
            result = np.around(np.array(v).reshape(len(classes), 2), decimals=4)
        else:
            result = np.around(v, decimals=4)
        for i in range(result.shape[0]):
            string = str(classes[i])+' '+str(result[i])
            cv2.putText(white_pad, string, (10, (i+2)*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, 0)
        img = np.vstack((img, white_pad))
        dst_img = osp.join(dst_path, osp.basename(k).split('.')[0]+'_{}.jpg'.format(osp.dirname(k)))
        cv2.imshow('demo', img)
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.imwrite(dst_img, img)

    print('### Done!')


if __name__ == '__main__':
    bad_cases_2_images()
