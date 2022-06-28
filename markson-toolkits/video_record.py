import cv2
import os
import json


label_dict = {
    '-1': 'no hand',
    '0': 'unknown',
    '1': 'thumb',
    '2': 'heart',
    '3': 'six',
    '4': 'fist',
    '5': 'palm',
    '6': 'one',
    '7': 'two',
    '8': 'ok',
    '9': 'rock',
    '10': 'cross',
    '11': 'hold',
    '12': 'greet',
    '13': 'photo',
    '14': 'heart',
    '15': 'merge',
    '16': 'eight',
    '17': 'halffist',
    '18': 'gun'
}


def main():
    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('/Users/markson/WorkSpace/uyscutiengine/MNN Engine/datasets/video/gesture_new.mp4',
                          fourcc, 20.0, (width, height), True)

    while True:
        _, frame = cap.read()
        cv2.imshow('video', frame)
        cv2.waitKey(1)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out.release()
    cap.release()
    cv2.destroyAllWindows()

def imgs_to_video():
    root = '/Users/markson/WorkSpace/uyscutiengine/MNN Engine/datasets/gesture_new'
    with open('/Users/markson/WorkSpace/uyscutiengine/MNN Engine/datasets/gesture_faceU+.json', 'r+') as fu:
        dic = json.load(fu)
    time_series_imgs = sorted(os.listdir(root), key=lambda x: os.path.getctime(os.path.join(root, x)), reverse=True)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter('/Users/markson/WorkSpace/uyscutiengine/MNN Engine/datasets/video/gesture_faceu_new.mp4',
                          fourcc, 15.0, (1280, 720), True)
    for img_nm in time_series_imgs:
        img = cv2.imread(os.path.join(root, img_nm))
        img = cv2.flip(img, 1)
        img_info = dic[img_nm]
        if 'rect' in img_info:
            xmin, ymin, xmax, ymax = img_info['rect'].split(' ')
            img = cv2.rectangle(img, (int(json.loads(xmin)), int(json.loads(ymin))),
                                ((int(json.loads(xmax)), int(json.loads(ymax)))), (0, 255, 0), 2, 2)
            label = label_dict[img_info['type']]
            score = img_info['score']
            img = cv2.putText(img, '{}: {}'.format(str(label), str(score)),
                              (int(json.loads(xmin)), int(json.loads(ymin))-25), 0, 1, (255, 0, 0), 2)
        cv2.imshow('demo', img)
        cv2.waitKey(5)
        out.write(img)
        cv2.destroyAllWindows()
    out.release()

if __name__ == '__main__':
    imgs_to_video()
