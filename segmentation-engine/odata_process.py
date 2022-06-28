import os
import cv2
import numpy as np


def label_process():
    path = r'/mnt/xuziyan/sky_dataset/mask-3'
    path_1 = r'/mnt/xuziyan/sky_dataset/mask-1'
    for filename in os.listdir(path):
        print(filename)

        img = cv2.imread(path+'/'+filename, cv2.IMREAD_GRAYSCALE)
        
        ret, img = cv2.threshold(img, 100, 1, cv2.THRESH_BINARY)

        print(np.unique(img))
        print(img)
        img = cv2.imwrite(path_1+'/'+filename, img)

        

if __name__ == '__main__':
    label_process()
