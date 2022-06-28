import os
import cv2
import shutil
import numpy as np


def main():
    demo_path = '/Volumes/Lexar/data/LaPa/demo_fuse'
    done_path = '/Volumes/Lexar/data/LaPa/done'
    pseudo_path = '/Volumes/Lexar/data/LaPa/train/pseudo_labels'
    temp_path = '/Volumes/Lexar/data/LaPa/train/temp'
    remove_path = '/Volumes/Lexar/data/LaPa/remove'

    os.makedirs(done_path, exist_ok=True)
    os.makedirs(remove_path, exist_ok=True)
    tot = len(os.listdir(demo_path))
    for idx, nm in enumerate(os.listdir(demo_path)):
        print('[{}/{}]loading: {}...'.format(idx+1, tot, nm))
        demo = os.path.join(demo_path, nm)
        a = cv2.imread(demo)
        try:
            b = cv2.resize(cv2.imread(os.path.join('/Volumes/Lexar/data/LaPa/train/images', nm)), (512, 512))
        except:
            b = cv2.resize(cv2.imread(os.path.join('/Volumes/Lexar/data/LaPa/val/images', nm)), (512, 512))
        cv2.imshow('demo', np.hstack((a, b)))

        src_t = os.path.join(temp_path, nm.replace('jpg', 'png'))
        dst_t = os.path.join(pseudo_path, nm.replace('jpg', 'png'))
        src = os.path.join(demo_path, nm)
        dst = os.path.join(done_path, nm)
        dst_d = os.path.join(remove_path, nm)
        dst_t_d = os.path.join(remove_path, nm.replace('jpg', 'png'))

        if cv2.waitKey(0) != ord('d'):
            shutil.move(src, dst)
            shutil.move(src_t, dst_t)
        else:
            shutil.move(src, dst_d)
            shutil.move(src_t, dst_t_d)

        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
