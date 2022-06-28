import os
import os.path as osp
import pandas as pd
import numpy as np
import cv2
import shutil
from tqdm import tqdm


class CSVProcess(object):
    def __init__(self, csv_file, img_folder=None):
        self.csv_file = csv_file
        self.img_folder = img_folder
        self.classes = {
            0: 'others',
            1: 'red_pocket',
            2: 'porn',
            3: 'game',
            4: 'violence',
            5: 'book'
        }

    def prob_to_folder(self, threshold=0.90, output_folder='0623-sbl-0.95'):
        assert self.img_folder is not None, 'please define img folder'
        # create high score classes for pseudo labels
        root = osp.dirname(self.img_folder)
        os.makedirs(osp.join(root, output_folder), exist_ok=True)
        for v in self.classes.values():
            op_dir = osp.join(root, output_folder, v+'_highscore')
            os.makedirs(op_dir, exist_ok=True)

        prob_df = pd.read_csv(self.csv_file)
        for _, row in tqdm(prob_df.iterrows(), total=prob_df.shape[0]):
            image_name = row['image_id']
            prob_result = np.asarray(np.matrix(row['label']).reshape(len(self.classes), 2))
            label = int(np.argmax(prob_result[:, 1]))
            pred = round(prob_result[label].flatten()[1], 4)
            if 0:
                img = cv2.imread(osp.join(self.img_folder, image_name))
                cv2.imshow('{}-{}'.format(self.classes[label], pred), img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            if pred < threshold:
                continue
            src = osp.join(self.img_folder, image_name)
            dst = osp.join(
                root, '{}/{}/{}_{}.jpg'.format(output_folder, self.classes[label]+'_highscore', image_name.split('.')[0], pred))
            shutil.copy(src, dst)
        print('#### Done!')

    def cls_to_folder(self):
        '''
        classify unknown image by csv file
        根据csv归类图片
        '''
        df = pd.read_csv(self.csv_file)
        root = osp.dirname(self.img_folder)
        for v in self.classes.values():
            op_dir = osp.join(root, v)
            if osp.exists(op_dir):
                continue
            os.makedirs(op_dir, exist_ok=True)
        for _, row in tqdm(df.iterrows()):
            src = osp.join(root, 'image', row['image_id'])
            dst = osp.join(root, self.classes[row['label']], row['image_id'])
            shutil.copy(src, dst)
        print('###### Done')


if __name__ == '__main__':
    csv_file = '/data2/zhangziwei/classification-engine/log/resnet18_MultiHead_sz360x240_lbs_0623-imbalanced/r18_prob.csv'
    img_folder = '/data1/zhangziwei/datasets/ad-dataset/version7/others'
    pp = CSVProcess(csv_file=csv_file,
                    img_folder=img_folder)
    pp.prob_to_folder(output_folder='0623-imbalanced')
