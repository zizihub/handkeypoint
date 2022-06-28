import os
import os.path as osp
from PIL import Image
from collections import defaultdict
from imagededup.methods import CNN
from time import time
from datetime import timedelta
from imagededup.utils import plot_duplicates
from check_image_valid import multi_process_check
import shutil


class CleanImageData():
    '''
    Clean Similar Image
    '''

    def __init__(self, src_path, exclude, min_similarity_threshold=0.90) -> None:
        self.src_path = src_path
        self.exclude = exclude
        self.min_similarity_threshold = min_similarity_threshold
        self.images_encoding_dic = defaultdict()
        self.image_duplicate_dic = defaultdict()
        self.combined_remove = True
        self.pharsher = CNN()

    def get_repeated(self, img_key):
        encodings = self.pharsher.encode_images(self.src_path)
        duplicates = self.pharsher.find_duplicates(encoding_map=encodings,
                                                   min_similarity_threshold=self.min_similarity_threshold,
                                                   scores=True)
        plot_duplicates(image_dir=self.src_path,
                        duplicate_map=duplicates,
                        filename=img_key)

    def clean_repeated(self, remove=False):
        tmp_path = './tmp_remove_dup'
        os.makedirs(tmp_path, exist_ok=True)
        if self.combined_remove:
            tmp_path = './tmp_remove_dup'
            os.makedirs(tmp_path, exist_ok=True)
            for root, dirs, files in os.walk(osp.join(self.src_path, 'train')):
                if files:
                    continue
                for dir_ in dirs:
                    print('### copying {} ===> {}'.format(osp.join(root, dir_), tmp_path))
                    os.system('cp -r {} {}'.format(osp.join(root, dir_), tmp_path))
                    os.system('cp -r {} {}'.format(osp.join(root, dir_).replace('train', 'test'), tmp_path))
            print('### finished merge tmp')
            self._clean_repeated(tmp_path, remove=remove)
            for root, dirs, files in os.walk(tmp_path):
                if dirs and ('train' not in dirs or 'test' not in dirs):
                    for dir_ in dirs:
                        os.makedirs(osp.join(osp.dirname(self.src_path),
                                    'data-content-cleaned', 'train', dir_), exist_ok=True)
                        os.makedirs(osp.join(osp.dirname(self.src_path),
                                    'data-content-cleaned', 'test', dir_), exist_ok=True)
                for f in files:
                    _, train_test, classes, f_nm = f.split('-')
                    src = osp.join(root, f)
                    dst = osp.join(osp.dirname(self.src_path), 'data-content-cleaned', train_test, classes, f_nm)
                    print('### {} ===> {}'.format(src, dst))
                    shutil.copy(src, dst)
            shutil.rmtree(tmp_path)
        else:
            start = time()
            self._clean_repeated(self.src_path, remove=remove)
            end = time()
            print('\ntime cost: {}'.format(timedelta(seconds=end-start)))

    def _clean_repeated(self, tmp_path, remove=False):
        tot_num, tot_dup_num = 0, 0
        for root, dirs, files in os.walk(tmp_path):
            if dirs or self.find_exclude(root):
                continue
            print('\n\033[1;35m####### encoding path {}\033[0m'.format(root))
            folder_duplicate_list = self.pharsher.find_duplicates_to_remove(image_dir=root,
                                                                            min_similarity_threshold=self.min_similarity_threshold,
                                                                            outfile='{}/{}_dup.json'.format(osp.dirname(root), osp.basename(root)))
            self.image_duplicate_dic[root] = folder_duplicate_list
            tot_num += len(files)
            tot_dup_num += len(folder_duplicate_list)
        if remove:
            for root, files in self.image_duplicate_dic.items():
                for f in files:
                    src_path = osp.join(root, f)
                    os.remove(src_path)
                    print('####### deleting image {}  |  {}'.format(src_path, --tot_dup_num))
        else:
            print('\n\033[1;35m####### duplicated images will be preserved\033[0m')

        print(self.image_duplicate_dic)
        print('\n\033[1;35m####### total duplicated image: {}/{}\033[0m'.format(tot_dup_num, tot_num))

    def find_exclude(self, root):
        for exc in self.exclude:
            if exc in root:
                return True
        return False


def vis_dup_json():
    import json
    cur = '/Users/markson/WorkSpace/ads_reviews/data-version5/红包分类'
    for root, dirs, files in os.walk(cur):
        for f in files:
            if f.endswith('.json'):
                with open(osp.join(root, f), 'r+') as f1:
                    dup_dic = json.load(f1)
                folder = osp.join(root, f.split('_')[0])
                for im_name in dup_dic:
                    im_path = osp.join(folder, im_name)
                    img = cv2.imread(im_path)
                    cv2.imshow(im_name, img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()


def clean_dir(src_path):
    multi_process_check(src_path)
    cid = CleanImageData(src_path=src_path,
                         exclude=['None'],
                         min_similarity_threshold=0.95)
    cid.clean_repeated(remove=True)


if __name__ == '__main__':
    src_path = '/data1/zhangziwei/datasets/ad-dataset/version7/data-content'
    clean_dir(src_path=src_path)
