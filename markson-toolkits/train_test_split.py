import os.path as osp
import os
import random
import shutil

src_path = '/Users/markson/WorkSpace/uyscutiengine/MNN Engine/datasets/HGR-dataset/version5/raw'


def train_test_split():
    # create train/test folder
    train_folder = osp.join(osp.dirname(src_path), 'train')
    test_folder = osp.join(osp.dirname(src_path), 'test')
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    print('#### Folder created!')
    # loop src path
    for root, dirs, files in os.walk(src_path):
        if dirs:
            continue
        src_list = os.listdir(root)
        random.shuffle(src_list)
        folder = root.split('/')[-1]
        train_path = osp.join(train_folder, folder)
        test_path = osp.join(test_folder, folder)
        # create classes folders

        os.makedirs(train_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)
        print('#### create classes folders \n{}\n{}'.format(train_path, test_path))
        # copy files
        for i in range(len(src_list)):
            src = osp.join(root, src_list[i])
            if i % 8 != 0:
                # train datasets
                dst = osp.join(train_path, src_list[i])
            else:
                # test datasets
                dst = osp.join(test_path, src_list[i])
            shutil.copy(src, dst)
    print('#### Done!')


def split_kfold():
    import shutil

    def split_percentage(im_list, percent=0.7):
        import random
        n = int(len(im_list)*percent)
        random.shuffle(im_list)
        return im_list[:n]

    classes = {
        0: 'others',
        1: 'red_pocket',
        2: 'porn',
        3: 'game',
        4: 'violence',
    }
    gt_path = '/Users/markson/WorkSpace/ads_reviews/data-version5/data-content/train'
    pse_path = '/Users/markson/WorkSpace/ads_reviews/data-version5/kfold'
    dst_path = '/Users/markson/WorkSpace/ads_reviews/data-version5/mix-train'
    for v in classes.values():
        gt_imlist = os.listdir(osp.join(gt_path, v))
        gt_imlist_7 = split_percentage(gt_imlist, 0.9)
        pse_imlist = os.listdir(osp.join(pse_path, v))
        pse_imlist_3 = split_percentage(pse_imlist, 0.5)
        os.makedirs(osp.join(dst_path, v), exist_ok=True)
        for im in gt_imlist_7:
            src = osp.join(gt_path, v, im)
            dst = osp.join(dst_path, v, im)
            shutil.copy(src, dst)
        for im in pse_imlist_3:
            src = osp.join(pse_path, v, im)
            dst = osp.join(dst_path, v, im)
            shutil.copy(src, dst)


if __name__ == '__main__':
    train_test_split()
