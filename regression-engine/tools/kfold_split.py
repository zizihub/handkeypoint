from sklearn.model_selection import StratifiedKFold
import os
import numpy as np
import json


def stratified_kfold_split():
    lines = []
    train_txt = '/data2/zhangziwei/datasets/HGR-dataset/version6/train.txt'
    test_txt = '/data2/zhangziwei/datasets/HGR-dataset/version6/test.txt'
    # load txt
    with open(train_txt, 'r') as f:
        lines.extend(f.readlines())
    with open(test_txt, 'r') as f:
        lines.extend(f.readlines())

    lines = np.array(lines)
    X, Y = [], []
    for line in lines:
        x, y = line.replace('\n', '').split(' ')
        X.append(x)
        Y.append(y)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1212)
    for i, (train_idx, test_idx) in enumerate(skf.split(X, Y)):
        print('train set nums: {} | test set nums: {}'.format(len(train_idx), len(test_idx)))
        with open(os.path.join(os.path.dirname(train_txt), 'train_fold_{}.txt'.format(i)), 'w+') as f:
            f.writelines(lines[train_idx])
        with open(os.path.join(os.path.dirname(train_txt), 'test_fold_{}.txt'.format(i)), 'w+') as f:
            f.writelines(lines[test_idx])


def check_kfold():
    src = '/data2/zhangziwei/datasets/HGR-dataset/version6'
    for i in range(5):
        with open(os.path.join(src, 'train_fold_{}.txt'.format(i)), 'r+') as f:
            train = f.readlines()
        with open(os.path.join(src, 'test_fold_{}.txt'.format(i)), 'r+') as f:
            test = f.readlines()
        print('fold', i, len(train)+len(test))


if __name__ == '__main__':
    stratified_kfold_split()
    check_kfold()
