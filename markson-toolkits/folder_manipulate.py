import os
import os.path as osp
import random
import string
import shutil
from web_download import random_name


class RenameFiles():
    def __init__(self):
        self.dst_name = self.random_name()

    def random_name(self):
        return ''.join(random.sample(string.ascii_letters + string.digits, 5))

    def rename(self, src):
        # dst = '{}/{}.jpg'.format(os.path.dirname(src), self.dst_name)
        bs_name = osp.basename(src)
        dst = osp.join(osp.dirname(src), bs_name.split('-')[-1])
        os.rename(src, dst)
        print('{} ========> {}'.format(src, dst))

    def folder_rename_files(self, src):
        for root, dirs, files in os.walk(src):
            if dirs:
                continue
            for f in files:
                self.rename(osp.join(root, f))


def folder_combine(src, dst):
    os.makedirs(dst, exist_ok=True)
    tot = 0
    cur = 0
    for root, dirs, files in os.walk(src):
        if dirs:
            continue
        tot += len(files)
        for f in files:
            cur += 1
            src_path = os.path.join(root, f)
            dst_path = os.path.join(dst, '{}.jpg'.format(random_name()))
            shutil.move(src_path, dst_path)
            print('[{}/{}]moving {} ====> {}'.format(cur, tot, src_path, dst_path))


def merge_folder(src, dst):
    pass


def batch_change_ext(src, ext='.jpg'):
    for root, dirs, files in os.walk(src):
        if dirs:
            continue
        for f in files:
            if not f.endswith(ext):
                _, pre_ext = os.path.splitext(f)
                s_f = os.path.join(root, f)
                d_f = os.path.join(s_f.replace(pre_ext, ext))
                print('### rename {} ==> {}'.format(os.path.basename(s_f), os.path.basename(d_f)))
                os.rename(s_f, d_f)


def folder_split(src, n_split=3):
    base_name = os.path.basename(src)
    dir_path = os.path.dirname(src)
    op_list = []
    for idx in range(n_split):
        op = os.path.join(dir_path, base_name+'_{}'.format(idx))
        os.makedirs(op, exist_ok=True)
        op_list.append(op)

    for i, f in enumerate(os.listdir(src)):
        src_path = os.path.join(src, f)
        residual = i % n_split
        dst_path = os.path.join(op_list[residual], f)
        shutil.move(src_path, dst_path)
        print('moving {} ====> {}'.format(src_path, dst_path))


if __name__ == "__main__":
    folder_combine('/Users/markson/Downloads/Validation', '/Users/markson/Downloads/total')
