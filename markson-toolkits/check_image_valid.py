import os
from sklearn.model_selection import train_test_split
import os.path as osp
import shutil
from PIL import Image


def check_image_valid(root, im_p):
    '''
    check image validation by src_path. It will loop src_path to check each single files
    '''
    imp_abs = osp.join(root, im_p)
    try:
        _ = Image.open(imp_abs).convert('RGB')
        # print('loading image {}/{}'.format(cur, tot), flush=True, end='\r')
        return 0
    except:
        os.remove(imp_abs)
        print('invalid image {}'.format(imp_abs))
        return 1


def multi_process_check(src_path):
    from multiprocessing import Pool
    from tqdm import trange
    pool = Pool(16)
    src_path = '/Users/markson/Downloads/Validation'
    results, tot, p = [], 0, None

    for root, dirs, f in os.walk(src_path):
        if dirs:
            continue
        tot += len(f)
        p = trange(tot)
        for im_p in f:
            count = pool.apply_async(
                check_image_valid,
                args=(root, im_p,),
                callback=lambda _: p.update(),
                error_callback=lambda _: p.update()
            )
            results.append(count)
    pool.close()
    counts = [i.get() for i in results]
    print('total invalid image {}/{}'.format(sum(counts), tot))


if __name__ == '__main__':
    multi_process_check()
