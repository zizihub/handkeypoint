import os
import csv
import os.path as osp
import urllib.request
from PIL import Image
from skimage import io
import random
import string

src = '/Users/markson/WorkSpace/ads_reviews/data-version7/URL'
dst = '/Users/markson/WorkSpace/ads_reviews/data-version7/correct'


IMG_EXTENSIONS = ('.ras', '.xwd', '.bmp', '.jpe', '.jpg', '.jpeg', '.xpm', '.ief',
                  '.pbm', '.tif', '.gif', '.ppm', '.xbm', '.tiff', '.rgb', '.pgm', '.png', '.pnm')
FAILED_DOWNLOAD = []


def random_name():
    return ''.join(random.sample(string.ascii_letters + string.digits, 5))


def transfer_imgurl_to_oss(url):
    url = url.replace('img.soulapp.cn', 'soul-app.oss-cn-hangzhou-internal.aliyuncs.com')
    url = url.replace('https', 'http')
    return url


def check_image_valid(root, im_p):
    '''
    check image validation by src_path. It will loop src_path to check each single files
    '''
    imp_abs = osp.join(root, im_p)
    try:
        _ = Image.open(imp_abs).convert('RGB')
        return 0
    except:
        # os.remove(imp_abs)
        print('invalid image {}'.format(imp_abs))
        return 1


def multi_process_check():
    from multiprocessing import Pool
    from tqdm import trange
    pool = Pool(16)
    src_path = dst
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


def download(line, dst_path):
    url = line.replace(',', '').replace('\n', '')
    try:
        if 0:
            url = transfer_imgurl_to_oss(url)
        urllib.request.urlretrieve(url, f'{dst_path}/{random_name()}.jpg')
        # print('downloaded {}'.format(url), end='\r', flush=True)
        return 0
    except:
        FAILED_DOWNLOAD.append(url)
        print('\n\033[1;35m unable to download {} \033[0m\n'.format(url))
        return 1


def custom_load_image():
    img_src = 'http://cpic.fancydsp.com/api/resizeStream?src=CIAFEMAHGl4KXAgCGlgKUGh0dHBzOi8vb3JhbmdlZmlyZS5kaWRpc3RhdGljLmNvbS9zdGF0aWMvb3JhbmdlZmlyZS84bmtjMDAwODhoNXVtNjFrd2hzc3NibWguanBnELgIGMAMIAEoATABOAE'
    image = io.imread(img_src)
    print(type(image))
    io.imshow(image)
    io.show()


def read_txt(txt_path):
    print('loading >>> {}'.format(txt_path))
    try:
        with open(txt_path, 'r+', encoding='cp1252') as f:
            lines = f.readlines()
    except:
        with open(txt_path, 'r+') as f:
            lines = f.readlines()
    return lines, osp.basename(txt_path).split('.')[0]


def load_folder(src):
    folder_dic = {}
    for root, dirs, files in os.walk(src):
        if dirs:
            continue
        for f in files:
            if f.startswith('.'):
                continue
            sp = osp.join(root, f)
            url_lines, txt_name = read_txt(sp)
            folder_dic[txt_name] = url_lines
    return folder_dic


def mp_download():
    from multiprocessing import Pool
    from tqdm import trange
    import time
    start = time.time()
    p = Pool(16)
    folder_dic = load_folder(src)
    results, tot = [], 0
    for txt_name, url_lines in folder_dic.items():
        tot += len(url_lines)
        t = trange(tot)
        for i, line in enumerate(url_lines):
            # random select
            # if i % 100 != 0:
            #     continue
            dst_pth = osp.join(dst, txt_name)
            os.makedirs(dst_pth, exist_ok=True)
            count = p.apply_async(download, args=(line, dst_pth),
                                  callback=lambda _: t.update(),
                                  error_callback=lambda _: t.update())
            results.append(count)
    p.close()
    failed = [i.get() for i in results]
    print('\nTime cost: {}s'.format(round(time.time()-start, 4)))
    print('Image fail to download: {}/{}'.format(sum(failed), tot))
    print('All subprocesses done.')


if __name__ == '__main__':
    # mp_download()
    multi_process_check()
