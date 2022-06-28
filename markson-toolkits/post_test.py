import requests
import json
import cv2
import numpy as np
import base64
from PIL import Image
import os

image_url = 'https://china-img.soulapp.cn/image/2020-10-28/776be9ab-9677-458e-a986-e78f12b35c4a-1603825815666.png'
# ab_url = 'https://china-img.soulapp.cn/image/2021-04-02/795457f8-42b7-4264-8ec2-8c9ad16b69e6-1617332224349.png'

TORCH_LOCAL = True
FLASK_LOCAL = False


def bbox_vis(rgb, dets):
    '''
    cv2 imshow bbox and landmarks
    '''
    # show image
    for b in dets:
        text = "{:.4f}".format(b[4])
        b = list(map(int, b))
        cv2.rectangle(rgb, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        cx = b[0]
        cy = b[1] + 12
        cv2.putText(rgb, text, (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
        # landms
        cv2.circle(rgb, (b[5], b[6]), 1, (0, 0, 255), 4)
        cv2.circle(rgb, (b[7], b[8]), 1, (0, 255, 255), 4)
        cv2.circle(rgb, (b[9], b[10]), 1, (255, 0, 255), 4)
        cv2.circle(rgb, (b[11], b[12]), 1, (0, 255, 0), 4)
        cv2.circle(rgb, (b[13], b[14]), 1, (255, 0, 0), 4)
    # save image
    cv2.imshow('demo', rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def transfer_imgurl_to_oss_former(url):
    '''
    旧转换接口
    '''
    url = url.replace("img.soulapp.cn", "soul-app.oss-cn-hangzhou-internal.aliyuncs.com")
    url = url.replace("https", "http")
    return url


def transfer_imgurl_to_oss_now(url):
    '''
    新转换接口
    '''
    url = url.replace("china-chat-img.soulapp.cn", "soul-chat.oss-cn-hangzhou-internal.aliyuncs.com")
    url = url.replace("china-img.soulapp.cn", "soul-app.oss-cn-hangzhou-internal.aliyuncs.com")
    url = url.replace("img.soulapp.cn", "soul-app.oss-cn-hangzhou-internal.aliyuncs.com")
    url = url.replace("https", "http")
    return url


def img2b64(image_bgr):
    image_en = cv2.imencode(
        '.jpg', image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 100])[1]
    b64 = str(base64.b64encode(image_en), encoding='utf-8')
    return b64


def torchserve_test(image_url):
    '''
    RFBNet for face-embedding
    '''
    response = requests.get(transfer_imgurl_to_oss_now(image_url))

    if not response.ok:
        print('unable to download image')

    rgb = cv2.imdecode(np.frombuffer(
        response.content, np.uint8), cv2.IMREAD_ANYCOLOR)

    if len(rgb.shape) == 3:
        print(rgb.shape)
        data = {"base64": img2b64(rgb)}
        if not TORCH_LOCAL:
            print('remote torchserve post...')
            result = requests.post("http://test-ad-classification-cpu-test.c.soulapp-inc.cn/predictions/RFBNet",
                                   data=json.dumps(data), headers={"content-type": "application/json"})
        else:
            result = requests.post("http://172.16.69.255:8082/predictions/Spam062901",
                                   data=json.dumps(data), headers={"content-type": "application/json"})
            # if add:
            #     print('add here')
            #     result = requests.post("http://172.16.21.223:13130/predictions/RETINAADD",
            #                            data=json.dumps(data), headers={"content-type": "application/json"})
            # else:
            #     result = requests.post("http://172.29.100.1:12120/predictions/RFBNet",
            #                            data=json.dumps(data), headers={"content-type": "application/json"})
        print(result.ok)
        dets = json.loads(result.content)
        print(dets, np.array(dets).shape)
        return rgb, dets


def flask_test(image_url, num):
    '''
    universal flask test
    '''
    data = {
        "add": [
            {
                "imgUrlList": ['https://china-img.soulapp.cn/image/2021-04-25/bda38314-63f4-46a0-8f52-8725b6d1992c-1619285701044.png']*num,
                "pid": 1863952619
            }
        ]
    }
    if not FLASK_LOCAL:
        result = requests.post("http://prod-content-review-milvus.c.soulapp-inc.cn/add",
                               data=json.dumps(data),
                               headers={"content-type": "application/json"})
    else:
        result = requests.post("http://172.29.100.1:6979/add",
                               data=json.dumps(data),
                               headers={"content-type": "application/json"})
    content = json.loads(result.content)
    code = content['code']
    # message = content['message']
    print(content)
    if code != 200:
        # print(imgurl, code, message)
        return 1
    return 0


def iter_test():
    '''
    for loop iteration test
    '''
    from tqdm import tqdm
    with open("/Users/markson/WorkSpace/ads_reviews/rejected-sample/rejected-sample/0.txt", "r+") as f:
        url_list = f.readlines()
    for url in tqdm(url_list[2000:2500]):
        flask_test(url.replace('\n', ''))


def press_test():
    '''
    pressure test
    '''
    from multiprocessing import Pool
    from tqdm import trange
    from time import sleep
    p = Pool(4)
    with open("/Users/markson/WorkSpace/ads_reviews/rejected-sample/rejected-sample/0.txt", "r+") as f:
        url_list = f.readlines()
    counts = []
    t = trange(len(url_list[20000:25000]))
    for url in url_list[20000:25000]:
        count = p.apply_async(flask_test,
                              args=(url.replace('\n', ''), ),
                              callback=lambda _: t.update(),
                              error_callback=lambda _: t.update())
        counts.append(count)
    p.close()
    sleep(1)
    counts = [i.get() for i in counts]
    print('total unable to download image: {}/{}'.format(sum(counts), 5000))


if __name__ == "__main__":
    from multiprocessing import Pool
    import time
    if 1:
        time_mark = time.time()
        pool = Pool(10)
        count = []
        for _ in range(10):
            res = pool.apply_async(flask_test, args=(None, 1000))
            count.append(res)
        pool.close()
        result = [i.get() for i in count]
        print("cost: {}".format(time.time() - time_mark))
    else:
        # for i in range(1000, 9000, 1000):
        time_mark = time.time()
        flask_test(None, 1300)
        print("cost: {}".format(time.time() - time_mark))
