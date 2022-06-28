import re
import os
import os.path as osp
import requests
import json
import numpy as np


def norm(encoding):
    encoding = np.array(encoding, dtype=np.float32)
    encoding_norm = np.linalg.norm(encoding)
    encoding = encoding / encoding_norm if encoding_norm > 1e-10 else encoding
    return list(map(lambda x: round(float(x), 8), encoding))


class ContentReviewManager(object):
    def __init__(self, file_path='/data/456_imgs/aug_may_june_0', out_path=''):
        super().__init__()
        self.file_path = file_path
        self.out_path = out_path
        self.pid_split = 1798140516
        self.embedding_api = 'http://prod-image-embedding-cpu.c.soulapp-inc.cn/embedding'
        self.import_api = 'http://prod-content-review-milvus.c.soulapp-inc.cn/embedding_batch_add'

    def file_split(self, nq=10):
        '''
        将大文件拆分成小文件
        Args:
            nq: 小文件个数
        '''
        from tqdm import tqdm
        with open(self.file_path, 'r+', errors='ignore') as f:
            lines = f.readlines()
        for idx, line in enumerate(tqdm(lines)):
            mod = idx % nq
            with open(self.file_path+'_{}'.format(mod), 'a+', errors='ignore') as f_o:
                f_o.write(line)

    def get_recent_data(self):
        '''
        根据pid获取最近的数据，pid随时间一直增长
        '''
        from tqdm import tqdm
        for root, dirs, files in os.walk(self.file_path):
            if dirs:
                continue
            for f in files:
                out_list = []
                print('### analysing {}'.format(f))
                with open(osp.join(root, f), 'r+') as in_f:
                    lines = in_f.readlines()
                for line in tqdm(lines):
                    pid, url = re.split("\x012\x01|\x013\x01", line.replace('\n', ''))
                    if int(pid) >= self.pid_split:
                        out_list.append(line)
                with open(self.out_path, 'a+') as f:
                    f.writelines(out_list)
                del out_list, lines

    def import_data(self, func):
        for root, dirs, files in os.walk(self.file_path):
            if dirs:
                continue
            for f in files:
                with open(osp.join(root, f), 'r+', errors='ignore') as in_f:
                    lines = in_f.readlines()
                func(lines)

    def import_single(self, func):
        with open(self.file_path, 'r+', errors='ignore') as in_f:
            lines = in_f.readlines()
        func(lines)

    def test(self, func, test_file='test_10k'):
        with open(test_file, 'r+') as in_f:
            lines = in_f.readlines()
            func(lines)

    def query_embedding(self, line, f_nm):
        # 请求图像Embedding
        pid, url = re.split("\x012\x01|\x013\x01", line.replace('\n', ''))
        embedding_body = json.dumps({"imgUrl": url,
                                     'mode': 'alex'})
        response = requests.post(url=self.embedding_api, data=embedding_body, headers={
            "content-type": "application/json"
        }, timeout=1000)
        result = response.json()
        if result["code"] != 200:
            print('fail to add line:', line)
            with open('add_failed.txt', 'a+') as f:
                f.write(line)
            return 1
        embedding = result['predictions'][0]['extra']
        # 按之前版本距离计算方式继续使用IP，需要归一化
        embedding = norm(embedding)
        with open('{}_batch_embedding.txt'.format(f_nm), 'a+') as f:
            f.write('{};#;{};#;{}\n'.format(pid, url, embedding))
        return 0

    def request_body(self, batch):
        '''
        flask test
        '''
        tmp_data = json.dumps(batch)
        result = requests.post(self.import_api,
                               data=tmp_data,
                               headers={"content-type": "application/json"})
        try:
            content = json.loads(result.content)
            code = content['code']
            if code != 200:
                return 1
        except:
            return 1
        return 0

    def mp_data_embedding(self, lines, f_nm):
        from multiprocessing import Pool
        from tqdm import trange
        tot = len(lines)
        pool = Pool(64)
        results = []
        t = trange(tot)
        for _, line in enumerate(lines):
            count = pool.apply_async(
                self.query_embedding,
                args=(line, f_nm,),
                callback=lambda _: t.update(),
                error_callback=lambda _: t.update()
            )
            results.append(count)
        pool.close()
        counts = sum([i.get() for i in results])
        print('### Fail to add {}'.format(counts))

    def mp_data_import(self, lines):
        from multiprocessing import Pool
        from tqdm import trange, tqdm

        # generate batch
        tot = len(lines)
        batch = []
        tmp_dict = {"imgUrlList": [],
                    "pidList": [],
                    "embedding_list": []}
        for i, line in enumerate(tqdm(lines)):
            if (i+1) % 10000 == 0 or (i+1) == tot:
                pid, url, embedding = line.replace('\n', '').split(';#;')
                tmp_dict["pidList"].append(pid)
                tmp_dict["imgUrlList"].append(url)
                tmp_dict["embedding_list"].append(eval(embedding))
                batch.append(tmp_dict)
                tmp_dict = {"imgUrlList": [],
                            "pidList": [],
                            "embedding_list": []}
            else:
                pid, url, embedding = line.replace('\n', '').split(';#;')
                tmp_dict["pidList"].append(pid)
                tmp_dict["imgUrlList"].append(url)
                tmp_dict["embedding_list"].append(eval(embedding))
                continue
        # batch inserts
        pool = Pool(32)
        results = []
        t = trange(len(batch))
        for b in batch:
            count = pool.apply_async(
                self.request_body,
                args=(b,),
                callback=lambda _: t.update(),
                error_callback=lambda _: t.update()
            )
            results.append(count)
        pool.close()
        counts = sum([i.get() for i in results])
        print('### Fail to add {}'.format(counts))


if __name__ == '__main__':
    crm = ContentReviewManager()
    crm.import_single(crm.mp_data_embedding)
    # crm.test(crm.mp_data_import, test_file='./batch_embedding.txt')
