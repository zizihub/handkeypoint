'''
基于上海机器输出待显示的图片信息
python import.py --source <image_folder> --soft-link <datasets_name> --save-name <datasets_name>.json
'''
import json

import argparse
from types import SimpleNamespace
import os

logger = SimpleNamespace()
logger.info = print
save_root = '/home/dickzhou/workspace/data/pages/stats/'
template = 'http://172.29.100.1:8081/pages/benchmark/?set=empty.labels.json&data={}'


class Stats(object):
    def __init__(self, _sentinel=True):
        if _sentinel:
            raise ValueError(
                f'direct instantiation of class {self.__class__.__name__} is not supported')

        self._records = []
        self._group2index = {}
        self._num_record = 0
        self._path = None

    def __iter__(self):
        for record in self._records:
            current_group = record['id']
            for datum in record['data']:
                yield [
                    current_group,
                    datum
                ]

    def __enter__(self):
        try:
            file = open(self._path, 'w')
            file.close()
        except IOError:
            raise ValueError(f'cannot write to {self._path}')

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save(self._path)

    def __len__(self):
        return self._num_record

    @classmethod
    def empty(cls):
        return cls(_sentinel=False)

    @classmethod
    def open(cls, path: str, truncate: bool = False):
        try:
            if truncate:
                raise IOError()
            if os.path.exists(path):
                raise ValueError(f'file already exists: {path}')
            instance = cls.load(path)
        except IOError:
            instance = cls(_sentinel=False)
        instance._path = path

        return instance

    @classmethod
    def load(cls, path: str):
        stats = cls(_sentinel=False)
        with open(path) as stats_file:
            stats._records = json.load(fp=stats_file)
        for index, record in enumerate(stats._records):
            stats._num_record += len(record['data'])
            stats._group2index[record['id']] = index

        logger.info(f'loaded {path} ({stats._num_record})')
        return stats

    def save(self, path):
        with open(path, 'w') as stats_file:
            json.dump(self._records, fp=stats_file,
                      indent=2, ensure_ascii=False)

        logger.info(f'saved {path} ({self._num_record})')
        return self

    def merge(self, stats_path):
        for group, datum in Stats.empty().load(stats_path):
            self.add_record(group=group, datum=datum)

        logger.info(f'merged {stats_path} ({self._num_record})')
        return self

    def add_record(self, group, datum):
        if group is None or datum is None:
            return self

        if group not in self._group2index:
            self._group2index[group] = len(self._group2index)
            self._records.append({
                'id': group,
                'data': []
            })
        index = self._group2index[group]
        self._records[index]['data'].append(datum)
        self._num_record += 1

        return self


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s',
        '--source',
        help='source folder'
    )
    parser.add_argument(
        '-sl',
        '--soft-link',
        help='softlink address'
    )
    parser.add_argument(
        '-n',
        '--save-name',
        required=True,
        help='name of the saved json'
    )
    args_parsed = parser.parse_args()
    source: str = args_parsed.source  # folder路径
    soft_link: str = args_parsed.soft_link
    save_name: str = args_parsed.save_name  # 保存位置，用于输出显示

    destination = os.path.join(save_root, save_name)
    with Stats.open(destination) as stat:
        for fn in os.listdir(source):
            if not fn.startswith('http://'):
                url = f'http://172.29.100.1:8081/{soft_link}/{fn}'

            stat.add_record(
                group='image',
                datum={
                    'url': url,
                    'tags': ['unlabelled']
                }
            )
    logger.info(f'imported as: {template.format(save_name)}')


if __name__ == '__main__':
    main()
