import os
import json
import argparse


def getArgs():
    args = argparse.ArgumentParser()
    args.add_argument('-i', '--input', help='input json')
    return vars(args.parse_args())


if __name__ == '__main__':
    args = getArgs()
    in_json = args['input']
    assert os.path.exists(in_json)
    rst_name = 'label_' + os.path.basename(in_json)

    with open(in_json, 'r') as fr:
        js = json.load(fr)[0]['data']
    print(js)
    with open(rst_name, 'w') as fw:
        for idx, line in enumerate(js):
            line = line['url'].strip()
            if line and (not line.startswith('http')):
                continue
            # json format
            data = {'id': idx,
                    'text': None,
                    'image_url': line,
                    'video_urls': None,
                    'audio_urls': None,
                    'comments': None,
                    'tags': None,
                    'recommend_label_id': None}
            rst = json.dumps(data) + '\n'
            fw.write(rst)
