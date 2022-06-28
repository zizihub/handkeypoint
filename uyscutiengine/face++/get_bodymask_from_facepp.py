import os
from PIL import Image
from tqdm import tqdm
import requests
from json import JSONDecoder
import base64
import argparse
from util_facepp import file_name, merge_output_img


http_url = "https://api-cn.faceplusplus.com/humanbodypp/v2/segment"
key = "FFIU24spbxVMdy2IZHuCQZP_a-zcqsZE"  # 更改成自己的密匙及密码即可
secret = "jDZY475CNLNOwJXNsPiuSp8XXOBRQmbQ"


def getPersonMaskFromFacepp():
    requests.adapters.DEFAULT_RETRIES = 5
    url = "https://api-cn.faceplusplus.com/humanbodypp/v2/segment"
    data = {"api_key": key, "api_secret": secret, "return_landmark": "1"}
    input_path = args.input_dir 
    save_path = args.save_dir
    L = file_name(input_path)
    L.sort(key=len)

    for index, image_name in enumerate(L):
        ori_img_path = image_name
        files = {"image_file": open(ori_img_path, "rb")}
        response = requests.post(url, data=data, files=files)
        req_con = response.content.decode('utf-8')
        req_dict = JSONDecoder().decode(req_con)
        if "result" not in req_dict:
            print("%s has not result."%(ori_img_path))
            continue
        result = req_dict['result']
        imgdata = base64.b64decode(result)
        mask_img_name = os.path.join(save_path, "mask_" + os.path.basename(ori_img_path))
        file = open(mask_img_name, 'wb')
        file.write(imgdata)
        file.close()

        if args.merge_flag:
            mask_img = Image.open(mask_img_name)
            ori_img = Image.open(ori_img_path)
            if mask_img.size[0] != ori_img.size[0]:
                ori_img = ori_img.rotate(270, expand=True)

            blend_img = Image.blend(ori_img.convert('RGBA'), mask_img.convert('RGBA'), 0.5)
            blend_img = blend_img.convert('RGB')
            merge_list = [ori_img, mask_img, blend_img]
            merge_img = merge_output_img(merge_list)
            merge_img.save(os.path.join(save_path, "merge_" + os.path.basename(ori_img_path)))
        index = index + 1

        if index % 2000 == 0:
            print('%d done'%(index))

def parse_args():
    parser = argparse.ArgumentParser(description='facepp_body_mask')
    parser.add_argument('--input_dir', default="./test_imgs/face", type=str, help="input dir for images")
    parser.add_argument('--save_dir', default="./test_output/body_mask", type=str, help="output_dir to save datas")
    parser.add_argument('--merge_flag', default=True, type=bool, help="save visible imgs")

    global args
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)


def main():
    parse_args()
    getPersonMaskFromFacepp()


if __name__ == "__main__":
    main()