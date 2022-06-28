#!/usr/bin/python
# -*- encoding: utf-8 -*-

from seg_engine.models.build import build_model
from seg_engine.config import get_cfg, get_outname
import torch

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
from copy import deepcopy
from tqdm import tqdm
from time import time
import MNN
import sys
import argparse
sys.path.append(
    '/Users/markson/WorkSpace/Ultra-Light-Fast-Generic-Face-Detector-1MB')
from vision.utils.misc import Timer  # NOQA: 402
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor  # NOQA: 402
from vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor  # NOQA: 402
from vision.ssd.config.fd_config import define_img_size  # NOQA:402


to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def setup():
    '''
    Create configs and perform basic setups.
    '''
    my_cfg = './face_config.yaml'
    cfg = get_cfg()
    cfg.merge_from_file(my_cfg)
    output_name = get_outname(cfg)
    cfg.merge_from_list(['OUTPUT_NAME', output_name])
    cfg.freeze()
    return cfg, my_cfg


def vis_parsing_maps(im, parsing_anno, stride=1, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    '''
    0: 'background',
    1: 'facial skin',
    2: 'brow',
    3: 'eye',
    4: 'nose',
    5: 'upper lip',
    6: 'inner mouth',
    7: 'lower lip',
    '''
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [105, 128, 112],
                   [85, 96, 225], [255, 0, 170],
                   [0, 255, 0], [85, 0, 255], [170, 255, 0],
                   [255, 255, 255], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]
    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(
        vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros(
        (vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        # if pi != 5 and pi != 6 and pi != 7:
        #     continue
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    vis_im = cv2.addWeighted(cv2.cvtColor(
        vis_im, cv2.COLOR_RGB2BGR), 0.7, vis_parsing_anno_color, 0.3, 0)

    # Save result or not
    if save_im:
        # cv2.imwrite(save_path[:-4] + '.png', vis_parsing_anno)
        cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    return vis_im, vis_parsing_anno_color


def face_model_init():
    input_img_size = 640
    # must put define_img_size() before 'import create_mb_tiny_fd, create_mb_tiny_fd_predictor'
    define_img_size(input_img_size)
    label_path = "/Users/markson/WorkSpace/Ultra-Light-Fast-Generic-Face-Detector-1MB/models/voc-model-labels.txt"
    class_names = [name.strip() for name in open(label_path).readlines()]
    test_device = 'cpu'
    candidate_size = 1000
    threshold = 0.7
    model_path = "/Users/markson/WorkSpace/Ultra-Light-Fast-Generic-Face-Detector-1MB/models/pretrained/version-slim-640.pth"
    net = create_mb_tiny_fd(
        len(class_names), is_test=True, device=test_device)
    predictor = create_mb_tiny_fd_predictor(
        net, candidate_size=candidate_size, device=test_device)
    net.load(model_path)
    return predictor, threshold, candidate_size


def mask_recover(frame, vis_parsing_anno_color, box, face_img, ratio):
    vis_im = frame.copy().astype(np.uint8)
    vis_parsing_anno_reverse = cv2.resize(
        vis_parsing_anno_color, (face_img.shape[1], face_img.shape[0]))
    blank_mat = np.zeros(
        (frame.shape[0], frame.shape[1], 3), dtype=np.uint8) + 255
    blank_mat[int(box[1]*(1-ratio)):int(box[3]*(1+ratio)),
              int(box[0]*(1-ratio)):int(box[2]*(1+ratio))] = vis_parsing_anno_reverse
    vis_img = cv2.addWeighted(vis_im, 0.7, blank_mat, 0.3, 0)
    return vis_img, blank_mat


def cam_inference(args, mnn=False):
    ###############  video cfg  ##################
    partial = 'large_model'
    src_video = '../video/obama.mov'
    input_size = args.input_size
    num_class = 7
    if mnn:
        ###############  MNN cfg  ##################
        mnn_model = args.mnn_model
        net_mnn = MNN.Interpreter(mnn_model)
        config = {}
        config['precision'] = 'low'
        session = net_mnn.createSession()
        input_tensor = net_mnn.getSessionInput(session)
    else:
        cfg, _ = setup()
        net = build_model(cfg)
        net.eval()
    # init face model
    predictor, threshold, candidate_size = face_model_init()
    cost = []
    # init camera
    if 0:
        # save video
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out_video = cv2.VideoWriter(
            f'../video/demo.mp4', fourcc, 30.0, (1280, 720), True)
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        count = 0
        # get image frame:
        while count < 480:
            count += 1
            print(count)
            _, frame = cap.read()
            cv2.imshow('image capture', frame)
            cv2.waitKey(1)
            out_video.write(frame)
        cv2.destroyAllWindows()
        cap.release()
        out_video.release()
        exit()
    # init video
    else:
        cap = cv2.VideoCapture(src_video)
        # second = 1565
        # count = 0
        # cap.set(cv2.CAP_PROP_POS_MSEC, second*1000)
        ret, frame = cap.read()
        frame_list = []
        # get image frame:
        while ret:
            ret, frame = cap.read()
            frame_list.append(frame)
        cap.release()
    # save video
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out_video = cv2.VideoWriter(
        f'../video/{partial}.mp4', fourcc, 30, (1280, 720), True)

    ############# video ################
    for frame in frame_list:
        if frame is None:
            break
    ############# camera ###############
    # cap = cv2.VideoCapture(0)
    # while True:
    #     ret, frame = cap.read()
    ####################################
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # face detect
        boxes, labels, probs = predictor.predict(
            image, candidate_size / 2, threshold)
        if boxes.size(0) == 0:
            continue
        box = boxes[0, :]
        ratio = 0.05
        face_img = image[int(box[1]*(1-ratio)):int(box[3]*(1+ratio)),
                         int(box[0]*(1-ratio)):int(box[2]*(1+ratio))]
        if face_img is None:
            continue
        image = cv2.resize(face_img, (input_size, input_size))
        # resize for face parsing
        if not mnn:
            parsing, cost_once = torch_inference(net, image)
        else:
            parsing, cost_once = mnn_inference(net_mnn, image, input_tensor, session, num_class)
        parsing = postprocessing(cv2.resize(face_img, (512, 512)), parsing)
        cost.append(cost_once)
        vis_img, vis_parsing_anno_color = vis_parsing_maps(
            cv2.resize(face_img, (512, 512)), parsing, stride=1, save_im=False)
        # reverse back to normal image
        cv2.imshow('face', vis_img)
        vis_img_recovered = mask_recover(frame, vis_parsing_anno_color, box, face_img, ratio)
        vis_img = cv2.resize(
            vis_img_recovered, (frame.shape[1], frame.shape[0]))
        # cv2.putText(vis_img, 'FPS {}'.format(str(round(1/np.mean(cost), 2))),
        #             (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('demo', vis_img)
        out_video.write(vis_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        print('avg cost: {}s'.format(np.mean(cost)))

    # out_video.release()
    cv2.destroyAllWindows()


def img_inference(args):
    num_class = 7

    ###############  MNN cfg  ##################
    mnn_model = args.mnn_model
    net_mnn = MNN.Interpreter(mnn_model)
    config = {}
    config['precision'] = 'low'
    session = net_mnn.createSession()
    input_tensor = net_mnn.getSessionInput(session)

    image = np.loadtxt('/Users/markson/Desktop/input.txt', delimiter=',')
    image = image.reshape(256, 256, 3)
    parsing, _ = mnn_inference(net_mnn, image, input_tensor, session, num_class, False)
    parsing2txt(parsing)
    # vis_img, _ = vis_parsing_maps(np.zeros((512, 512, 3)), parsing)
    # cv2.imshow('face', vis_img)
    # cv2.waitKey(1)
    # cv2.destroyAllWindows()


def postprocessing(img, parsing):
    eye_mat = img.copy()
    eye_mat = cv2.cvtColor(eye_mat, cv2.COLOR_RGB2GRAY)
    eye_mat[parsing != 3] = 0
    ret, thres = cv2.threshold(eye_mat, 127, 255, cv2.THRESH_BINARY)
    parsing[thres == 255] = 8
    return parsing


def torch_inference(net, image):
    img = to_tensor(image)
    img = torch.unsqueeze(img, 0)
    start = time()
    out = net(img)
    cost_once = time()-start
    preds = out['masks']
    if 'fine' in out:
        preds = out['fine']
    preds = F.interpolate(preds,
                          size=[512, 512],
                          mode='bilinear',
                          align_corners=False)
    parsing = preds.squeeze(0).detach().numpy().argmax(0)
    return parsing, cost_once


def mnn_inference(net, image, input_tensor, session, num_class, raw=True):
    # resize to mobile_net tensor size
    input_size = image.shape[0]
    if raw:
        image = image / 255  # totensor
        image = image - (0.485, 0.456, 0.406)
        image = image / (0.229, 0.224, 0.225)
        # preprocess it
    image = image.transpose((2, 0, 1))
    # change numpy data type as np.float32 to match tensor's format
    image = image.astype(np.float32)
    # cv2 read shape is NHWC, Tensor's need is NCHW,transpose it
    start = time()
    tmp_input = MNN.Tensor((1, 3, input_size, input_size), MNN.Halide_Type_Float,
                           image, MNN.Tensor_DimensionType_Caffe)
    input_tensor.copyFrom(tmp_input)
    net.runSession(session)
    output_tensor = net.getSessionOutputAll(session)
    cost_once = time() - start
    last_tensor = []
    last = []
    for k, v in output_tensor.items():
        # print(k, v.getShape())
        if v.getShape() != (1, num_class+1, input_size, input_size):
            continue
        last_tensor.append(v)
    # constuct a tmp tensor and copy/convert in case output_tensor is nc4hw4
    print('output len', len(last_tensor))
    for _ in range(len(last_tensor)):
        last.append(MNN.Tensor((1, num_class+1, input_size, input_size),
                               MNN.Halide_Type_Float,
                               np.ones([1, num_class+1, input_size, input_size]).astype(np.float32),
                               MNN.Tensor_DimensionType_Caffe))
    for i, tensor in enumerate(last_tensor):
        tensor.copyToHostTensor(last[i])
    mat_np = []
    for mat in last:
        mat_np.append(mat.getData())
    last_fuse = np.mean(np.concatenate(mat_np), axis=0).transpose(1, 2, 0)
    # upsample to 512ex
    last_fuse = cv2.resize(last_fuse, (512, 512), interpolation=cv2.INTER_CUBIC)
    last_fuse = last_fuse.argmax(2)
    print('output shape', last_fuse.shape)
    return last_fuse, cost_once


def parsing2txt(parsing):
    txt = ''
    for x in range(parsing.shape[0]):
        for y in range(parsing.shape[1]):
            txt += str(parsing[x, y])
        txt += '\n'

    with open('/Users/markson/Desktop/imshow.txt', 'w+') as f:
        f.write(txt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pytorch2onnx')
    parser.add_argument(
        '--mnn_model',
        default="mnn/mnn_model/tiny_dlv3_c7_256_dfcl_best.mnn")
    parser.add_argument(
        '--input_size',
        type=int,
        default=256)
    args = parser.parse_args()
    cam_inference(args, True)
    # img_inference(args)
