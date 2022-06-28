#!/usr/bin/python
# -*- encoding: utf-8 -*-

from seg_engine.utils.logger import setup_logger
from face_dataset import FaceMask
from portrait_dataset import PortraitMatting, PortraitSegmentation
from sky_dataset import SkyParsing
from test import test
from seg_engine.optim import build_optimizer, MultiStepLR
from seg_engine.loss import build_loss
from seg_engine.engine import Trainer
from seg_engine.utils import MetricLogger, SmoothedValue, SmoothedValueInference
from seg_engine.config import get_cfg, get_outname
from seg_engine.models import build_model

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from copy import deepcopy

import os
import os.path as osp
import logging
import warnings
import shutil

import argparse
warnings.filterwarnings('ignore')


def setup(my_cfg='./face_config.yaml'):
    '''
    Create configs and perform basic setups.
    '''
    cfg = get_cfg()
    cfg.merge_from_file(my_cfg)
    output_name = get_outname(cfg)
    cfg.merge_from_list(['OUTPUT_NAME', output_name])
    cfg.freeze()
    return cfg, my_cfg


def setup_optimizer(net, config):
    try:
        param_dict = [{'params': net.encoder.parameters(), 'lr': config.SOLVER.OPTIMIZER.LR*0.5},   # encoder
                      {'params': net.decoder.parameters(
                      ), 'lr': config.SOLVER.OPTIMIZER.LR},       # decoder
                      # head
                      {'params': net.head.parameters(
                      ), 'lr': config.SOLVER.OPTIMIZER.LR},
                      {'params': net.post.parameters(), 'lr': config.SOLVER.OPTIMIZER.LR}]          # post architecture
    except:
        param_dict = [
            {'params': net.parameters(), 'lr': config.SOLVER.OPTIMIZER.LR}]
    return param_dict


def main(args):
    # basic info
    config, my_cfg = setup(args.cfg)
    logger = logging.getLogger()
    log_path = osp.join('./log', config.TASK, config.DATE, config.OUTPUT_NAME)
    os.makedirs(log_path, exist_ok=True)
    setup_logger(config, log_path)
    shutil.copy(my_cfg, osp.join(log_path, osp.basename(my_cfg)))

    logger.info(config)
    # Average Meter
    metric_logger_train = MetricLogger(max_epoch=config.MAX_EPOCH,
                                       smooth_value=SmoothedValue,
                                       delimiter='  ')
    metric_logger_valid = MetricLogger(max_epoch=config.MAX_EPOCH,
                                       smooth_value=SmoothedValueInference,
                                       delimiter='  ')

    # model
    net = deepcopy(build_model(config)).cuda()
    # Resume
    if config.MODEL.WEIGHTS:
        ckpt = torch.load(config.MODEL.WEIGHTS, map_location='cuda')
        net.load_state_dict(ckpt['net'])
        start_ep = ckpt['epoch']
        metric_logger_valid.update(best=ckpt['best'])
        logger.info('>>>>>>>>>> start from: {}'.format(start_ep))
        logger.info(
            '>>>>>>>>>> previous ckpts best mIoU: {}'.format(ckpt['best']))
    else:
        start_ep = 0

    # criterion
    criterion = build_loss(config)
    # datasets
    trainset = eval(config.DATASET.CLASSFUNC)(config, mode='train')
    validset = eval(config.DATASET.CLASSFUNC)(config, mode='val')
    logger.info(trainset)
    logger.info(validset)

    # teacher net
    if config.MODEL.TEACHER_NET:
        from seg_engine.models.kd import KDModel
        config_teacher, _ = setup(os.path.join(
            config.MODEL.TEACHER_NET, 'face_config.yaml'))
        net_teacher = deepcopy(build_model(config_teacher)).cuda()
        if config.SOLVER.KD.MODE == 'DML':
            logger.info('>>>>>>>>>> Deep Mutual Learning is implemented!')
        else:
            for f in os.listdir(config.MODEL.TEACHER_NET):
                if not f.endswith('pth'):
                    continue
                ckpt = torch.load(os.path.join(
                    config.MODEL.TEACHER_NET, f), map_location='cuda')
            net_teacher.load_state_dict(ckpt['net'])
            logger.info(
                '>>>>>>>>>> teacher net ckpts best mIoU: {}'.format(ckpt['best']))
        net_kd = KDModel(net, net_teacher, criterion,
                         mode=config.SOLVER.KD.MODE)

        net = deepcopy(net_kd)

    if torch.cuda.device_count() > 1:
        torch.distributed.init_process_group(backend='nccl',                        # 通信后端
                                             # 任意localhost地址
                                             init_method='tcp://localhost:{}'.format(
                                                 args.port),
                                             rank=0,                                # 本机节点
                                             world_size=1)                          # 机器数量
        net = nn.parallel.DistributedDataParallel(
            net, find_unused_parameters=True)

    net.train()

    sampler = None
    trainloader = DataLoader(trainset,
                             sampler=sampler,
                             batch_size=config.SOLVER.BATCH_SIZE,
                             shuffle=(sampler is None),
                             num_workers=config.SOLVER.NUM_WORKERS,
                             pin_memory=True,
                             drop_last=True)
    validloader = DataLoader(validset,
                             batch_size=config.SOLVER.BATCH_SIZE,
                             shuffle=False,
                             num_workers=config.SOLVER.NUM_WORKERS,
                             pin_memory=True,
                             drop_last=False)
    # init optimizer
    if config.MODEL.TEACHER_NET:
        param_dict = setup_optimizer(net.student_net, config)
        # DML
        if config.SOLVER.KD.MODE == 'DML':
            param_dict.extend(setup_optimizer(net.teacher_net, config))
    else:
        param_dict = setup_optimizer(net, config)

    if hasattr(net, 'point_head') and hasattr(net.point_head, 'parameters'):
        param_dict.append({'params': net.point_head.parameters(),
                          'lr': config.SOLVER.OPTIMIZER.LR*0.1})
    optim, lrs = build_optimizer(config, param_dict, len(trainloader))

    # train loop
    trainer = Trainer(epoch=config.MAX_EPOCH,
                      net=net,
                      criterion=criterion,
                      optimizer=optim,
                      lr_scheduler=lrs,
                      trainloader=trainloader,
                      validloader=validloader,
                      train_metric_logger=metric_logger_train,
                      valid_metric_logger=metric_logger_valid,
                      logger=logger,
                      cfg=config
                      )
    trainer.fit(start_ep)


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--port', type=int, default=23456,
                       help='DDP backend port')
    parse.add_argument(
        '--cfg', type=str, default='./sky_config.yaml', help='training config yaml')
    args = parse.parse_args()
    main(args)
