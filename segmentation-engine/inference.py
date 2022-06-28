#!/usr/bin/python
# -*- encoding: utf-8 -*-

from seg_engine.models.build import build_model
from seg_engine.utils.logger import setup_logger
from face_dataset import FaceMask
from portrait_dataset import PortraitMatting, PortraitSegmentation
from sky_dataset import SkyParsing
from seg_engine.engine import Trainer
from seg_engine.utils.utils import MetricLogger, SmoothedValueInference
from seg_engine.config import get_cfg, get_outname
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from copy import deepcopy

import os
import os.path as osp
import logging
import warnings
import argparse
warnings.filterwarnings('ignore')


def setup(args):
    '''
    Create configs and perform basic setups.
    '''
    cfg = get_cfg()
    cfg.merge_from_file(args.cfg)
    output_name = get_outname(cfg)
    cfg.merge_from_list(['OUTPUT_NAME', output_name])
    cfg.merge_from_list(['MODEL.PRETRAINED', ''])
    cfg.freeze()
    return cfg


def main(args):
    # basic info
    logger = logging.getLogger()
    config = setup(args)
    log_path = osp.join('./log', config.TASK, config.DATE,
                        config.OUTPUT_NAME, 'inference')
    os.makedirs(log_path, exist_ok=True)
    setup_logger(config, log_path, inference=True)

    logger.info(config)

    # datasets
    validset = eval(config.DATASET.CLASSFUNC)(config, mode='val')

    logger.info(validset)

    validloader = DataLoader(validset,
                             batch_size=16,
                             shuffle=False,
                             num_workers=config.SOLVER.NUM_WORKERS,
                             pin_memory=True,
                             drop_last=False)

    # model
    net = deepcopy(build_model(config))
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net.cuda()
    # AverageMeter
    metric_logger_valid = MetricLogger(max_epoch=config.MAX_EPOCH,
                                       smooth_value=SmoothedValueInference,
                                       delimiter='  ')
    metric_logger_valid.update(best=0.0)
    print('loading checkpoint...')
    ckpt = torch.load(
        f'./log/{config.TASK}/{config.DATE}/{config.OUTPUT_NAME}/{config.OUTPUT_NAME}_best.pth', map_location='cuda')
    net.load_state_dict(ckpt['net'])
    trainer = Trainer(net=net,
                      validloader=validloader,
                      valid_metric_logger=metric_logger_valid,
                      logger=logger,
                      cfg=config)
    trainer.evaluate(0)


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--cfg', type=str,
                       default='sky_config.yaml', help='training config yaml')
    args = parse.parse_args()
    main(args)
