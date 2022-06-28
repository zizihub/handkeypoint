#!/usr/bin/python
# -*- encoding: utf-8 -*-
from reg_engine.utils.logger import setup_logger
from reg_engine.config import get_cfg, get_outname
from reg_engine.engine import HyperparameterSearcher
from reg_engine.dataset import *

from my_dataset import *
import os
import os.path as osp
import logging
import warnings
import shutil
import numpy as np
from ray import tune
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
    return cfg, my_cfg


def train_tfs():
    train_tfs_list = [
        'GaussNoise',
        'RandomBrightnessContrast',
        'ShiftScaleRotate',
        'HorizontalFlip',
        'RandomCrop'
    ]
    tfs = np.random.choice(train_tfs_list, size=np.random.randint(3, len(train_tfs_list)), replace=False).tolist()
    return tfs


def main(args):
    # basic info
    config, my_cfg = setup(args.cfg)
    logger = logging.getLogger()
    log_path = osp.join('./log', config.TASK, config.DATE, config.OUTPUT_NAME)
    os.makedirs(log_path, exist_ok=True)
    setup_logger(config, log_path)
    shutil.copy(my_cfg, osp.join(log_path, osp.basename(my_cfg)))
    logger.info(config)
    # ----------------------------------------------------------------
    #                      hyerparamater define
    # ----------------------------------------------------------------
    max_epoch = 20
    num_samples = 20

    search_space = {
        'lr': tune.loguniform(1e-4, 1e-2),
        'batch_size': tune.choice([128]),
        'criterions': tune.choice(['L1Loss']),
        'optimizer': tune.choice(['AdaBelief']),
        'data_aug': tune.sample_from(lambda _: train_tfs()),
    }
    metric = 'ma'
    maxmin = 'max'
    metric_columns = ['mae', 'mse', 'evs', 'r2', 'ma', 'training_iteration']
    hypersearch = HyperparameterSearcher(config,
                                         logger,
                                         max_epoch,
                                         num_samples,
                                         search_space,
                                         metric,
                                         maxmin,
                                         log_path,
                                         metric_columns)
    hypersearch.search()


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--cfg', type=str, default='./face_config.yaml', help='training config yaml')
    args = parse.parse_args()
    main(args)
