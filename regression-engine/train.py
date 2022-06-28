from reg_engine.engine import Trainer
from reg_engine.config import get_cfg, get_outname
from reg_engine.utils.logger import setup_logger
from reg_engine.utils import MetricLogger, seed_reproducer
import logging
import os
import shutil
import warnings

warnings.filterwarnings('ignore')
seed_reproducer(1212)


def setup(my_cfg='./myconfig.yaml'):
    '''
    Create configs and perform basic setups.
    '''
    cfg = get_cfg()
    cfg.merge_from_file(my_cfg)
    output_name = get_outname(cfg)
    cfg.merge_from_list(['OUTPUT_NAME', output_name])
    cfg.freeze()
    return cfg, my_cfg


def main(args):
    config, my_cfg = setup(args.cfg)
    log_path = './log/{}/{}'.format(config.TASK, config.DATE)
    os.makedirs(log_path, exist_ok=True)
    logger = logging.getLogger()
    log_path = os.path.join(log_path, config.OUTPUT_NAME)
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    setup_logger(config, log_path)
    shutil.copy(my_cfg, os.path.join(log_path, os.path.basename(my_cfg)))

    logger.info(config)

    metric_logger_train = MetricLogger(max_epoch=config.MAX_EPOCH, delimiter='  ')
    metric_logger_valid = MetricLogger(max_epoch=config.MAX_EPOCH, delimiter='  ')

    trainer = Trainer(logger=logger,
                      kfold=config.KFOLD,
                      epoch=config.MAX_EPOCH,
                      train_metric_logger=metric_logger_train,
                      valid_metric_logger=metric_logger_valid,
                      cfg=config,
                      setup_func=setup)
    trainer.fit()


if __name__ == "__main__":
    import argparse
    parse = argparse.ArgumentParser()
    parse.add_argument('--cfg', type=str, default='./myconfig.yaml', help='training config yaml')
    args = parse.parse_args()
    main(args)
