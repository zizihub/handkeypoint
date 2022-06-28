# internal lib
from genericpath import exists
from dataset.my_dataset import RegressionDataset
from reg_engine.utils import seed_reproducer
# external lib
import logging
import os
from reg_engine.config import get_cfg, get_outname
from reg_engine.utils.logger import setup_logger
from reg_engine.engine import Tester


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
    return cfg


def inference(args):
    config = setup(args.cfg)
    # output path
    logger = logging.getLogger()
    log_path = './log/{}/{}/{}'.format(config.TASK, config.DATE, config.OUTPUT_NAME)
    os.makedirs(log_path, exist_ok=True)
    inference_path = os.path.join(log_path, 'inference')
    os.makedirs(inference_path, exist_ok=True)
    setup_logger(config, inference_path, inference=True)
    logger.info(config)
    tester = Tester(logger=logger,
                    inference_path=inference_path,
                    cfg=config)
    
    tester.inference()


if __name__ == "__main__":
    import argparse
    parse = argparse.ArgumentParser()
    parse.add_argument('--cfg', type=str, default='./myconfig.yaml', help='training config yaml')
    args = parse.parse_args()
    inference(args)
