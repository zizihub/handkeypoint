from numpy.lib.npyio import save
from torchstat import stat
import torch
import sys
import os.path as osp
from copy import deepcopy
import os
from tqdm import tqdm
import cv2
from PIL import Image, ImageOps
from time import time
import numpy as np
from torchvision.transforms.functional import to_tensor, normalize
sys.path.append('..')
from seg_engine import build_model  # noqa: E402
from seg_engine.config import get_cfg, get_outname  # noqa: E402
from seg_engine.utils import eval_metrics  # noqa: E402


def setup():
    '''
    Create configs and perform basic setups.
    '''
    my_cfg = './test.yaml'
    cfg = get_cfg()
    cfg.merge_from_file(my_cfg)
    output_name = get_outname(cfg)
    cfg.merge_from_list(['OUTPUT_NAME', output_name])
    cfg.freeze()
    return cfg


class Cleaner(object):
    def __init__(self):
        self.dspth = '/data2/zhangziwei/datasets/KHairstyle500k/image'
        self.num_classes = 2

    def load_model(self, config):
        if config is None:
            config, _ = setup()
        self.respth = osp.join('../log', config.TASK, config.DATE, config.OUTPUT_NAME, 'demo_fuse')
        os.makedirs(self.respth, exist_ok=True)
        net = deepcopy(build_model(config))
        net.cuda()
        save_pth = osp.join('../log', config.TASK, config.DATE, config.OUTPUT_NAME,
                            '{}_best.pth'.format(config.OUTPUT_NAME))
        print('loading', save_pth)
        net.load_state_dict(torch.load(save_pth)['net'])
        net.eval()
        self.net = net

    @torch.no_grad()
    def clean(self):
        config = setup()
        self.load_model(config)
        cost = []
        for i, image_path in enumerate(tqdm(os.listdir(self.dspth))):
            image = Image.open(osp.join(self.dspth, image_path)).resize((512, 512))
            mask = np.array(Image.open(osp.join(self.dspth, image_path)
                                       .replace('image', 'mask-hair')
                                       .replace('jpg', 'png'))
                            .convert('L'))
            start = time()
            pred_t = self.predict_one_image(image)
            cost.append(time() - start)
            mask_t = mask.astype(np.int64)[np.newaxis, :]
            (
                overall_acc,
                avg_per_class_acc,
                avg_jacc,
                avg_dice,
                class_scores
            ) = eval_metrics(mask_t, pred_t, self.num_classes)
            if class_scores[1] < 0.5:
                print(overall_acc.item(), avg_per_class_acc.item(), avg_jacc.item(), avg_dice.item(), class_scores)
                pred = pred_t.squeeze(0).cpu().detach().numpy()
                cv2.imwrite(osp.join(self.respth, image_path.replace('.jpg', '.png')), np.hstack([pred*100, mask*100]))

        print('avg cost: {}s'.format(np.mean(cost)))

    def predict_one_image(self, image):
        img = normalize(to_tensor(image), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        img = torch.unsqueeze(img, 0)
        img = img.cuda()
        out = self.net(img)
        preds = out['masks']
        if 'fine' in out:
            preds = out['fine']
        pred = preds.argmax(1)
        return pred


if __name__ == '__main__':
    cleaner = Cleaner()
    cleaner.clean()
