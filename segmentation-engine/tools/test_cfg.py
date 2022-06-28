from torchstat import stat
import torch
import sys
sys.path.append('..')
from seg_engine import build_model, build_optimizer, build_loss  # noqa: E402
from seg_engine.config import get_cfg, get_outname  # noqa: E402
from face_dataset import FaceMask  # noqa: E402
from portrait_dataset import PortraitMatting, PortraitSegmentation  # noqa: E402
from sky_dataset import SkyParsing   # noqa: E402


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


def test_model():
    cfg = setup()
    print(cfg)
    model = build_model(cfg)
    model.eval()
    x = torch.randn(12, 4 if cfg.DATASET.POSTPROCESS.four_channel else 3,
                    cfg.INPUT.SIZE[0], cfg.INPUT.SIZE[1])
    print('='*40)
    print('input shape: {}'.format(x.shape))
    stat(model, (4 if cfg.DATASET.POSTPROCESS.four_channel else 3,
         cfg.INPUT.SIZE[0], cfg.INPUT.SIZE[1]))
    print('='*40)
    with torch.no_grad():
        out = model(x)
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                print('{} shape: {}'.format(k, v.shape))
            else:
                for i, out in enumerate(v):
                    print('feature {} shape: {}'.format(i, out.shape))


def test_loss():
    cfg = setup()
    loss = build_loss(cfg)
    model = build_model(cfg)
    x = torch.rand(
        [16, 4 if cfg.DATASET.POSTPROCESS.four_channel else 3, 256, 256], dtype=torch.float32)
    y = torch.randint(7, (16, 256, 256))
    with torch.no_grad():
        model.eval()
        res = model(x)
        for k, v in res.items():
            if isinstance(v, torch.Tensor):
                print('{} shape: {}'.format(k, v.shape))
            else:
                for i, out in enumerate(v):
                    print('feature {} shape: {}'.format(i, out.shape))
    print(loss)
    print(res.keys())
    print(loss(res, y))


def test_dataloader():
    from seg_engine.dataset.transform import MotionBlur
    cfg = setup()
    trainset = eval(cfg.DATASET.CLASSFUNC)(cfg, mode='train')
    testset = eval(cfg.DATASET.CLASSFUNC)(cfg, mode='val')
    print(trainset)
    print(testset)
    img_tensors, mask_tensors = [], []
    for i in range(10):
        outs = trainset.__getitem__(i)
        img, phas = outs['img'], outs['label']

        img_tensors.append(img)
        mask_tensors.append(phas)
    img_tensors = torch.stack(img_tensors)
    mask_tensors = torch.stack(mask_tensors)
    print(img_tensors.shape, mask_tensors.shape)
    vis_tensor(img_tensors, mask_tensors)


def vis_tensor(im_tensors, mask_tensors, col=2):
    import cv2
    import numpy as np
    img_mask_vstack = None
    img_mask_hstack = None
    for i, (img, mask) in enumerate(zip(im_tensors, mask_tensors)):
        img = img.detach().numpy().transpose((1, 2, 0))[:, :, ::-1] * 255.
        mask = mask.detach().squeeze(0).numpy() * 255.
        # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask = cv2.merge((mask, mask, mask))
        img = cv2.resize(img, (mask.shape[1], mask.shape[0]))
        img_mask = np.hstack((img, mask))
        img_mask = cv2.resize(img_mask, (1024, 512))
        if i % col != 0 and i != 0:
            img_mask_hstack = np.hstack((img_mask_hstack, img_mask))
            if isinstance(img_mask_vstack, np.ndarray):
                img_mask_vstack = np.vstack((img_mask_vstack, img_mask_hstack))
            else:
                img_mask_vstack = img_mask_hstack.copy()
            img_mask_hstack = None
        else:
            img_mask_hstack = img_mask
    cv2.imwrite('vis.jpg', img_mask_vstack)


if __name__ == '__main__':
    # test_loss()
    # test_model()
    test_dataloader()
