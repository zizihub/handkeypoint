import sys
sys.path.append('..')  # noqa: E402
from reg_engine.config import get_cfg, get_outname
from reg_engine.models import EfficientNet
from reg_engine import build_model
import torch
from pytorch_grad_cam import GuidedBackpropReLUModel, GradCAMPlusPlus, GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image, deprocess_image
from PIL import Image
import numpy as np
import cv2
import os.path as osp


def setup():
    '''
    Create configs and perform basic setups.
    '''
    my_cfg = '../myconfig.yaml'
    cfg = get_cfg()
    cfg.merge_from_file(my_cfg)
    output_name = get_outname(cfg)
    cfg.merge_from_list(['OUTPUT_NAME', output_name])
    cfg.merge_from_list(['DEPLOY', True])
    cfg.merge_from_file(f'../log/{cfg.TASK}/{cfg.DATE}/{cfg.OUTPUT_NAME}/myconfig.yaml')
    cfg.freeze()
    return cfg


class CamVisOperator(object):
    def __init__(self):
        super(CamVisOperator).__init__()
        CAM = GradCAMPlusPlus
        cfg = setup()
        print('>>>>>>>>>>> Loading', cfg.OUTPUT_NAME)
        self.use_cuda = True if cfg.DEVICE == 'cuda' else False
        self.target_category = None
        ###########################################################################
        model = build_model(cfg)
        model.load_state_dict(torch.load(f'../log/{cfg.TASK}/{cfg.DATE}/{cfg.OUTPUT_NAME}/{cfg.OUTPUT_NAME}_best.pth',
                                         map_location='cuda')['net'])

        target_layer = model.backbone.layer4[-1]
        print('last layer: {}'.format(target_layer))
        if isinstance(model.backbone, EfficientNet):
            model.backbone.set_swish(memory_efficient=False)
        self.model = model
        self.cam = CAM(model=model,
                       target_layer=target_layer,
                       use_cuda=self.use_cuda)

    def op_gradcam(self, img_path):
        rgb_img = np.array(Image.open(img_path).convert('RGB'), dtype=np.float32) / 255.
        input_tensor = preprocess_image(rgb_img,
                                        mean=[0.5, 0.5, 0.5],
                                        std=[0.5, 0.5, 0.5])
        grayscale_cam = self.cam(input_tensor=input_tensor,
                                 target_category=self.target_category,
                                 aug_smooth=True,
                                 eigen_smooth=True)
        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

        gb_model = GuidedBackpropReLUModel(model=self.model, use_cuda=self.use_cuda)
        gb = gb_model(input_tensor, target_category=self.target_category)

        cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
        cam_gb = deprocess_image(cam_mask * gb)
        gb = deprocess_image(gb)

        ######## Output Save ########
        op_fnm = f'../vis/{osp.splitext(osp.basename(img_path))[0]}_gradcam++.jpg'
        cv2.imwrite(op_fnm, cam_image)
        print('### Done {}'.format(op_fnm))
        # cv2.imwrite(f'../vis/image_gb.jpg', gb)
        # cv2.imwrite(f'../vis/image_cam_gb.jpg', cam_gb)


if __name__ == '__main__':
    cvo = CamVisOperator()
    # import os
    # for root, dirs, files in os.walk('/data2/zhangziwei/datasets/HGR-dataset/homemade/test'):
    #     if dirs:
    #         continue
    #     for idx, f in enumerate(files):
    #         if idx % 100 != 1:
    #             continue
    #         img_path = osp.join(root, f)
    #         cvo.op_gradcam(img_path)
    cvo.op_gradcam('/data2/zhangziwei/datasets/HGR-dataset/version3/test/ok/ok_1_veEbB.jpg')
