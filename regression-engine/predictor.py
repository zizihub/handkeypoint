from reg_engine import build_model
from reg_engine.config import get_cfg, get_outname
from sklearn.metrics import accuracy_score, classification_report
from collections import defaultdict
from scipy.special import softmax
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from copy import deepcopy
import numpy as np
from PIL import Image
import os
import torch
import pandas as pd
from reg_engine.utils import seed_reproducer
from dataset.my_dataset import CustomDataset
from dataset.my_dataset import CustomDataset, get_transform


seed_reproducer(1212)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def setup():
    '''
    Create configs and perform basic setups.
    '''
    cfg = get_cfg()
    cfg.merge_from_file('./myconfig.yaml')
    output_name = get_outname(cfg)
    print('>>>>>> loading {}'.format(output_name))
    cfg.merge_from_list(['OUTPUT_NAME', output_name])
    cfg.merge_from_file(f'./log/{cfg.TASK}/{cfg.DATE}/{output_name}/myconfig.yaml')
    cfg.freeze()
    return cfg, output_name


class PredictDataset(Dataset):
    def __init__(self, test_dir, im_list, transforms):
        self.test_dir = test_dir
        self.im_list = im_list
        self.transforms = transforms

    def __getitem__(self, index):
        image_src = f'{self.test_dir}/{self.im_list[index]}'
        img = Image.open(image_src).convert('RGB')
        im_tensor = self.transforms(img)
        return {'inputs': im_tensor}

    def __len__(self):
        return len(self.im_list)


class Predictor(object):
    def __init__(self):
        self.test_dir = '/data1/zhangziwei/datasets/ad-dataset/version7/others'
        self.im_list = os.listdir(self.test_dir)
        self.raw_result = []
        self.predict_name = []
        self.H = 360  # resolution height
        self.W = 240  # resolution width
        self.batch_size = 128  # predict batch size
        self.prob_threshold = 0.5  # pseudo-labelling threshold
        self.kfold = 0
        self.cfg, output_name = setup()
        self.csv_softmax = f'./log/{output_name}/r18_prob.csv'
        self.pseudo_csv_label = f'./log/{output_name}/r18_pseudolb.csv'
        # for oof predict
        self.fold_imlist = []
        self.fold_result = []
        self.fold_df = './result/kfold_b0.csv'

    def model_select(self, k=0):
        net = build_model(self.cfg)
        if self.cfg.MODEL.WEIGHTS:
            state = torch.load(self.cfg.MODEL.WEIGHTS)['net']
        else:
            if self.cfg.KFOLD:
                state = torch.load(
                    f'./log/{self.cfg.TASK}/{self.cfg.DATE}/{self.cfg.OUTPUT_NAME}/{self.cfg.OUTPUT_NAME}_{k}fold_best.pth')['net']
            else:
                state = torch.load(
                    f'./log/{self.cfg.TASK}/{self.cfg.DATE}/{self.cfg.OUTPUT_NAME}/{self.cfg.OUTPUT_NAME}_best.pth')['net']
        net.load_state_dict(state)
        net.to('cuda')
        net.eval()
        return net

    def predict_transform(self):
        tfs = [
            transforms.Resize((360, 240)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], inplace=True)
        ]
        return transforms.Compose(transforms=tfs)

    def predict_single_model(self, net, testloader, oof=False):
        predict_list = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm(testloader)):
                inputs = batch['inputs']
                inputs = inputs.to(DEVICE)  # to cuda
                outputs, _ = net(inputs)  # send batches into net
                predict_list.extend(outputs.cpu().detach().tolist())
            if oof:
                self.fold_result.extend(predict_list)
            else:
                self.raw_result.append(predict_list)

    def idx2name(self, x): return self.df.loc[x, 'image_id']

    def predict_oof(self, k):
        net = self.model_select(k)
        test_dataset = CustomDataset('test',
                                     transforms=get_transform(train=False),
                                     config=self.cfg,
                                     k=k)
        print(test_dataset)
        testloader = DataLoader(test_dataset, batch_size=self.batch_size,
                                shuffle=False, num_workers=8, pin_memory=True)
        self.fold_imlist.extend(test_dataset.data)
        self.predict_single_model(net, testloader, oof=True)
        # clean model cache
        del net
        torch.cuda.empty_cache()

    def predict_kfold(self):
        for i in range(5):
            self.predict_oof(i)
        # self.fold_result  [N,2,5]
        print(np.array(self.fold_result).shape)
        if len(np.array(self.fold_result).shape) == 3:
            result = np.argmax(np.array(self.fold_result)[:, 1, :], axis=1)
        else:
            result = np.argmax(np.array(self.fold_result), axis=1)
        print(result.shape)
        # check csv
        assert len(result) == len(
            self.fold_imlist), "shape doesn't match!! {} <--> {}".format(len(result), len(self.fold_imlist))
        # output csv
        df = pd.DataFrame(data=list(zip(self.fold_imlist, result)),
                          columns=['image_id', 'label'])
        df.to_csv(self.fold_df, index=False)

    def predict(self):
        # last vs. best
        net = self.model_select()
        test_dataset = PredictDataset(test_dir=self.test_dir,
                                      im_list=self.im_list,
                                      transforms=self.predict_transform())
        testloader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8, pin_memory=True)
        self.predict_single_model(net, testloader)
        # clean model cache
        del net
        torch.cuda.empty_cache()
        # processing raw output(fold, test_num, num_classes)
        raw_result_mean = np.mean(self.raw_result, axis=0)
        softmax_result_mean = softmax(raw_result_mean, axis=1)
        # softmax class output
        print('softmax output shape', np.array(self.raw_result).shape)
        print('softmax output shape after mean', raw_result_mean.shape)
        print('sample softmax output:', raw_result_mean[0])
        result = np.argmax(raw_result_mean, axis=1)
        print('softmax output shape after argmax', result.shape)

        assert len(result) == len(
            self.im_list), "shape doesn't match!! {} <--> {}".format(len(result), len(self.im_list))

        # to csv
        if len(softmax_result_mean.shape) == 3:
            self.oof_output = softmax_result_mean.transpose(0, 2, 1)
            df = pd.DataFrame(data=list(zip(self.im_list, softmax_result_mean.transpose(0, 2, 1))),
                              columns=['image_id', 'label'])
        else:
            self.oof_output = softmax_result_mean.transpose(1, 0)
            df = pd.DataFrame(data=list(zip(self.im_list, softmax_result_mean.transpose(1, 0))),
                              columns=['image_id', 'label'])
        df.to_csv(self.csv_softmax, index=False)

    def pseudo_labelling(self):
        df = pd.read_csv(self.csv_softmax)
        print('dataframe total nums:', len(df))
        classes = defaultdict(int)
        psd_op = []
        for i in range(len(df)):
            x = np.fromstring(df.loc[i, 'label'].replace('[', '').replace(']', ''),
                              dtype=np.float,
                              sep=' ')
            if np.max(x) > self.prob_threshold:
                classes[np.argmax(x)] += 1
                psd_op.append([df.loc[i, 'image_id'], np.argmax(x)])
        print(classes)
        print('pseudo-lb nums:', sum(classes.values()))
        psd_df = pd.DataFrame(data=psd_op, columns=['image_id', 'label'])
        psd_df.to_csv(self.pseudo_csv_label, index=False)
        print(psd_df.head(10))


if __name__ == "__main__":
    pred = Predictor()
    pred.predict()
