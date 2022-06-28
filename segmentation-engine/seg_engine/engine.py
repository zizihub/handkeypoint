from seg_engine.utils.utils import SmoothedValue, seed_reproducer, vis_parsing_maps
from seg_engine.utils.metrics import eval_metrics, eval_metrics_matting
from seg_engine.models.base import point_sample
from seg_engine.models.kd import KDModel
from seg_engine.optim.lr_scheduler import MultiStepLR
import os
import torch
from collections import defaultdict
import numpy as np
import cv2
import random
import torch.nn.functional as F


seed_reproducer(1212)


class Trainer:
    def __init__(self,
                 epoch=None,
                 net=None,
                 criterion=None,
                 optimizer=None,
                 lr_scheduler=None,
                 trainloader=None,
                 validloader=None,
                 train_metric_logger=None,
                 valid_metric_logger=None,
                 logger=None,
                 cfg=None):
        self.epoch = epoch
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.trainloader = trainloader
        self.validloader = validloader
        self.train_metric_logger = train_metric_logger
        self.valid_metric_logger = valid_metric_logger
        self.logger = logger
        self.cfg = cfg
        if 'Mat' in cfg.TASK:
            self.mode = 'matting'
            # * matting
            self.best_index = 'total'
            if 'best' not in self.valid_metric_logger.meters:
                self.valid_metric_logger.update(best=10e5)
            self.condition = "self.valid_metric_logger.meters[self.best_index].median < self.valid_metric_logger.meters['best'].min"
        else:
            self.mode = 'seg'
            # * segmentation
            self.best_index = 'mIoU'
            if 'best' not in self.valid_metric_logger.meters:
                self.valid_metric_logger.update(best=0.0)
            self.condition = "self.valid_metric_logger.meters[self.best_index].avg > self.valid_metric_logger.meters['best'].max"

    def fit(self, start_ep):
        for i in range(start_ep, self.epoch+1):
            # * multistep lrscheduler for epoch
            if isinstance(self.lr_scheduler, MultiStepLR):
                self.lr_scheduler.step(i)
            self.train(i)
            self.evaluate(i)
            if i % 5 == 0:
                self.result_visualize(i)
            self.save_checkpoint(i)

    def train(self, i):
        '''Train trainloader in every epoch'''
        self.net.train()
        self.train_metric_logger.add_meter('lr', SmoothedValue(
            window_size=1, fmt='{value:.6f}'))
        header = 'train epoch: [{}]'.format(i)
        loader_len = len(self.trainloader) - 1

        for batch_idx, inputs_tuple in enumerate(self.train_metric_logger.log_every(self.logger, self.trainloader, print_freq=100, header=header)):
            inputs, targets, kwargs = self.cuda_input(inputs_tuple, train=True)
            if isinstance(self.net, KDModel):
                # * Knowledge Distillation (Segmentation)
                output_dic = self.net.forward(inputs, targets)
                self.train_metric_logger.update(**output_dic)
            else:
                # * Normal Training
                loss = self.loss_compute(
                    inputs, targets, self.train_metric_logger, **kwargs)
                loss.backward()
            # * record learning rate
            self.train_metric_logger.update(lr=self.lr_scheduler.get_lr()[0])
            # * CosineAnnealingLR lr scheduler for iteration
            if self.lr_scheduler is not None and not isinstance(self.lr_scheduler, MultiStepLR):
                self.lr_scheduler.step()
            # * optimizer step
            if ((batch_idx+1) % self.cfg.SOLVER.ACC_STEP) == 0 or (batch_idx == loader_len):
                self.optimizer.step()
                self.optimizer.zero_grad()

    @torch.no_grad()
    def evaluate(self, i):
        '''Testing testloader in every epoch'''
        n_classes = self.validloader.dataset.n_classes+1
        classes_name = self.validloader.dataset.classes_name
        self.net.eval()
        header = 'Valid epoch: [{}]'.format(i)
        class_iou = defaultdict(list)

        for inputs_tuple in self.valid_metric_logger.log_every(self.logger, self.validloader, print_freq=100, header=header):
            inputs, targets, kwargs = self.cuda_input(
                inputs_tuple, train=False)
            class_scores = self.eval_report(inputs, targets, n_classes)
            if class_scores is not None:
                for i, iou in enumerate(class_scores):
                    class_iou[i].append(iou)

        for k, v in class_iou.items():
            self.logger.info('class {:<15} iou {:.5f}'.format(
                classes_name[k], np.mean(v)))

    def eval_report(self, inputs, targets, n_classes):
        """eval model when evaluting

        Args:
            inputs (torch.tensor): inputs
            targets (torch.tensor): targets
            n_classes (int): num of classes

        Returns:
            dict: classes scores dict
        """
        if self.mode == 'seg':
            # send batches into net
            if isinstance(self.net, KDModel):
                outs = self.net.predict(inputs)
            else:
                outs = self.net(inputs)

            preds = outs['masks']
            if 'fine' in outs:
                preds = outs['fine']
                if preds.shape[-1] != inputs.shape[-1]:
                    preds = F.interpolate(
                        preds, inputs.shape[-2:], mode='bilinear', align_corners=True)
            if isinstance(preds, tuple):
                preds = torch.mean(torch.stack(preds), dim=0)
            parsing = preds.argmax(1)
            (
                overall_acc,
                avg_per_class_acc,
                avg_jacc,
                avg_dice,
                class_scores
            ) = eval_metrics(targets, parsing, n_classes)
            self.valid_metric_logger.update(overall_acc=overall_acc,
                                            avg_per_class_acc=avg_per_class_acc,
                                            mIoU=avg_jacc,
                                            avg_dice=avg_dice,)
            return class_scores
        elif self.mode == 'matting':
            outs = self.net(inputs)
            pred_alpha = outs['alpha']
            (
                mad,
                mse,
                grad,
                conn,
            ) = eval_metrics_matting(targets, pred_alpha)
            self.valid_metric_logger.update(mad=mad,
                                            mse=mse,
                                            grad=grad,
                                            conn=conn,
                                            total=mad+mse+grad+conn,)
        else:
            raise NotImplementedError

    def loss_compute(self, inputs, targets, metric_logger, **kwargs):
        """loss compute when training

        Args:
            inputs (torch.tensor): CUDA type inputs
            targets (torch.tensor): CUDA type targets

        Returns:
            (torch.tensor): total loss for tasks
        """
        outs = self.net(inputs)
        if self.mode == 'seg' and (not isinstance(outs['masks'], tuple)):
            # * point rend output
            if outs['masks'].shape[-1] != inputs.shape[-1]:
                outs.update(masks=F.interpolate(outs['masks'],
                            inputs.shape[-2:], mode='bilinear', align_corners=True))
        # * seg loss compute
        seg_loss = self.criterion(outs, targets, **kwargs)
        if isinstance(seg_loss, dict):
            metric_logger.update(**seg_loss)
            loss = 0.0
            for loss_ in seg_loss.values():
                loss += loss_
        else:
            metric_logger.update(seg_loss=seg_loss)
            loss = seg_loss
        # * point-rend
        if 'rend' in outs:
            with torch.no_grad():
                gt_points = point_sample(targets.to(torch.float).unsqueeze(1),
                                         outs['points'],
                                         mode='nearest',
                                         align_corners=False).squeeze(1).to(torch.long)
            points_loss = F.cross_entropy(
                outs['rend'], gt_points, reduction='mean', ignore_index=255)
            loss += points_loss
            metric_logger.update(points_loss=points_loss)
        # * auxiliary loss
        if 'auxiliary' in outs:
            auxiliary_loss = self.criterion(outs['auxiliary'], targets)
            loss += 0.4*auxiliary_loss
            metric_logger.update(auxiliary_loss=auxiliary_loss*0.4)
        return loss

    @staticmethod
    def cuda_input(inputs_tuple, train=True):
        inputs = inputs_tuple['img'].cuda(non_blocking=True)  # to cuda
        targets = inputs_tuple['label']
        kwargs = {}
        if train:
            if 'edge_map' in inputs_tuple:
                edge_map = inputs_tuple['edge_map'].cuda(non_blocking=True)
                kwargs.update({'edge_map': edge_map})
            if 'grayscale' in inputs_tuple:
                grayscale = inputs_tuple['grayscale'].cuda(non_blocking=True)
                kwargs.update({'grayscale': grayscale})
        if 'foreground' in inputs_tuple:
            foreground = inputs_tuple['foreground'].cuda(non_blocking=True)
            kwargs.update({'foreground': foreground})
        if targets.dim() != 4:
            targets = targets.cuda(non_blocking=True).unsqueeze(1)
        else:
            targets = targets.cuda(non_blocking=True)
        return inputs, targets, kwargs

    def save_checkpoint(self, i):
        # Save checkpoint.
        if eval(self.condition):
            if self.mode == 'seg':
                # * segmentation
                self.logger.info(
                    f'######## saving best at {i} epoch | {self.best_index} {round(self.valid_metric_logger.meters["best"].max, 4)} ---> {round(self.valid_metric_logger.meters[self.best_index].avg, 4)} #########')
                cur_best_index = self.valid_metric_logger.meters[self.best_index].avg
            elif self.mode == 'matting':
                # * matting
                self.logger.info(
                    f'######## saving best at {i} epoch | {self.best_index} {round(self.valid_metric_logger.meters["best"].min, 4)} ---> {round(self.valid_metric_logger.meters[self.best_index].median, 4)} #########')
                cur_best_index = self.valid_metric_logger.meters[self.best_index].median
            else:
                raise NotImplementedError
            try:
                weight = self.net.module.state_dict() if hasattr(
                    self.net, 'module') else self.net.state_dict()
            except:
                weight = self.net.student_net.module.state_dict() if hasattr(
                    self.net, 'module') else self.net.student_net.state_dict()
            state = {
                'net': weight,
                'epoch': i,
                'best': cur_best_index
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(
                state, f'./log/{self.cfg.TASK}/{self.cfg.DATE}/{self.cfg.OUTPUT_NAME}/{self.cfg.OUTPUT_NAME}_best.pth')
            self.valid_metric_logger.update(
                best=self.valid_metric_logger.meters[self.best_index].avg)

    @torch.no_grad()
    def result_visualize(self, epoch):
        log_path = os.path.join('./log', self.cfg.TASK,
                                self.cfg.DATE, self.cfg.OUTPUT_NAME)
        vis_save_path = os.path.join(log_path, 'inter_vis')
        os.makedirs(vis_save_path, exist_ok=True)

        def tensor_to_numpy(tensor, px=255., mean=0., std=1.):
            tensor = tensor * std + mean
            return tensor.detach().cpu().numpy().transpose((1, 2, 0)) * px
        iterator = iter(self.validloader)
        inputs_tuple = iterator.next()
        inputs, targets, kwargs = self.cuda_input(inputs_tuple, train=False)
        # * KD Model visualization
        if isinstance(self.net, KDModel):
            outs = self.net.student_net(inputs)
        else:
            outs = self.net(inputs)
        vis_patch_vstack = None
        vis_patch_hstack = None
        if self.mode == 'matting':
            row = 8
            col = 2
            pred_fgrs, pred_phas = outs['foreground'], outs['alpha']
            true_fgrs, true_phas = kwargs['foreground'], targets
            regular_size = (pred_phas.shape[3], pred_phas.shape[2])
            for i, (pred_fgr, pred_pha, true_fgr, true_pha) in enumerate(zip(pred_fgrs, pred_phas, true_fgrs, true_phas)):
                # get numpy
                pred_fgr_np = tensor_to_numpy(
                    pred_fgr).astype(np.uint8)[:, :, ::-1]
                pred_pha_np = cv2.cvtColor(tensor_to_numpy(
                    pred_pha).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                true_fgr_np = tensor_to_numpy(
                    true_fgr).astype(np.uint8)[:, :, ::-1]
                true_pha_np = cv2.cvtColor(tensor_to_numpy(
                    true_pha).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                # resize
                pred_fgr_np = cv2.resize(pred_fgr_np, regular_size)
                pred_pha_np = cv2.resize(pred_pha_np, regular_size)
                true_fgr_np = cv2.resize(true_fgr_np, regular_size)
                true_pha_np = cv2.resize(true_pha_np, regular_size)

                vis_patch = np.hstack(
                    (pred_fgr_np, pred_pha_np, true_fgr_np, true_pha_np))
                if isinstance(vis_patch_vstack, np.ndarray):
                    vis_patch_vstack = np.vstack((vis_patch_vstack, vis_patch))
                else:
                    vis_patch_vstack = vis_patch.copy()
                if i % (row*col) == 0 and i != 0:
                    vis_patch_hstack = np.hstack(
                        (vis_patch_hstack, vis_patch_vstack))
                    break
                elif i % row == 0:
                    vis_patch_hstack = vis_patch_vstack.copy()
                    vis_patch_vstack = None
            cv2.imwrite(os.path.join(vis_save_path,
                        'epoch{}_vis.jpg'.format(epoch)), vis_patch_hstack)
        elif self.mode == 'seg':
            row = 8
            col = 2
            pred_masks = outs['masks']
            if isinstance(pred_masks, tuple):
                pred_masks = torch.mean(torch.stack(pred_masks), dim=0)
            pred_masks = pred_masks.argmax(dim=1)
            true_masks = targets.squeeze(1)
            imgs = inputs
            regular_size = (pred_masks.shape[2], pred_masks.shape[1])
            for i, (pred_mask, true_mask, img) in enumerate(zip(pred_masks, true_masks, imgs)):
                # get numpy
                img_np = tensor_to_numpy(
                    img, mean=0.5, std=0.5).astype(np.uint8)
                pred_mask_np, _ = vis_parsing_maps(
                    img_np, pred_mask.detach().cpu().numpy(), demo=True)
                true_mask_np, _ = vis_parsing_maps(
                    img_np, true_mask.detach().cpu().numpy(), demo=True)
                # resize
                pred_mask_np = cv2.resize(pred_mask_np, regular_size)
                true_mask_np = cv2.resize(true_mask_np, regular_size)
                img_np = cv2.resize(img_np, regular_size)[:, :, ::-1]
                vis_patch = np.hstack((img_np, pred_mask_np, true_mask_np))
                if isinstance(vis_patch_vstack, np.ndarray):
                    vis_patch_vstack = np.vstack((vis_patch_vstack, vis_patch))
                else:
                    vis_patch_vstack = vis_patch.copy()
                if i % (row*col) == 0 and i != 0:
                    vis_patch_hstack = np.hstack(
                        (vis_patch_hstack, vis_patch_vstack))
                    break
                elif i % row == 0:
                    vis_patch_hstack = vis_patch_vstack.copy()
                    vis_patch_vstack = None
            cv2.imwrite(os.path.join(vis_save_path,
                        'epoch{}_vis.jpg'.format(epoch)), vis_patch_hstack)

        else:
            raise NotImplementedError
