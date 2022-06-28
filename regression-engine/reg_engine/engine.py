from bz2 import compress
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import albumentations as albu
import torch
from copy import deepcopy
from reg_engine.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, RandomSampler
from dataset import RegressionDataset, get_transform
from collections import defaultdict
from reg_engine import build_loss, build_model, build_optimizer, build_scheduler
from reg_engine.utils import SmoothedValue, seed_reproducer, mean_accuracy, mean_accuracy_log
from reg_engine.dataset import *
from reg_engine.models import Distiller
from reg_engine.utils import seed_reproducer, AUMCalculator
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
import datetime
# -------------------- Hyerparamater Search ---------------------------------
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler


seed_reproducer(1212)


class Trainer:
    def __init__(
        self,
        logger=None,
        kfold=None,
        epoch=None,
        train_metric_logger=None,
        valid_metric_logger=None,
        cfg=None,
        setup_func=None,
    ) -> None:
        self.logger = logger
        self.kfold = kfold
        self.epoch = epoch
        self.train_metric_logger = train_metric_logger
        self.valid_metric_logger = valid_metric_logger
        self.cfg = cfg
        self.setup_func = setup_func
        self.aum_calculator = AUMCalculator(
            os.path.abspath('log/{}/{}/{}'.format(cfg.TASK, cfg.DATE, cfg.OUTPUT_NAME)), compressed=False)

        self.teacher_net = []
        self.metric = "ma"

    def fit(self):
        for k in range(5 if self.cfg.KFOLD is True else 1):
            self.valid_metric_logger.update(best=0)
            if self.cfg.KFOLD is True:
                self.logger.info('>'*27 + ' {} fold training '.format(k))
            self.get_data(k)
            ep = self.get_model(self.cfg, k, teacher=False)
            self.get_parameters(self.net)
            # knowledge distillation
            if self.cfg.MODEL.TEACHERNET:
                for teacher_cfg in self.cfg.MODEL.TEACHERNET:
                    cfg_teacher, _ = self.setup_func(my_cfg=teacher_cfg)
                    _ = self.get_model(cfg_teacher, k, teacher=True)
                self.net = Distiller(teacher_nets=self.teacher_net,
                                     student_net=self.net,
                                     kd_loss=self.criterion,
                                     config=self.cfg)
            # DML
            if 'KnowledgeDistillationLoss' in self.cfg.SOLVER.LOSS.NAME and self.cfg.SOLVER.LOSS.MODE == 'DML':
                self.net = Distiller(teacher_nets=self.net,
                                     student_net=deepcopy(self.net),
                                     kd_loss=self.criterion,
                                     config=self.cfg)
            # avoid warning message
            self.optimizer.zero_grad()
            self.optimizer.step()
            for epoch in range(ep, self.cfg.MAX_EPOCH+1):
                # * multisteplrscheduler for epoch
                if isinstance(self.lr_scheduler, MultiStepLR):
                    self.lr_scheduler.step(epoch)
                self.train(epoch)
                self.evaluate(epoch)
                self.save_checkpoint(epoch, k)
                torch.cuda.synchronize()
            # clean cache
            del self.net, self.criterion, self.optimizer, self.lr_scheduler, self.trainloader, self.testloader, self.valid_metric_logger.meters
            torch.cuda.empty_cache()
        self.aum_calculator.finalize()

    def get_data(self, k_th):
        """get datasets and dataloaders"""
        train_dataset = eval(self.cfg.DATASET.FUNC)(
            mode=self.cfg.DATASET.TRAIN_MODE,
            transforms=get_transform(auto_aug=self.cfg.DATASET.AUGMENTATION.AUTO,
                                     train=True),
            config=self.cfg,
            k=k_th,
        )
        test_dataset = eval(self.cfg.DATASET.FUNC)(
            mode='valid',
            transforms=get_transform(auto_aug=False,
                                     train=False),
            config=self.cfg,
            k=k_th,
        )
        self.trainloader = DataLoader(train_dataset,
                                      batch_size=self.cfg.BATCH_SIZE,
                                      sampler=ImbalancedDatasetSampler(
                                          train_dataset) if self.cfg.DATASET.SAMPLER == 'Imbalanced' else None,
                                      shuffle=True if self.cfg.DATASET.SAMPLER == 'Random' else False,
                                      num_workers=self.cfg.NUM_WORKERS,
                                      pin_memory=True)
        self.testloader = DataLoader(test_dataset,
                                     batch_size=self.cfg.BATCH_SIZE,
                                     shuffle=False,
                                     num_workers=self.cfg.NUM_WORKERS,
                                     pin_memory=True)
        self.logger.info(train_dataset)
        self.logger.info(test_dataset)

    def get_model(self, cfg, k_th, teacher=False):
        """get targeted model"""
        print('get {} model done...'.format(cfg.OUTPUT_NAME))
        net = deepcopy(build_model(cfg))
        if cfg.MODEL.WEIGHTS:
            ckpts = torch.load(cfg.MODEL.WEIGHTS)
            if isinstance(ckpts, dict):
                if 'net' in ckpts.keys():
                    net.load_state_dict(ckpts['net'], strict=False)
                    # self.logger.info('>>>>>>>>>> previous ckpts best metric: {}'.format(ckpts[self.metric]))
                else:
                    net.load_state_dict(ckpts)
            else:
                net = ckpts
            # resume training
            if 0:
                ep = ckpts['epoch']+1  # start from next epoch
                # global best_metric
                self.valid_metric_logger.update(best=ckpts[self.metric])
                self.logger.info('>>>>>>>>>> start from: {}'.format(ep))
            else:
                ep = 1
        elif teacher:
            if cfg.KFOLD:
                checkpoint = torch.load('./log/{}/{}/{}/{}_{}fold_best.pth'.format(cfg.TASK, cfg.DATE,
                                        cfg.OUTPUT_NAME, cfg.OUTPUT_NAME, k_th))
            else:
                checkpoint = torch.load('./log/{}/{}/{}/{}_best.pth'.format(cfg.TASK, cfg.DATE,
                                        cfg.OUTPUT_NAME, cfg.OUTPUT_NAME))
            if 'net' in checkpoint.keys():
                net.load_state_dict(checkpoint['net'], strict=True)
            else:
                net.load_state_dict(checkpoint, strict=True)
            ep = 1
            # global best_metric
            # self.logger.info('>>>>>>>>>> previous ckpts best metric: {}'.format(checkpoint[self.metric]))
        else:
            ep = 1

        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
        net.to(self.cfg.DEVICE)
        if teacher:
            self.teacher_net.append(net)
        else:
            self.net = net
        del net
        return ep

    def get_parameters(self, net):
        """get loss and optimizer"""
        self.criterion = build_loss(cfg=self.cfg)
        self.optimizer = build_optimizer(
            cfg=self.cfg, net_params=net.parameters())
        self.lr_scheduler = build_scheduler(
            cfg=self.cfg, optim=self.optimizer, step_per_epoch=len(self.trainloader))

    def train(self, epoch):
        """Train one epoch"""
        # set net to train mode
        self.net.train()
        self.train_metric_logger.add_meter('lr', SmoothedValue(
            window_size=1, fmt='{value:.6f}'))
        header = 'train epoch: [{}]'.format(epoch)
        # init
        loader_len = len(self.trainloader)-1

        for batch_idx, batch_input in enumerate(self.train_metric_logger.log_every(self.logger, self.trainloader, print_freq=100, header=header)):
            inputs, targets, kwargs = self.cuda_input(batch_input)
            # * loss compute
            loss = self.compute_train_loss(inputs,
                                           targets,
                                           batch_input=batch_input,
                                           **kwargs)
            self.train_metric_logger.update(train_loss=loss)
            self.train_metric_logger.update(
                lr=self.optimizer.param_groups[0]['lr'])
            loss.backward()
            # * CosineAnnealingLR lr scheduler for iteration
            if not isinstance(self.lr_scheduler, MultiStepLR):
                self.lr_scheduler.step()
            # accumulate step for larger batch
            if ((batch_idx+1) % self.cfg.SOLVER.ACCUMULATE_STEPS) == 0 or (batch_idx == loader_len):
                # optimizer.step
                self.optimizer.step()
                self.optimizer.zero_grad()
        # * ReduceLROnPlateau for epoches
        # lr_scheduler.step(metrics=acc)

    @torch.no_grad()
    def evaluate(self, epoch):
        """Test one epoch"""
        # set net to eval mode
        self.net.eval()
        header = 'Valid epoch: [{}]'.format(epoch)
        test_loss = 0

        targets_dic = defaultdict(list)
        predict_dic = defaultdict(list)

        for batch_idx, batch_input in enumerate(self.valid_metric_logger.log_every(self.logger, self.testloader, print_freq=100, header=header)):
            inputs, targets, kwargs = self.cuda_input(batch_input)
            loss, outputs = self.compute_valid_loss(inputs, targets, **kwargs)

            # test-loss & output predicts
            test_loss += loss.item()
            targets_dic['default'].extend(targets.cpu().detach().tolist())
            predict_dic['default'].extend(outputs['ranking'].cpu().detach().tolist())
            # cal metrics
            mae = mean_absolute_error(
                targets_dic['default'], predict_dic['default'])
            mse = mean_squared_error(
                targets_dic['default'], predict_dic['default'])
            evs = explained_variance_score(
                targets_dic['default'], predict_dic['default'])
            r2 = r2_score(targets_dic['default'], predict_dic['default'])

            ma = mean_accuracy_log(
                targets_dic['default'], predict_dic['default'])

            self.valid_metric_logger.update(
                **{"mae": mae, "mse": mse, "evs": evs, "r2": r2, "ma": ma})
            self.valid_metric_logger.update(valid_loss=loss)

    def compute_train_loss(self, inputs, targets, **kwargs):
        if self.cfg.DATASET.AUGMENTATION.NAME == 'mixup':
            inputs, y_a, y_b, lam = mixup_data(inputs, targets)
        elif self.cfg.DATASET.AUGMENTATION.NAME == 'cutmix':
            inputs, y_a, y_b, lam = cutmix_data(inputs, targets)
        # KD will use distiller for loss compute
        if 'KnowledgeDistillationLoss' in self.cfg.SOLVER.LOSS.NAME:
            outputs, feature, loss_dic = self.net(
                inputs, **kwargs['batch_input'])
        else:
            outputs, feature = self.net(inputs)
            if 'dist_label' in outputs:
                self.aum_calculator.update(outputs['dist_label'], kwargs['dist_label'].argmax(
                    1), kwargs['batch_input']['image_path'])
            loss_dic = {}
            for loss_name, loss_ in self.criterion.items():
                loss_tmp = loss_(outputs, targets, **kwargs)
                loss_dic.update(**{loss_name: loss_tmp})
            # print('train_loss',loss_dic)
        # * multi-task network back propergate with different weights
        loss = 0
        for (loss_name, loss_), weight in zip(loss_dic.items(), self.cfg.SOLVER.LOSS.WEIGHTS):
            self.train_metric_logger.update(**{loss_name: loss_})
            loss += weight * loss_
        loss = loss / self.cfg.SOLVER.ACCUMULATE_STEPS
        return loss

    def compute_valid_loss(self, inputs, targets, **kwargs):
        outputs, feature = self.net(inputs)  # send batches into net
        loss_dic = {}
        for loss_name, loss_ in self.criterion.items():
            if loss_name == 'KnowledgeDistillationLoss':
                loss_ = F.mse_loss
            loss_tmp = loss_(outputs, targets, **kwargs)
            loss_dic.update(**{loss_name: loss_tmp})
        # print('valid_loss',loss_dic)
        # multi-task loss
        loss = 0
        for (loss_name, loss_), weight in zip(loss_dic.items(), self.cfg.SOLVER.LOSS.WEIGHTS):
            self.valid_metric_logger.update(**{loss_name: loss_})
            loss += weight * loss_
        return loss, outputs

    @staticmethod
    def cuda_input(inputs_tuple):
        kwargs = {}
        inputs = inputs_tuple['inputs'].cuda()  # to cuda
        targets = inputs_tuple['targets'].cuda()
        if 'dist_label' in inputs_tuple:
            dist_label = inputs_tuple['dist_label'].cuda()
            kwargs.update({'dist_label': dist_label})
        return inputs, targets, kwargs

    def save_checkpoint(self, epoch, k_th):
        # Save checkpoint.
        kfold = f'_{k_th}fold' if self.cfg.KFOLD else ''
        if self.valid_metric_logger.meters[self.metric].avg > self.valid_metric_logger.meters['best'].max:
            self.logger.info(
                f'############### saving best at {epoch} epoch...  {round(self.valid_metric_logger.meters["best"].max, 4)} ---> {round(self.valid_metric_logger.meters[self.metric].avg, 4)} ###############')
            if torch.cuda.device_count() == 1:
                state = {
                    'net': self.net.state_dict() if not hasattr(self.net, 's_net') else self.net.s_net.state_dict(),
                }
            else:
                state = {
                    'net': self.net.module.state_dict() if not hasattr(self.net, 's_net') else self.net.s_net.module.state_dict(),
                }
            state.update({
                self.metric: self.valid_metric_logger.meters[self.metric].avg,
                'epoch': epoch,
            })
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')

            torch.save(
                state, f'./log/{self.cfg.TASK}/{self.cfg.DATE}/{self.cfg.OUTPUT_NAME}/{self.cfg.OUTPUT_NAME}{kfold}_best.pth')
            self.valid_metric_logger.update(
                best=self.valid_metric_logger.meters[self.metric].avg)


class Tester:
    def __init__(self,
                 logger,
                 inference_path,
                 cfg,):
        self.logger = logger
        self.inference_path = inference_path
        self.cfg = cfg
        self.mae_list = []
        self.mse_list = []
        self.evs_list = []
        self.r2_list = []
        self.ma_list = []
        self.threshold = []
        self.metric = 'ma'

    @staticmethod
    def net_forward(net, testloader):
        net.to('cuda')
        net.eval()
        predict_list = []
        target_list = []
        with torch.no_grad():
            for batch_idx, batch_input in enumerate(tqdm(testloader)):
                inputs = batch_input['inputs']
                targets = batch_input['targets']
                inputs = inputs.to('cuda')
                outputs, _ = net(inputs)  # send batches into net
                predict_list.extend(outputs['ranking'].cpu().detach().tolist())
                target_list.extend(targets.cpu().detach().tolist())
        return predict_list, target_list

    def inference(self):
        start_from_begin = torch.cuda.Event(enable_timing=True)
        end_from_begin = torch.cuda.Event(enable_timing=True)
        start_from_begin.record()
        ####################################### model inference ####################################################
        for k_th in range(5 if self.cfg.KFOLD else 1):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            # load dataset
            test_dataset = eval(self.cfg.DATASET.FUNC)(mode='ued_puff',
                                                       transforms=get_transform(
                                                           train=False),
                                                       config=self.cfg,
                                                       k=k_th)
            self.logger.info(test_dataset)

            (
                mae,
                mse,
                evs,
                r2,
                ma, threshold
            ) = self.infer_one_fold(k_th, test_dataset)
            end.record()
            inference_time = start.elapsed_time(end)
            eta = datetime.timedelta(milliseconds=inference_time)
            for thres, m in zip(threshold, ma):
                self.logger.info(
                    'at threshold: {}  | Mean Acc: {}'.format(thres, m))

            self.logger.info(
                f"{k_th} fold Inference time: {eta} | valid MAE:{mae:7.4f} | MSE:{mse:7.4f} | EVS:{evs:7.4f} | R2:{r2:7.4f}| MA:{np.mean(ma):7.4f}")

        end_from_begin.record()
        inference_time_from_begin = start_from_begin.elapsed_time(
            end_from_begin)
        eta_from_begin = datetime.timedelta(
            milliseconds=inference_time_from_begin)
        if self.cfg.KFOLD:
            self.logger.info(
                f"5 fold Inference time: {eta_from_begin} | \
                valid MAE:{np.mean(self.mae_list):7.4f}({np.std(self.mae_list):0.4f}) | \
                MSE:{np.mean(self.mse_list):7.4f}({np.std(self.mse_list):0.4f}) | \
                EVS:{np.mean(self.evs_list):7.4f}({np.std(self.evs_list):0.4}) | \
                R2:{np.mean(self.r2_list):7.4f}({np.std(self.r2_list):0.4}| \
                MA:{np.mean(self.ma_list):7.4f}({np.std(self.ma_list):0.4})"
            )

    def infer_one_fold(self, k_th, test_dataset):
        # last vs. best
        net = build_model(self.cfg)

        if self.cfg.MODEL.WEIGHTS:
            ckpts = torch.load(self.cfg.MODEL.WEIGHTS)['net']
        else:
            if self.cfg.KFOLD:
                ckpts = torch.load(
                    f'./log/{self.cfg.TASK}/{self.cfg.DATE}/{self.cfg.OUTPUT_NAME}/{self.cfg.OUTPUT_NAME}_{k_th}fold_best.pth')
            else:
                ckpts = torch.load(
                    f'./log/{self.cfg.TASK}/{self.cfg.DATE}/{self.cfg.OUTPUT_NAME}/{self.cfg.OUTPUT_NAME}_best.pth')
        if 'net' in ckpts.keys():
            net.load_state_dict(ckpts['net'], strict=False)
            self.logger.info('>>>>>>>>>> previous ckpts best: {}'.format(ckpts[self.metric]))
        else:
            net.load_state_dict(ckpts, strict=False)
        if 'RepVGG' in self.cfg.MODEL.NAME:
            from reg_engine.models.repvgg import repvgg_model_convert
            net = repvgg_model_convert(net)
            self.logger.info('>>> RepVGG model converted...')
        # create dataloader
        testloader = DataLoader(
            test_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)
        predict_list, target_list = self.net_forward(net, testloader)

        # print("predict_list,target_list:",predict_list,target_list)

        # clean model cache
        del net
        torch.cuda.empty_cache()
        _, target = test_dataset.get_target()
        ########################################## regression report ######################################################
        print(np.array(predict_list).shape, np.array(target_list).shape)
        mae = mean_absolute_error(predict_list, target_list)
        mse = mean_squared_error(predict_list, target_list)
        evs = explained_variance_score(predict_list, target_list)
        r2 = r2_score(predict_list, target_list)
        ma, threshold = mean_accuracy(predict_list, target_list)

        self.mae_list.append(mae)
        self.mse_list.append(mse)
        self.evs_list.append(evs)
        self.r2_list.append(r2)
        self.ma_list.append(ma)
        self.threshold.append(threshold)
        return mae, mse, evs, r2, ma, threshold

# ------------------------------------------------------------
#                    HyperParameterSearch
# ------------------------------------------------------------


class HyperparameterSearcher(Trainer, Tester):

    def __init__(self,
                 cfg,
                 logger,
                 max_epoch,
                 num_samples,
                 search_space,
                 metric,
                 maxmin,
                 log_path,
                 metric_columns):
        self.cfg = cfg
        self.search_space = search_space
        self.max_epoch = max_epoch
        self.cpus = cfg.NUM_WORKERS
        self.gpus = torch.cuda.device_count()
        self.num_samples = num_samples
        self.metric_columns = metric_columns
        self.metric = metric
        self.maxmin = maxmin
        self.logger = logger
        self.data_dir = log_path

    def setup_optimizer(self, net, lr):
        param_dict = [{'params': filter(lambda p: p.requires_grad, net.parameters()),
                       'lr': lr}]
        return param_dict

    def search(self):
        scheduler = ASHAScheduler(
            metric=self.metric,
            mode=self.maxmin,
            max_t=self.max_epoch,
            grace_period=1,
            reduction_factor=2)
        reporter = CLIReporter(
            metric_columns=self.metric_columns)
        result = tune.run(
            self.fit,
            resources_per_trial={"cpu": self.cpus, "gpu": self.gpus},
            config=self.search_space,
            num_samples=self.num_samples,
            scheduler=scheduler,
            progress_reporter=reporter,
            local_dir=self.data_dir)
        best_trial = result.get_best_trial(self.metric, 'max', "last")
        self.logger.info("Best trial config: {}".format(best_trial.config))

    def fit(self, config, checkpoint_dir=None):
        net = deepcopy(build_model(self.cfg))
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
            if torch.cuda.device_count() > 1:
                net = nn.DataParallel(net)
        net.to(device)
        if self.cfg.MODEL.WEIGHTS.endswith('pth'):
            ckpts = torch.load(self.cfg.MODEL.WEIGHTS, map_location='cuda')
            if 'net' in ckpts.keys():
                net.load_state_dict(ckpts['net'], strict=False)
                self.logger.info('>>>>>>>>>> previous ckpts best: {}'.format(ckpts[self.metric]))
            else:
                net.load_state_dict(ckpts, strict=False)
        # loss
        if 'criterion' in config:
            criterions = build_loss(self.cfg, config['criterions'])
        else:
            criterions = build_loss(self.cfg)
        # load dataset
        trainset = eval(self.cfg.DATASET.FUNC)('train', get_transform(tfs=config['data_aug']), self.cfg)
        testset = eval(self.cfg.DATASET.FUNC)('valid', get_transform(), self.cfg)
        # load dataloader
        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=int(self.cfg.BATCH_SIZE if not "batch_size" in config else config["batch_size"]),
            shuffle=True,
            num_workers=8)
        valloader = torch.utils.data.DataLoader(
            testset,
            batch_size=int(self.cfg.BATCH_SIZE if not "batch_size" in config else config["batch_size"]),
            shuffle=True,
            num_workers=8)
        # load optimizer
        param_dict = self.setup_optimizer(net, config['lr'])
        optimizer = build_optimizer(self.cfg, param_dict)
        if checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(checkpoint_dir, "checkpoint"))
            net.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)
        # train
        self.train(net, trainloader, valloader, optimizer, criterions)

    def train(self, net, trainloader, valloader, optimizer, criterions):
        for epoch in range(self.max_epoch):  # loop over the dataset multiple times
            running_loss = 0.0
            epoch_steps = 0
            for idx, inputs_tuple in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, targets, kwargs = self.cuda_input(inputs_tuple)

                # zero the parameter gradients
                optimizer.zero_grad()
                loss = self.loss_compute(net, criterions, inputs, targets, **kwargs)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                epoch_steps += 1

            # Validation loss
            result = defaultdict(float)
            predict_list, target_list = self.net_forward(net, valloader)
            mae = mean_absolute_error(predict_list, target_list)
            mse = mean_squared_error(predict_list, target_list)
            evs = explained_variance_score(predict_list, target_list)
            r2 = r2_score(predict_list, target_list)
            ma = mean_accuracy_log(target_list, predict_list)
            result = {
                'mae': mae,
                'mse': mse,
                'evs': evs,
                'r2': r2,
                'ma': ma,
            }
            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((net.state_dict(), optimizer.state_dict()), path)
            tune.report(**result)
        print("Finished Training")
        # clean cache
        # del net, criterions, optimizer, trainloader, valloader
        # torch.cuda.empty_cache()

    def loss_compute(self, net, criterion, inputs, targets, **kwargs):
        outputs, _ = net(inputs)
        loss_dic = {}
        for loss_name, loss_ in criterion.items():
            loss_tmp = loss_(outputs, targets, **kwargs)
            loss_dic.update(**{loss_name: loss_tmp})
        # * multi-task network back propergate with different weights
        loss = 0
        for (loss_name, loss_), weight in zip(loss_dic.items(), self.cfg.SOLVER.LOSS.WEIGHTS):
            loss += weight * loss_
        loss = loss / self.cfg.SOLVER.ACCUMULATE_STEPS
        return loss

    def test(self, best_trial):
        best_trained_model = deepcopy(build_model(self.cfg))
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
            if self.gpus > 1:
                best_trained_model = nn.DataParallel(best_trained_model)
        best_trained_model.to(device)

        best_checkpoint_dir = best_trial.checkpoint.value
        model_state, optimizer_state = torch.load(os.path.join(
            best_checkpoint_dir, "checkpoint"))
        best_trained_model.load_state_dict(model_state)
        # test loss
        result = defaultdict(float)
        test_steps = 0
        testloader = torch.utils.data.DataLoader(self.testset, batch_size=16, shuffle=False, num_workers=8)
        n_classes = testloader.dataset.n_classes+1
        for idx, inputs_tuple in enumerate(testloader, 0):
            with torch.no_grad():
                test_steps += 1
                inputs, targets = self.cuda_input(inputs_tuple)
                result = self.eval_report(best_trained_model, inputs, targets, n_classes, result)
        for k, v in result.items():
            self.logger.info("Best {}: {}".format(k, v/test_steps))
