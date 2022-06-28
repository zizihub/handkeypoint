# Segmentation Engine

### Introduction

A PyTorch version engine for image segmentation

[seg_engine](./seg_engine): library of this engine

[config](./seg_engine/config): configuration set up for the entire engine

[dataset](./seg_engine/dataset): dataset Base Class. All resize related function are using **OpenCV** to align with mobile device usage.

[loss](./seg_engine/loss):  segmentation loss. e.g. IoULoss, DiceLoss, CrossEntropyLoss, LovaszLoss, FocalLoss, etc.

[models](./seg_engine/models):  

1. [base](./seg_engine/models/base): FPN, PANet, BIFPN, PointRend, SCSE, modules
2. [deeplabv3](./seg_engine/models/deeplabv3): segmentation decoder BiSENet & DDRNet & DeepLabV3Plus.
3. [encoders](./seg_engine/models/encoders): Backbone Networks e.g. ResNet, MobilenetV3, GhostNet, etc.

[optim](./seg_engine/optim): optimizers and learning rate scheduler

[tools](./tools): test_cfg, preprocess_data, rename_checkpoint, crop_image_by_mask, model_convert_pt, etc.

### Bag of tricks
1. Auxiliary Loss at the second layer of backbone(P2). It will help the model to recover lower level feature information when compute the loss.
2. Depth Seperate Convolution to replace normal Convolution.
3. Different Lr for different part of model(encoder, decoder).
4. (Upcoming): Knowledge distillation for semantic segmentation.
5. (Upcoming): 4 Channels input to stablize segmentation performance and reduce model parameters in video segmentation.
6. (Upcoming): Conditional Random Field as post-processing

### How to use

1. set up your own dataset by refering [face_dataset.py](./face_dataset.py)

```python
class FaceMask(SegmentationDataset):
    def __init__(self, rootpth, n_classes, cropsize=(448, 448), mode='train', resize=(512, 512), *args, **kwargs):
        super(FaceMask, self).__init__(
            rootpth, n_classes, mode, resize, *args, **kwargs)
        # define classes dictionary
        self.classes_name = {
            0: 'background',
            1: 'facial skin',
            2: 'left brow',
            3: 'right brow',
            4: 'left eye',
            5: 'right eye',
            6: 'nose',
            7: 'upper lip',
            8: 'inner mouth',
            9: 'lower lip',
        }

        #  pre-processing
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.trans_train = Compose([
            ColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.5),
            HorizontalFlip(),
            RandomScale((0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
            RandomCrop(cropsize)
        ])




```
2. set up your own configuration by refering [face_config.yaml](./face_config.yaml) and [defaults.py](./seg_engine/config/defaults.py)

```yaml
DATASET:
  TRAIN_PATH: [
      "/data2/zhangziwei/datasets/CelebA-HQ-img",
      "/data2/zhangziwei/datasets/CelebA-HQ-Occlusion",
    ]
  TEST_PATH: [
      "/data2/zhangziwei/datasets/CelebA-HQ-img",
      "/data2/zhangziwei/datasets/CelebA-HQ-Occlusion",
    ]
  NUM_CLASSES: 6
  GRAYSCALE: False

MODEL:
  PRETRAINED: "imagenet"
  ENCODER:
    NAME: "DDRNet23_slim"
    OUTPUT_STRIDE: 16
  FPN:
    NAME: ""
    OUT_CHANNELS: 88
    PANET: False
    NORM: None
    STACK: 3
  DECODER:
    # NAME: "DeepLabV3Plus"
    NAME: "DeepDualResolutionNet"
    CHANNELS: 256
    ASTROUS_RATES: (3, 5, 7)
  POSTPROCESS:
    NAME: [""]
    FC_DIM: 512
    NUM_FC: 3
    REDUCTION: 16
  BLOCK: "BasicBlock"

# INPUT
INPUT:
  SIZE: (256, 256)
  CROP: (224, 224)

# SOLVER
SOLVER:
  OPTIMIZER:
    NAME: "AdaBelief"
    LR: 1e-3
    WARMUP_LR: 1e-5
  LR_SCHEDULER:
    NAME: "OneCycleLR"
  LOSS: "DC_and_Focal_loss"
  # LOSS: "JointEdgeSegLoss"
  NUM_WORKERS: 16
  BATCH_SIZE: 64
  ACC_STEP: 1
MAX_EPOCH: 100
DATE: "2021-09-22"
EXTRA: "occluded"
TASK: "Face-6"


```
3. Define your own training loop by refering [train.py](./train.py)
```python
def main():
    # basic info
  
    # datasets
    trainset = FaceMask()
    validset = FaceMask()
    testset = FaceMask()

    trainloader = DataLoaderX()
    validloader = DataLoaderX()
    testloader = DataLoaderX()

    # model
    net = deepcopy(build_model(config))
    criterion = build_loss(config)
    optim = build_optimizer(config, param_dict, len(trainset))

    # train loop
    for epoch in range(start_ep, config.SOLVER.MAX_EPOCH+1):
        train(logger, epoch, net, criterion, optim,
              trainloader, metric_logger_train)
        evaluate(config, logger, epoch, net, validloader,
                 metric_logger_valid)
    # final test on other dataset
    evaluate(logger, epoch, net, testloader,
             metric_logger_valid)
```

**Run Model**

```bash
CUDA_VISIBLE_DEVICES='1' python train.py
```

### Log & Metrics
```yaml
INFO utils.py(143): train epoch: [0]  [ 100/2945]  eta: 5:58:34  lr: 0.000021  seg_loss: 1.7625 (1.7672)  time: 0.1324  max mem: 835  @2021-04-07 14:08:17
INFO utils.py(143): train epoch: [0]  [ 200/2945]  eta: 5:28:02  lr: 0.000046  seg_loss: 1.6519 (1.6534)  time: 0.1231  max mem: 835  @2021-04-07 14:08:29
INFO utils.py(143): train epoch: [0]  [ 300/2945]  eta: 5:26:08  lr: 0.000098  seg_loss: 1.4752 (1.4765)  time: 0.1244  max mem: 835  @2021-04-07 14:08:42
INFO utils.py(143): train epoch: [0]  [ 400/2945]  eta: 5:10:14  lr: 0.000212  seg_loss: 1.2320 (1.2441)  time: 0.1261  max mem: 835  @2021-04-07 14:08:53
INFO utils.py(143): train epoch: [0]  [ 500/2945]  eta: 5:12:44  lr: 0.000457  seg_loss: 1.0780 (1.0771)  time: 0.1335  max mem: 835  @2021-04-07 14:09:06
INFO utils.py(143): train epoch: [0]  [ 600/2945]  eta: 5:08:58  lr: 0.000985  seg_loss: 0.9897 (0.9804)  time: 0.1144  max mem: 835  @2021-04-07 14:09:17
INFO optimizer.py(69): ==> warmup done, start to implement poly lr strategy
INFO utils.py(143): train epoch: [0]  [ 700/2945]  eta: 5:09:22  lr: 0.000999  seg_loss: 0.8119 (0.8173)  time: 0.1228  max mem: 835  @2021-04-07 14:09:30
INFO utils.py(143): train epoch: [0]  [ 800/2945]  eta: 5:09:20  lr: 0.000999  seg_loss: 0.6671 (0.6759)  time: 0.1255  max mem: 835  @2021-04-07 14:09:42
INFO utils.py(143): train epoch: [0]  [ 900/2945]  eta: 5:08:09  lr: 0.000998  seg_loss: 0.6161 (0.6191)  time: 0.1269  max mem: 835  @2021-04-07 14:09:55
INFO utils.py(143): train epoch: [0]  [1000/2945]  eta: 5:07:22  lr: 0.000998  seg_loss: 0.5608 (0.5710)  time: 0.1084  max mem: 835  @2021-04-07 14:10:07
INFO utils.py(143): train epoch: [0]  [1100/2945]  eta: 5:06:17  lr: 0.000997  seg_loss: 0.5177 (0.5322)  time: 0.1203  max mem: 835  @2021-04-07 14:10:19
INFO utils.py(143): train epoch: [0]  [1200/2945]  eta: 5:04:32  lr: 0.000996  seg_loss: 0.4993 (0.5210)  time: 0.1344  max mem: 835  @2021-04-07 14:10:30
INFO utils.py(143): train epoch: [0]  [1300/2945]  eta: 5:04:27  lr: 0.000996  seg_loss: 0.4669 (0.4760)  time: 0.1235  max mem: 835  @2021-04-07 14:10:43
INFO utils.py(143): train epoch: [0]  [1400/2945]  eta: 5:04:33  lr: 0.000995  seg_loss: 0.4301 (0.4422)  time: 0.1204  max mem: 835  @2021-04-07 14:10:55
INFO utils.py(143): train epoch: [0]  [1500/2945]  eta: 5:05:39  lr: 0.000995  seg_loss: 0.3869 (0.4092)  time: 0.1309  max mem: 835  @2021-04-07 14:11:08
INFO utils.py(143): train epoch: [0]  [1600/2945]  eta: 5:05:18  lr: 0.000994  seg_loss: 0.3830 (0.3862)  time: 0.1237  max mem: 835  @2021-04-07 14:11:20
INFO utils.py(143): train epoch: [0]  [1700/2945]  eta: 5:05:28  lr: 0.000994  seg_loss: 0.3786 (0.3948)  time: 0.1261  max mem: 835  @2021-04-07 14:11:33
INFO utils.py(143): train epoch: [0]  [1800/2945]  eta: 5:05:28  lr: 0.000993  seg_loss: 0.3524 (0.3561)  time: 0.1293  max mem: 835  @2021-04-07 14:11:46
INFO utils.py(143): train epoch: [0]  [1900/2945]  eta: 5:05:58  lr: 0.000992  seg_loss: 0.3496 (0.3504)  time: 0.1258  max mem: 835  @2021-04-07 14:11:58
INFO utils.py(143): train epoch: [0]  [2000/2945]  eta: 5:05:21  lr: 0.000992  seg_loss: 0.3382 (0.3483)  time: 0.1169  max mem: 835  @2021-04-07 14:12:11
INFO utils.py(143): train epoch: [0]  [2100/2945]  eta: 5:05:17  lr: 0.000991  seg_loss: 0.3467 (0.3439)  time: 0.1300  max mem: 835  @2021-04-07 14:12:23
INFO utils.py(143): train epoch: [0]  [2200/2945]  eta: 5:04:50  lr: 0.000991  seg_loss: 0.3327 (0.3402)  time: 0.1269  max mem: 835  @2021-04-07 14:12:35
INFO utils.py(143): train epoch: [0]  [2300/2945]  eta: 5:04:53  lr: 0.000990  seg_loss: 0.3325 (0.3249)  time: 0.1209  max mem: 835  @2021-04-07 14:12:48
INFO utils.py(143): train epoch: [0]  [2400/2945]  eta: 5:04:34  lr: 0.000989  seg_loss: 0.3260 (0.3495)  time: 0.1211  max mem: 835  @2021-04-07 14:13:00
INFO utils.py(143): train epoch: [0]  [2500/2945]  eta: 5:04:36  lr: 0.000989  seg_loss: 0.3193 (0.3402)  time: 0.1295  max mem: 835  @2021-04-07 14:13:13
INFO utils.py(143): train epoch: [0]  [2600/2945]  eta: 5:04:32  lr: 0.000988  seg_loss: 0.3112 (0.3162)  time: 0.1180  max mem: 835  @2021-04-07 14:13:25
INFO utils.py(143): train epoch: [0]  [2700/2945]  eta: 5:04:14  lr: 0.000988  seg_loss: 0.2947 (0.3068)  time: 0.1129  max mem: 835  @2021-04-07 14:13:37
INFO utils.py(143): train epoch: [0]  [2800/2945]  eta: 5:04:02  lr: 0.000987  seg_loss: 0.2932 (0.2908)  time: 0.1232  max mem: 835  @2021-04-07 14:13:50
INFO utils.py(143): train epoch: [0]  [2900/2945]  eta: 5:03:44  lr: 0.000986  seg_loss: 0.2842 (0.2929)  time: 0.1144  max mem: 835  @2021-04-07 14:14:02
INFO utils.py(143): train epoch: [0]  [2944/2945]  eta: 5:03:46  lr: 0.000986  seg_loss: 0.3146 (0.3162)  time: 0.1278  max mem: 835  @2021-04-07 14:14:08
INFO utils.py(153): train epoch: [0] Total time: 0:06:04 (0.1239 s / it)
INFO utils.py(143): Valid epoch: [0]  [100/188]  eta: 0:29:45  best: 0.0000 (0.0000)  overall_acc: 0.9485 (0.9466)  avg_per_class_acc: 0.8831 (0.8857)  mIoU: 0.7457 (0.7473)  avg_dice: 0.8510 (0.8511)  time: 0.1410  max mem: 835  @2021-04-07 14:14:27
INFO utils.py(143): Valid epoch: [0]  [187/188]  eta: 0:24:17  best: 0.0000 (0.0000)  overall_acc: 0.9458 (0.9456)  avg_per_class_acc: 0.8883 (0.8863)  mIoU: 0.7530 (0.7534)  avg_dice: 0.8554 (0.8555)  time: 0.1501  max mem: 835  @2021-04-07 14:14:37
INFO utils.py(153): Valid epoch: [0] Total time: 0:00:29 (0.1545 s / it)
INFO engine.py(101): class background      iou 0.94332
INFO engine.py(101): class facial skin     iou 0.83347
INFO engine.py(101): class brow            iou 0.64997
INFO engine.py(101): class eye             iou 0.67178
INFO engine.py(101): class nose            iou 0.82712
INFO engine.py(101): class upper lip       iou 0.66677
INFO engine.py(101): class inner mouth     iou 0.71638
INFO engine.py(101): class lower lip       iou 0.70817
INFO engine.py(105): ######## saving best at 0 epoch | mIoU 0 ---> 0.7504 #########
```