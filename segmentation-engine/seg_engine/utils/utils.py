from collections import defaultdict, deque
import torch
import torch.distributed as dist
import time
import datetime
import random
import numpy as np
import os
import sys
import cv2
from typing import Optional


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=200, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        if not torch.is_tensor(value):
            value = torch.tensor(value)
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total],
                         dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @ property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @ property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @ property
    def global_avg(self):
        return self.total / self.count

    @ property
    def min(self):
        return min(self.deque).item()

    @ property
    def max(self):
        return max(self.deque).item()

    @ property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class SmoothedValueInference(SmoothedValue):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=50000, fmt=None):
        super(SmoothedValueInference, self).__init__(
            window_size=window_size, fmt=fmt)


class MetricLogger(object):
    def __init__(self, max_epoch, smooth_value, delimiter="\t"):
        self.meters = defaultdict(smooth_value)
        self.delimiter = delimiter
        self.max_epoch = max_epoch

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, logger, iterable, print_freq, header=None):
        i = 1
        if not header:
            header = ''

        current_epoch = int(''.join(list(filter(str.isdigit, header))))
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)+1))) + 'd'
        log_msg = self.delimiter.join([
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'max mem: {memory:.0f}',
            '@{date_time}'
        ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * \
                    (len(iterable) * (self.max_epoch-current_epoch+1) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                date_time = time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime())
                logger.info(log_msg.format(
                    i, len(iterable), eta=eta_string,
                    meters=str(self),
                    time=str(iter_time),
                    memory=torch.cuda.max_memory_allocated() / MB,
                    date_time=date_time))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def seed_reproducer(seed=2020):
    """Reproducer for pytorch experiment.
    Parameters
    ----------
    seed: int, optional (default = 2019)
        Radnom seed.
    Example
    -------
    seed_reproducer(seed=2019).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True


def env_log(logger):
    logger.info('__Python VERSION:', sys.version)
    logger.info('__pyTorch VERSION:', torch.__version__)
    logger.info('__CUDA VERSION')
    from subprocess import call
    # call(["nvcc", "--version"]) does not work
    #! nvcc --version
    logger.info('__CUDNN VERSION:', torch.backends.cudnn.version())
    logger.info('__Number CUDA Devices:', torch.cuda.device_count())
    logger.info('__Devices')
    call(["nvidia-smi", "--format=csv",
          "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
    logger.info('Active CUDA Device: GPU', torch.cuda.current_device())

    logger.info('Available devices ', torch.cuda.device_count())
    logger.info('Current cuda device ', torch.cuda.current_device())


def one_hot(labels: torch.Tensor,
            num_classes: int,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            eps: Optional[float] = 1e-6) -> torch.Tensor:
    """Converts an integer label 2D tensor to a one-hot 3D tensor.

    Args:
        labels (torch.Tensor) : tensor with labels of shape :math:`(N, H, W)`,
                                where N is batch siz. Each value is an integer
                                representing correct classification.
        num_classes (int): number of classes in labels.
        device (Optional[torch.device]): the desired device of returned tensor.
         Default: if None, uses the current device for the default tensor type
         (see torch.set_default_tensor_type()). device will be the CPU for CPU
         tensor types and the current CUDA device for CUDA tensor types.
        dtype (Optional[torch.dtype]): the desired data type of returned
         tensor. Default: if None, infers data type from values.

    Returns:
        torch.Tensor: the labels in one hot tensor.

    Examples::
        >>> labels = torch.LongTensor([[[0, 1], [2, 0]]])
        >>> tgm.losses.one_hot(labels, num_classes=3)
        tensor([[[[1., 0.],
                  [0., 1.]],
                 [[0., 1.],
                  [0., 0.]],
                 [[0., 0.],
                  [1., 0.]]]]
    """
    if not torch.is_tensor(labels):
        raise TypeError("Input labels type is not a torch.Tensor. Got {}"
                        .format(type(labels)))
    if not len(labels.shape) == 3:
        raise ValueError("Invalid depth shape, we expect BxHxW. Got: {}"
                         .format(labels.shape))
    if not labels.dtype == torch.int64:
        raise ValueError(
            "labels must be of the same dtype torch.int64. Got: {}" .format(
                labels.dtype))
    if num_classes < 1:
        raise ValueError("The number of classes must be bigger than one."
                         " Got: {}".format(num_classes))
    batch_size, height, width = labels.shape
    one_hot = torch.zeros(batch_size, num_classes, height, width,
                          device=device, dtype=dtype)
    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps


def vis_parsing_maps(im, parsing_anno, demo=False):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [105, 128, 112],
                   [85, 96, 225], [255, 0, 170], [0, 255, 0],
                   [85, 0, 255], [170, 255, 0], [255, 255, 255],
                   [0, 255, 170], [0, 0, 255], [85, 0, 255],
                   [170, 0, 255], [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]
    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)

    # create whiteboard
    vis_parsing_anno_color = np.zeros(
        (vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255
    # get class index
    num_of_class = np.max(vis_parsing_anno)

    # Image demostration
    if demo:
        # draw color on whiteboard
        for pi in range(1, num_of_class + 1):
            index = np.where(vis_parsing_anno == pi)
            vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

        vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
        vis_im = cv2.addWeighted(cv2.cvtColor(
            vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)
        return vis_im, vis_parsing_anno_color
    else:
        return None, vis_parsing_anno


if __name__ == '__main__':
    import cv2
    true = cv2.imread(
        '/data/zhangziwei/face-parsing.PyTorch/log/test_res/3262777136_1.png', 0)
    pred = cv2.imread(
        '/data/zhangziwei/face-parsing.PyTorch/log/test_res/3266693323_1.png', 0)

    true = torch.as_tensor([true], dtype=torch.int64)
    pred = torch.as_tensor([pred], dtype=torch.int64)

    print(eval_metrics(true, pred, 19))
