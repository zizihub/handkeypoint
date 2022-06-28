"""Common image segmentation metrics.
"""

import torch
import kornia
import numpy as np
import cv2
EPS = 1e-10

# ----------------------------------------------------------------
# Segmentation Evaluation Metric
# ----------------------------------------------------------------


def nanmean(x):
    """Computes the arithmetic mean ignoring any NaNs."""
    return torch.mean(x[x == x])


def _fast_hist(true, pred, num_classes):
    """Pixel Level Confusion Matrix"""
    mask = (true >= 0) & (true < num_classes)
    hist = torch.bincount(
        num_classes * true[mask] + pred[mask],
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes).float().to(pred.device)
    return hist


def overall_pixel_accuracy(hist):
    """Computes the total pixel accuracy.
    The overall pixel accuracy provides an intuitive
    approximation for the qualitative perception of the
    label when it is viewed in its overall shape but not
    its details.
    Args:
        hist: confusion matrix.
    Returns:
        overall_acc: the overall pixel accuracy.
    """
    correct = torch.diag(hist).sum()
    total = hist.sum()
    overall_acc = correct / (total + EPS)
    return overall_acc


def per_class_pixel_accuracy(hist):
    """Computes the average per-class pixel accuracy.
    The per-class pixel accuracy is a more fine-grained
    version of the overall pixel accuracy. A model could
    score a relatively high overall pixel accuracy by
    correctly predicting the dominant labels or areas
    in the image whilst incorrectly predicting the
    possibly more important/rare labels. Such a model
    will score a low per-class pixel accuracy.
    Args:
        hist: confusion matrix.
    Returns:
        avg_per_class_acc: the average per-class pixel accuracy.
    """
    correct_per_class = torch.diag(hist)
    total_per_class = hist.sum(dim=1)
    per_class_acc = correct_per_class / (total_per_class + EPS)
    avg_per_class_acc = nanmean(per_class_acc)
    return avg_per_class_acc


def jaccard_index(hist):
    """Computes the Jaccard index, a.k.a the Intersection over Union (IoU).
    Args:
        hist: confusion matrix.
    Returns:
        avg_jacc: the average per-class jaccard index.
    """
    A_inter_B = torch.diag(hist)
    A = hist.sum(dim=1)
    B = hist.sum(dim=0)
    jaccard = A_inter_B / (A + B - A_inter_B + EPS)
    avg_jacc = nanmean(jaccard)
    return avg_jacc


def dice_coefficient(hist):
    """Computes the Sørensen–Dice coefficient, a.k.a the F1 score.
    Args:
        hist: confusion matrix.
    Returns:
        avg_dice: the average per-class dice coefficient.
    """
    A_inter_B = torch.diag(hist)
    A = hist.sum(dim=1)
    B = hist.sum(dim=0)
    dice = (2 * A_inter_B) / (A + B + EPS)
    avg_dice = nanmean(dice)
    return avg_dice


def mean_iou(hist, num_classes):
    classes = num_classes
    class_scores = torch.zeros((classes))
    for i in range(classes):
        class_scores[i] = hist[i, i] / \
            (max(1, torch.sum(hist[i, :])+torch.sum(hist[:, i])-hist[i, i]))
    return class_scores


def eval_metrics(true, pred, num_classes):
    """Computes various segmentation metrics on 2D feature maps.
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        pred: a tensor of shape [B, H, W] or [B, 1, H, W].
        num_classes: the number of classes to segment. This number
            should be less than the ID of the ignored class.
    Returns:
        overall_acc: the overall pixel accuracy.
        avg_per_class_acc: the average per-class pixel accuracy.
        avg_jacc: the jaccard index.
        avg_dice: the dice coefficient.
    """
    if not torch.is_tensor(true):
        true = torch.as_tensor(true).type_as(pred)
    true = true.to(pred.device)
    hist = torch.zeros((num_classes, num_classes)).to(pred.device)
    for t, p in zip(true, pred):
        hist += _fast_hist(t.flatten(), p.flatten(), num_classes)
    overall_acc = overall_pixel_accuracy(hist)
    avg_per_class_acc = per_class_pixel_accuracy(hist)
    avg_jacc = jaccard_index(hist)
    avg_dice = dice_coefficient(hist)
    class_scores = mean_iou(hist, num_classes)
    return overall_acc, avg_per_class_acc, avg_jacc, avg_dice, class_scores

# ----------------------------------------------------------------
# Matting Evaluation Metric
# ----------------------------------------------------------------


def eval_metrics_matting(true, pred):
    mad = MetricMAD()
    mse = MetricMSE()
    grad = MetricGRAD()
    conn = MetricCONN()
    return mad(pred, true), mse(pred, true), grad(pred, true), conn(pred, true)


class MetricMAD:
    def __call__(self, pred, true):
        return (pred - true).abs_().mean() * 1e3


class MetricMSE:
    def __call__(self, pred, true):
        return ((pred - true) ** 2).mean() * 1e3


class MetricGRAD:
    def __init__(self, sigma=1.4):
        self.filter_x, self.filter_y = self.gauss_filter(sigma)
        self.filter_x = torch.from_numpy(self.filter_x).unsqueeze(0).cuda()
        self.filter_y = torch.from_numpy(self.filter_y).unsqueeze(0).cuda()

    def __call__(self, pred, true):
        true_grad = self.gauss_gradient(true)
        pred_grad = self.gauss_gradient(pred)
        return ((true_grad - pred_grad) ** 2).sum() / 1000

    def gauss_gradient(self, img):
        img_filtered_x = kornia.filters.filter2D(img, self.filter_x, border_type='replicate')[0, 0]
        img_filtered_y = kornia.filters.filter2D(img, self.filter_y, border_type='replicate')[0, 0]
        return (img_filtered_x**2 + img_filtered_y**2).sqrt()

    @staticmethod
    def gauss_filter(sigma, epsilon=1e-2):
        half_size = np.ceil(sigma * np.sqrt(-2 * np.log(np.sqrt(2 * np.pi) * sigma * epsilon)))
        size = np.int(2 * half_size + 1)

        # create filter in x axis
        filter_x = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                filter_x[i, j] = MetricGRAD.gaussian(i - half_size, sigma) * MetricGRAD.dgaussian(
                    j - half_size, sigma)

        # normalize filter
        norm = np.sqrt((filter_x**2).sum())
        filter_x = filter_x / norm
        filter_y = np.transpose(filter_x)

        return filter_x, filter_y

    @staticmethod
    def gaussian(x, sigma):
        return np.exp(-x**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

    @staticmethod
    def dgaussian(x, sigma):
        return -x * MetricGRAD.gaussian(x, sigma) / sigma**2


class MetricCONN:
    def __call__(self, preds, trues):
        preds = preds.detach().cpu().numpy()
        trues = trues.detach().cpu().numpy()
        step = 0.1
        thresh_steps = np.arange(0, 1 + step, step)
        conn_error = []
        for pred, true in zip(preds, trues):
            pred = pred.transpose(1, 2, 0)
            true = true.transpose(1, 2, 0)
            round_down_map = -np.ones_like(true)
            for i in range(1, len(thresh_steps)):
                true_thresh = true >= thresh_steps[i]
                pred_thresh = pred >= thresh_steps[i]
                intersection = (true_thresh & pred_thresh).astype(np.uint8)
                # connected components
                _, output, stats, _ = cv2.connectedComponentsWithStats(
                    intersection, connectivity=4)
                # start from 1 in dim 0 to exclude background
                size = stats[1:, -1]

                # largest connected component of the intersection
                omega = np.zeros_like(true)
                if len(size) != 0:
                    max_id = np.argmax(size)
                    # plus one to include background
                    omega[output == max_id + 1] = 1

                mask = (round_down_map == -1) & (omega == 0)
                round_down_map[mask] = thresh_steps[i - 1]
            round_down_map[round_down_map == -1] = 1

            true_diff = true - round_down_map
            pred_diff = pred - round_down_map
            # only calculate difference larger than or equal to 0.15
            true_phi = 1 - true_diff * (true_diff >= 0.15)
            pred_phi = 1 - pred_diff * (pred_diff >= 0.15)

            connectivity_error = np.sum(np.abs(true_phi - pred_phi))
            conn_error.append(connectivity_error / 1000)
        return np.mean(conn_error)


class MetricDTSSD:
    def __call__(self, pred_t, pred_tm1, true_t, true_tm1):
        dtSSD = ((pred_t - pred_tm1) - (true_t - true_tm1)) ** 2
        dtSSD = dtSSD.sum() / true_t.numel()
        dtSSD = dtSSD.sqrt()
        return dtSSD * 1e2
