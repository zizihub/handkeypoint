from seg_engine.models.base.build import POSTPROCESS_REGISTRY
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...config import configurable

nn.SyncBatchNorm.convert_sync_batchnorm

"""
Shape shorthand in this module:
    N: minibatch dimension size, i.e. the number of RoIs for instance segmenation or the
        number of images for semantic segmenation.
    R: number of ROIs, combined over all images, in the minibatch
    P: number of points
"""


def point_sample(input, point_coords, **kwargs):
    """
    A wrapper around: function: `torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike: function: `torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x[0, 1] square.
    Args:
        input(Tensor): A tensor of shape(N, C, H, W) that contains features map on a H x W grid.
        point_coords(Tensor): A tensor of shape(N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x[0, 1] normalized point coordinates.
    Returns:
        output(Tensor): A tensor of shape(N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as: function: `torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


def get_uncertain_point_coords_with_randomness(
    coarse_logits, uncertainty_func, num_points, oversample_ratio, importance_sample_ratio
):
    """
    Sample points in [0, 1] x[0, 1] coordinate space based on their uncertainty. The unceratinties
        are calculated for each point using 'uncertainty_func' function that takes point's logit
        prediction as input.
    See PointRend paper for details.
    Args:
        coarse_logits(Tensor): A tensor of shape(N, C, Hmask, Wmask) or (N, 1, Hmask, Wmask) for
            class-specific or class-agnostic prediction.
        uncertainty_func: A function that takes a Tensor of shape(N, C, P) or (N, 1, P) that
            contains logit predictions for P points and returns their uncertainties as a Tensor of
            shape(N, 1, P).
        num_points(int): The number of points P to sample.
        oversample_ratio(int): Oversampling parameter.
        importance_sample_ratio(float): Ratio of points that are sampled via importnace sampling.
    Returns:
        point_coords(Tensor): A tensor of shape(N, P, 2) that contains the coordinates of P
            sampled points.
    """
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)
    # It is crucial to calculate uncertainty based on the sampled prediction value for the points.
    # Calculating uncertainties of the coarse predictions first and sampling them for points leads
    # to incorrect results.
    # To illustrate this: assume uncertainty_func(logits)=-abs(logits), a sampled point between
    # two coarse predictions with -1 and 1 logits has 0 logits, and therefore 0 uncertainty value.
    # However, if we calculate uncertainties for the coarse predictions first,
    # both will have -1 uncertainty, and the sampled point will get -1 uncertainty.
    point_uncertainties = uncertainty_func(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
        num_boxes, num_uncertain_points, 2
    )
    if num_random_points > 0:
        point_coords = torch.cat(
            [
                point_coords,
                torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device),
            ],
            dim=1,
        )
    return point_coords


def get_uncertain_point_coords_on_grid(uncertainty_map, num_points):
    """
    Find `num_points` most uncertain points from `uncertainty_map` grid.
    Args:
        uncertainty_map(Tensor): A tensor of shape(N, 1, H, W) that contains uncertainty
            values for a set of points on a regular H x W grid.
        num_points(int): The number of points P to select.
    Returns:
        point_indices(Tensor): A tensor of shape(N, P) that contains indices from
            [0, H x W) of the most uncertain points.
        point_coords(Tensor): A tensor of shape(N, P, 2) that contains[0, 1] x[0, 1] normalized
            coordinates of the most uncertain points from the H x W grid.
    """
    R, _, H, W = uncertainty_map.shape
    h_step = 1.0 / float(H)
    w_step = 1.0 / float(W)

    num_points = min(H * W, num_points)
    point_indices = torch.topk(uncertainty_map.view(R, H * W), k=num_points, dim=1)[1]
    point_coords = torch.zeros(R, num_points, 2, dtype=torch.float, device=uncertainty_map.device)
    point_coords[:, :, 0] = w_step / 2.0 + (point_indices % W).to(torch.float) * w_step
    point_coords[:, :, 1] = h_step / 2.0 + (point_indices // W).to(torch.float) * h_step
    return point_indices, point_coords


def calculate_uncertainty(sem_seg_logits):
    """
    For each location of the prediction `sem_seg_logits` we estimate uncerainty as the
        difference between top first and top second predicted logits.
    Args:
        mask_logits (Tensor): A tensor of shape (N, C, ...), where N is the minibatch size and
            C is the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (N, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    top2_scores = torch.topk(sem_seg_logits, k=2, dim=1)[0]
    return (top2_scores[:, 1] - top2_scores[:, 0]).unsqueeze(1)


@POSTPROCESS_REGISTRY.register()
class PointRend(nn.Module):
    @configurable
    def __init__(self, fc_dim_in=32, fc_dim=256, num_classes=8, num_fc=3):
        super().__init__()
        self.coarse_pred_each_layer = True
        self.num_fc = num_fc
        for k in range(num_fc):
            fc = nn.Conv1d(fc_dim_in, fc_dim, kernel_size=1, stride=1, padding=0, bias=True)
            self.add_module('fc{}'.format(k+1), fc)
            fc_dim_in = fc_dim
            fc_dim_in += num_classes if self.coarse_pred_each_layer else 0
        self.k = 3
        self.beta = 0.75
        self.steps = 2
        self.predictor = nn.Conv1d(fc_dim_in, num_classes, kernel_size=1, stride=1, padding=0)

    @classmethod
    def from_config(cls, cfg, encoder_channels):
        return{
            'fc_dim_in': encoder_channels[-4]+cfg.DATASET.NUM_CLASSES+1,
            'fc_dim': cfg.MODEL.POSTPROCESS.FC_DIM,
            'num_classes': cfg.DATASET.NUM_CLASSES+1,
            'num_fc': cfg.MODEL.POSTPROCESS.NUM_FC,
        }

    def point_head(self, x, coarse):
        for i in range(1, self.num_fc+1):
            x = F.relu(eval(f'self.fc{i}')(x))
            if self.coarse_pred_each_layer:
                x = torch.cat([x, coarse], dim=1)
        return self.predictor(x)

    def forward(self, x, result):
        """
        1. Fine-grained features are interpolated from p2 for DeeplabV3
        2. During training we sample as many points as there are on a stride 16 feature map of the input
        3. To measure prediction uncertainty
           we use the same strategy during training and inference: the difference between the most
           confident and second most confident class probabilities.
        """
        p2 = result['features'][-4]
        coarse_sem_seg_logits = result['masks']
        if not self.training:
            return self.inference(x, coarse_sem_seg_logits, p2)
        with torch.no_grad():
            point_coords = get_uncertain_point_coords_with_randomness(coarse_logits=coarse_sem_seg_logits,
                                                                      uncertainty_func=calculate_uncertainty,
                                                                      num_points=coarse_sem_seg_logits.shape[-1]**2,
                                                                      oversample_ratio=self.k,
                                                                      importance_sample_ratio=self.beta,
                                                                      )
        coarse = point_sample(coarse_sem_seg_logits, point_coords, align_corners=False)
        fine = point_sample(p2, point_coords, align_corners=False)
        feature_representation = torch.cat([coarse, fine], dim=1)

        rend = self.point_head(feature_representation, coarse)

        return {"rend": rend, "points": point_coords}

    @torch.no_grad()
    def inference(self, x, coarse_sem_seg_logits, p2):
        """
        During inference, subdivision uses N = 4096
        (i.e., the number of points in the stride 4 map of a 256x256 image)
        """
        out = coarse_sem_seg_logits.clone()
        num_points = (coarse_sem_seg_logits.shape[-1]*2)**2

        for _ in range(self.steps):
            out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=False)
            uncertainty_map = calculate_uncertainty(out)

            point_indices, point_coords = get_uncertain_point_coords_on_grid(uncertainty_map=uncertainty_map,
                                                                             num_points=num_points)
            coarse = point_sample(coarse_sem_seg_logits, point_coords, align_corners=False)
            fine = point_sample(p2, point_coords, align_corners=False)

            feature_representation = torch.cat([coarse, fine], dim=1)

            rend = self.point_head(feature_representation, coarse)

            N, C, H, W = out.shape
            point_indices = point_indices.unsqueeze(1).expand(-1, C, -1)
            out = (out.reshape(N, C, -1)
                      .scatter_(2, point_indices, rend)
                      .view(N, C, H, W))

        return {"fine": out}
