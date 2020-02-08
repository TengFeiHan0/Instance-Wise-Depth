import math
from typing import List, Dict
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import ShapeSpec
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY

from adet.layers import DFConv2d, IOULoss
from .polarmask_outputs import PolarMaskOutputs


__all__ = ["PolarMask"]

INF = 100000000


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


@PROPOSAL_GENERATOR_REGISTRY.register()
class PolarMask(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        # fmt: off
        self.in_features          = cfg.MODEL.PolarMask.IN_FEATURES
        self.fpn_strides          = cfg.MODEL.PolarMask.FPN_STRIDES
        self.focal_loss_alpha     = cfg.MODEL.PolarMask.LOSS_ALPHA
        self.focal_loss_gamma     = cfg.MODEL.PolarMask.LOSS_GAMMA
        self.center_sample        = cfg.MODEL.PolarMask.CENTER_SAMPLE
        self.strides              = cfg.MODEL.PolarMask.FPN_STRIDES
        self.radius               = cfg.MODEL.PolarMask.POS_RADIUS
        self.pre_nms_thresh_train = cfg.MODEL.PolarMask.INFERENCE_TH_TRAIN
        self.pre_nms_thresh_test  = cfg.MODEL.PolarMask.INFERENCE_TH_TEST
        self.pre_nms_topk_train   = cfg.MODEL.PolarMask.PRE_NMS_TOPK_TRAIN
        self.pre_nms_topk_test    = cfg.MODEL.PolarMask.PRE_NMS_TOPK_TEST
        self.nms_thresh           = cfg.MODEL.PolarMask.NMS_TH
        self.post_nms_topk_train  = cfg.MODEL.PolarMask.POST_NMS_TOPK_TRAIN
        self.post_nms_topk_test   = cfg.MODEL.PolarMask.POST_NMS_TOPK_TEST
        self.thresh_with_ctr      = cfg.MODEL.PolarMask.THRESH_WITH_CTR
        # fmt: on
        self.iou_loss = IOULoss(cfg.MODEL.PolarMask.LOC_LOSS_TYPE)
        # generate sizes of interest
        soi = []
        prev_size = -1
        for s in cfg.MODEL.PolarMask.SIZES_OF_INTEREST:
            soi.append([prev_size, s])
            prev_size = s
        soi.append([prev_size, INF])
        self.sizes_of_interest = soi
        self.polarmask_head = PolarMaskHead(cfg, [input_shape[f] for f in self.in_features])

    def forward(self, images, features, gt_instances):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        features = [features[f] for f in self.in_features]
        locations = self.compute_locations(features)
        cls_score, bbox_pred, ctrness, mask_pred = self.polarmask_head(features)

        if self.training:
            pre_nms_thresh = self.pre_nms_thresh_train
            pre_nms_topk = self.pre_nms_topk_train
            post_nms_topk = self.post_nms_topk_train
        else:
            pre_nms_thresh = self.pre_nms_thresh_test
            pre_nms_topk = self.pre_nms_topk_test
            post_nms_topk = self.post_nms_topk_test

        outputs = PolarMaskOutputs(
            images,
            locations,
            cls_score,
            bbox_pred,
            ctrness,
            mask_pred,
            self.focal_loss_alpha,
            self.focal_loss_gamma,
            self.iou_loss,
            self.center_sample,
            self.sizes_of_interest,
            self.strides,
            self.radius,
            self.polar_head.num_classes,
            pre_nms_thresh,
            pre_nms_topk,
            self.nms_thresh,
            post_nms_topk,
            self.thresh_with_ctr,
            gt_instances
        )

        if self.training:
            losses, _ = outputs.losses()
            return None, losses
        else:
            proposals = outputs.predict_proposals()
            return proposals, {}

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations


class PolarMaskHead(nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super().__init__()
        # TODO: Implement the sigmoid version first.
        self.num_classes = cfg.MODEL.PolarMask.NUM_CLASSES
        self.fpn_strides = cfg.MODEL.PolarMask.FPN_STRIDES
        head_configs = {"cls": (cfg.MODEL.PolarMask.NUM_CLS_CONVS,
                                False),
                        "bbox": (cfg.MODEL.PolarMask.NUM_BOX_CONVS,
                                 cfg.MODEL.PolarMask.USE_DEFORMABLE),
                        "share": (cfg.MODEL.PolarMask.NUM_SHARE_CONVS,
                                  cfg.MODEL.PolarMask.USE_DEFORMABLE)
                        "mask": (cfg.MODEL.PolarMask.NUM_MASK_CONVS,
                                 cfg.MODEL.PolarMask.USE_DEFORMABLE),}
        
        norm = None if cfg.MODEL.PolarMask.NORM == "none" else cfg.MODEL.PolarMask.NORM

        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]
       
        for head in head_configs:
            tower = []
            num_convs, use_deformable = head_configs[head]
            if use_deformable:
                conv_func = DFConv2d
            else:
                conv_func = nn.Conv2d
            for i in range(num_convs):
                tower.append(conv_func(
                        in_channels, in_channels,
                        kernel_size=3, stride=1,
                        padding=1, bias=True
                ))
                if norm == "GN":
                    tower.append(nn.GroupNorm(32, in_channels))
                tower.append(nn.ReLU())
            self.add_module('{}_tower'.format(head),
                            nn.Sequential(*tower))

        self.polar_cls = nn.Conv2d(in_channels, self.num_classes,kernel_size=3, padding=1)
        
        self.polar_mask = nn.Conv2d(in_channels, 36, kernel_size=3,padding=1)
        
        self.polar_ctrness = nn.Conv2d(in_channels, 1, kernel_size=3,padding=1)
        
        self.polar_reg = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)
        
        if cfg.MODEL.PolarMask.USE_SCALE:
            self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in self.fpn_strides])            
        else:
            self.scales = None

        for modules in [
            self.cls_tower, self.bbox_tower,
            self.share_tower, self.polar_cls,
            self.polar_reg, self.polar_ctrness, self.polar_mask
        ]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.PolarMask.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.polar_cls.bias, bias_value)

    def forward(self, x):
        cls_score = []
        bbox_pred = []
        ctrness = []
        mask_pred = []
        
        for l, feature in enumerate(x):
            feature = self.share_tower(feature)
            cls_tower = self.cls_tower(feature)
            bbox_tower = self.bbox_tower(feature)
            mask_tower = self.mask_tower(feature)
            
            cls_score.append(self.polar_cls(cls_tower))
            ctrness.append(self.polar_ctrness(bbox_tower))
            
            bbox_reg = self.polar_reg(bbox_tower)
            mask = self.polar_mask(mask_tower)
            if self.scales is not None:
                bbox_reg = self.scales[l](bbox_reg)
                mask = self.scales[l](mask)
            # Note that we use relu, as in the improved FCOS, instead of exp.
            bbox_pred.append(F.relu(bbox_reg))
            mask_pred.append(F.relu(mask))

        return cls_score, bbox_pred, ctrness, mask_pred
    
    def polar_centerness_target(self, mask_targets):
        # only calculate pos centerness targets, otherwise there may be nan
        centerness_targets = (mask_targets.min(dim=-1)[0] / mask_targets.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)

    def distance2mask(points, distances, angles, max_shape=None):
        '''Decode distance prediction to 36 mask points
        Args:
            points (Tensor): Shape (n, 2), [x, y].
            distance (Tensor): Distance from the given point to 36,from angle 0 to 350.
            angles (Tensor):
            max_shape (tuple): Shape of the image.

        Returns:
            Tensor: Decoded masks.
        '''
        num_points = points.shape[0]
        points = points[:, :, None].repeat(1, 1, 36)
        c_x, c_y = points[:, 0], points[:, 1]

        sin = torch.sin(angles)
        cos = torch.cos(angles)
        sin = sin[None, :].repeat(num_points, 1)
        cos = cos[None, :].repeat(num_points, 1)

        x = distances * sin + c_x
        y = distances * cos + c_y

        if max_shape is not None:
            x = x.clamp(min=0, max=max_shape[1] - 1)
            y = y.clamp(min=0, max=max_shape[0] - 1)

        res = torch.cat([x[:, None, :], y[:, None, :]], dim=1)
        return res



