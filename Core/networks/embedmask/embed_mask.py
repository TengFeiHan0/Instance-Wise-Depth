import math
import torch
import torch.nn.functional as F
import torch.nn as nn 

from Core.utils.embedmask_utils import Scale

from .losses import EmbedMaskLoss
from .proposals import EmbedMaskProposal

class EmbedMaskHead(nn.Module):
    def __init__(self, cfg):
        super(EmbedMaskHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        self.fpn_strides = cfg.MODEL.EMBED_MASK.FPN_STRIDES
        self.norm_reg_targets = cfg.MODEL.EMBED_MASK.NORM_REG_TARGETS
        self.centerness_on_reg = cfg.MODEL.EMBED_MASK.CENTERNESS_ON_REG
        in_channels = cfg.MODEL.EMBED_MASK.IN_CHANNEL
        num_classes = cfg.MODEL.EMBED_MASK.NUM_CLASSES 
        embed_dim = cfg.MODEL.EMBED_MASK.EMBED_DIM
        prior_margin = cfg.MODEL.EMBED_MASK.PRIOR_MARGIN
        self.box_to_margin_scale = cfg.MODEL.EMBED_MASK.BOX_TO_MARGIN_SCALE
        self.box_to_margin_block = cfg.MODEL.EMBED_MASK.BOX_TO_MARGIN_BLOCK
        self.init_sigma_bias = math.log(-math.log(0.5) / (prior_margin ** 2))
        
        
        head_configs = {
            "cls": cfg.MODEL.EMBED_MASK.HEAD_NUM_CONVS,
            "bbox": cfg.MODEL.EMBED_MASK.HEAD_NUM_CONVS,
            "mask": cfg.MODEL.EMBED_MASK.HEAD_NUM_CONVS,
        }
        for head in head_configs:
            tower = []
            num_convs = head_configs[head]
            for i in range(num_convs):
                tower.append(
                    nn.Conv2d(
                        in_channels, in_channels, kernel_size=3,
                        stride = 1, padding = 1, bias=True
                    )
                )
                tower.append(
                    nn.GroupNorm(32, in_channels)
                )
                tower.append(
                    nn.ReLU(inplace=True)
                )
            self.add_module('{}_tower'.format(head), nn.Sequential(*tower))    
            
            
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3, stride=1,
            padding=1
        )
        self.centerness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )
        # initialization
        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, self.bbox_pred,
                        self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)
        
        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.EMBED_MASK.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])
        
        ########### Mask Predictions ############
        # proposal embedding
        self.proposal_embed_pred = nn.Conv2d(
            in_channels, embed_dim, kernel_size=3, stride=1, padding=1, bias=True
        )
        torch.nn.init.normal_(self.proposal_embed_pred.weight, std=0.01)
        torch.nn.init.constant_(self.proposal_embed_pred.bias, 0)
        # proposal margin
        self.proposal_margin_pred = nn.Conv2d(
            4, 1, kernel_size=1, stride=1, padding=0, bias=True
        )
        torch.nn.init.normal_(self.proposal_margin_pred.weight, std=0.01)
        torch.nn.init.constant_(self.proposal_margin_pred.bias, self.init_sigma_bias)

        # pixel embedding
        self.pixel_embed_pred = nn.Conv2d(
            in_channels, embed_dim, kernel_size=3, stride=1, padding=1, bias=True
        )
        for modules in [self.mask_tower, self.pixel_embed_pred]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)
    
    def forward(self, x):
        logits = []
        bbox_reg = []
        centerness = []
        proposal_margin = []
        proposal_embed = []
        for l, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            box_tower = self.bbox_tower(feature)

            logits.append(self.cls_logits(cls_tower))
            if self.centerness_on_reg:
                centerness.append(self.centerness(box_tower))
            else:
                centerness.append(self.centerness(cls_tower))

            bbox_pred = self.scales[l](self.bbox_pred(box_tower))
            if self.norm_reg_targets:
                bbox_pred = F.relu(bbox_pred)
                if self.training:
                    bbox_reg.append(bbox_pred)
                else:
                    bbox_reg.append(bbox_pred * self.fpn_strides[l])
            else:
                bbox_pred = torch.exp(bbox_pred)
                bbox_reg.append(bbox_pred)

            ############### Mask Prediction ###########
            embed_x = box_tower

            embed = self.proposal_embed_pred(embed_x)
            proposal_embed.append(embed)

            _, _, h, w = x[0].shape
            scale_size = self.box_to_margin_scale
            if scale_size == -1:
                scale_size = min(h, w) * self.fpn_strides[0]
            if self.norm_reg_targets:
                margin_x = (bbox_pred * self.fpn_strides[l] / scale_size)
            else:
                margin_x = bbox_pred / scale_size
            if self.box_to_margin_block:
                margin_x = margin_x.detach()
            proposal_margin.append(torch.exp(self.proposal_margin_pred(margin_x)))

        # pixel embedding
        mask_x = x[0]
        mask_x = self.mask_tower(mask_x)
        pixel_embed = self.pixel_embed_pred(mask_x)

        return logits, bbox_reg, centerness, proposal_embed, proposal_margin, pixel_embed


class EMBED_MASK(nn.Module):
    def __init__(self,cfg):
        super(EMBED_MASK, self).__init__()
        self.head = EmbedMaskHead(cfg)
        self.fpn_strides = cfg.MODEL.EMBED_MASK.FPN_STRIDES
        
        self.loss = EmbedMaskLoss(cfg)
        self.proposal = EmbedMaskProposal(cfg)
        
    def forward(self, input, features, targets=None):
        
        box_cls, box_regression, centerness, proposal_embed, proposal_margin, pixel_embed = self.head(features)
        locations = self.compute_locations(features)
        
        if self.training:
            loss_box_cls, loss_box_reg, loss_centerness, \
            mask_loss, smooth_loss = self.loss(
                locations, box_cls, box_regression, centerness, proposal_embed, proposal_margin, pixel_embed, targets
            )
            losses = {
                "loss_cls": loss_box_cls,
                "loss_reg": loss_box_reg,
                "loss_centerness": loss_centerness,
                'mask_loss': mask_loss,
                'smooth_loss': smooth_loss,
            }
            return None, losses
        
        else:
            boxes = self.proposal(
            locations, box_cls, box_regression,
            centerness, proposal_embed, proposal_margin, pixel_embed, targets)
            
            return boxes, {}
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
                

def build_embed_mask(cfg, in_channels):
        return EMBED_MASK(cfg, in_channels)       
