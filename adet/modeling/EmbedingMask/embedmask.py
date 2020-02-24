import math
from typing import List, Dict
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import ShapeSpec
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY

from adet.layers import DFConv2d, IOULoss

from .losses import make_embed_mask_loss_evaluator
from .proposals import make_embed_mask_postprocessor

from adet.utils.comm import Scale


__all__ =["EmbedMask"]
INF = 100000000




@PROPOSAL_GENERATOR_REGISTRY.register()
class EmbedMask(nn.Module):
    def __init__(self,cfg, input_shape: Dict[str, ShapeSpec]):
        super(EmbedMask, self).__init__()
        
        # self.in_features = cfg.MODEL.EMBED_MASK.IN_FEATURES
        
        # based on embedmask(maskrcnn-benchmark version)
        self.norm_reg_targets = cfg.MODEL.EMBED_MASK.NORM_REG_TARGETS
        
        self.embed_head = EmbedMaskHead(cfg, [input_shape[f] for f in self.in_features])
        self.box_selector_test = make_embed_mask_postprocessor(cfg)
        self.loss_evaluator = make_embed_mask_loss_evaluator(cfg)
    
    def forward(self, images, features, gt_instances):
        
        locations = self.compute_locations(features)
        box_cls, box_regression, centerness,proposal_embed, proposal_margin, pixel_embed = self.embed_head(features)
        
        if self.training:
            losses =  self.loss_evaluator( 
            locations, box_cls, box_regression, centerness, 
            proposal_embed, proposal_margin, pixel_embed, gt_instances)
            return losses
        else:
            proposals = self.bo_selector_test(
            locations, box_cls, box_regression,
            centerness, proposal_embed, proposal_margin, pixel_embed,
            image.image_sizes, gt_instances)
            
            return proposals
    
    
        
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
        
class EmbedMaskHead(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super(EmbedMaskHead, self).__init__()
        self.fpn_strides = cfg.MODEL.EMBED_MASK.FPN_STRIDES
        self.norm_reg_targets = cfg.MODEL.EMBED_MASK.NORM_REG_TARGETS
       
        num_classes = cfg.MODEL.EMBED_MASK.NUM_CLASSES
        embed_dim = cfg.MODEL.EMBED_MASK.EMBED_DIM
        prior_margin = cfg.MODEL.EMBED_MASK.PRIOR_MARGIN
        self.box_to_margin_scale = cfg.MODEL.EMBED_MASK.BOX_TO_MARGIN_SCALE
        self.box_to_margin_block = cfg.MODEL.EMBED_MASK.BOX_TO_MARGIN_BLOCK
        self.init_sigma_bias = math.log(-math.log(0.5) / (prior_margin ** 2))
        
        head_configs = {"cls": (cfg.MODEL.EMBED_MASK.NUM_CLS_CONVS,
                                False),
                        "bbox": (cfg.MODEL.EMBED_MASK.NUM_BOX_CONVS,
                                 cfg.MODEL.EMBED_MASK.USE_DEFORMABLE),
                        "share": (cfg.MODEL.EMBED_MASK.NUM_SHARE_CONVS,
                                  cfg.MODEL.EMBED_MASK.USE_DEFORMABLE)
                        "mask": (cfg.MODEL.EMBED_MASK.NUM_MASK_CONVS,
                                 cfg.MODEL.EMBED_MASK.USE_DEFORMABLE)}
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
        
        self.cls_logits = nn.Conv2d(in_channels, in_channels,
            kernel_size=3, stride=1, padding = 1)
        self.bbox_pred = nn.Conv2d(in_channels, 4,
            kernel_size=3, stride=1, padding =1)
        self.centerness = nn.Conv2d(in_channels, 1
            kernel_size=3, stride=1, padding=1)
        
        
        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in self.fpn_strides])
        
        
         for modules in [
            self.cls_tower, self.bbox_tower,
            self.share_tower, self.cls_logits,
            self.bbox_pred, self.centerness
        ]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.EMBED_MASK.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)
        
        #proposal embeding
        self.proposal_embed_pred = nn.Conv2d(in_channels, embed_dim,
                kernel_size=3, stride=1, padding =1 , bias=True)
        torch.nn.init.normal_(self.proposal_embed_pred.weight, std=0.01)
        torch.nn.init.constant_(self.proposal_embed_pred.bias, 0)
        
        #proposal margin
        self.proposal_margin_pred = nn.Conv2d(
            4, 1, kernel_size=1, stride=1, padding=0, bias=True
        )
        torch.nn.init.normal_(self.proposal_margin_pred.weight, std=0.01)
        torch.nn.init.constant_(self.proposal_margin_pred.bias, self.init_sigma_bias)
        
        #pixel embeding
        self.pixel_embed_pred = nn.Conv2d(in_channels, embed_dim, 
                kernel_size=3, stride=1, padding =1, bias=True)
        torch.nn.init.normal_(self.pixel_embed_pred.weight, std=0.01)       
        torch.nn.init.constant_(self.pixel_embed_pred.bias, self.init_sigma_bias) 
        
                 
    def forward(self,x):
        logits = []
        bbox_reg = []
        centerness = []
        proposal_margin = []
        proposal_embed = []
        
        for l , feature in enumerate(x):
            feature = self.share_tower(feature)
            cls_tower = self.cls_tower(feature)
            bbox_tower = self.bbox_tower(feature)
            
            logits.append(self.cls_logits(cls_tower))
            centerness.append(self.centerness(bbox_tower))
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
            
            #mask predictions
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
        #pixel embedding
        mask_x = x[0]
        mask_x = self.mask_tower(mask_x)
        pixel_embed = self.pixel_embed_pred(mask_x)
        
        return logits, bbox_reg, centerness, proposal_embed, proposal_margin, pixel_embed
            
            

        