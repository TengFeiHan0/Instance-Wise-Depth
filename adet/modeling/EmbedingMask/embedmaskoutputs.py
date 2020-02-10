import logging
import torch
import torch.nn.functional as F

from detectron2.layers import cat
from detectron2.structures import Instances, Boxes
from adet.utils.comm import get_world_size
from fvcore.nn import sigmoid_focal_loss_jit

from adet.utils.comm import reduce_sum
from adet.layers import ml_nms


logger = logging.getLogger(__name__)

INF = 100000000

"""
Shape shorthand in this module:

    N: number of images in the minibatch
    L: number of feature maps per image on which RPN is run
    Hi, Wi: height and width of the i-th feature map
    4: size of the box parameterization

Naming convention:

    labels: refers to the ground-truth class of an position.

    reg_targets: refers to the 4-d (left, top, right, bottom) distances that parameterize the ground-truth box.

    logits_pred: predicted classification scores in [-inf, +inf];
    
    reg_pred: the predicted (left, top, right, bottom), corresponding to reg_targets 

    ctrness_pred: predicted centerness scores
    
"""

class EmbedMaskOutputs(object):
    def __init__(self, 
                 images,
                 locations,
                 box_cls, box_regression, centerness,
                 proposal_embed, proposal_margin,
                 pixel_embed,gt_instances):
        self.box_cls = box_cls
        self.box_regression = box_regressions
        self.centerness = centerness
        self.proposal_embed = proposal_embed
        self.proposal_margin = proposal_margin
        self.pixel_embed = pixel_embed
        self.gt_instances = gt_instances
        self.image_sizes = images.image_sizes
        self.locations = locations
        self.fix_margin = cfg.MODEL.EMBED_MASK.FIX_MARGIN
        prior_margin = cfg.MODEL.EMBED_MASK.PRIOR_MARGIN
        self.init_margin = -math.log(0.5) / (prior_margin ** 2)
    
    def forward_for_single_feature_map(
            self, locations, box_cls,
            box_regression, centerness,
            proposal_embed, proposal_margin, image_sizes):
        """
        Arguments:
            anchors: list[BoxList]
            box_cls: tensor of size N, A * C, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        N, C, H, W = box_cls.shape

        # put in the same format as locations
        box_cls = box_cls.view(N, C, H, W).permute(0, 2, 3, 1)
        box_cls = box_cls.reshape(N, -1, C).sigmoid()
        box_regression = box_regression.view(N, 4, H, W).permute(0, 2, 3, 1)
        box_regression = box_regression.reshape(N, -1, 4)
        centerness = centerness.view(N, 1, H, W).permute(0, 2, 3, 1)
        centerness = centerness.reshape(N, -1).sigmoid()
        proposal_embed = proposal_embed.view(N, -1, H, W).permute(0, 2, 3, 1)
        proposal_embed = proposal_embed.reshape(N, H*W, -1)
        proposal_margin = proposal_margin.view(N, 1, H, W).permute(0, 2, 3, 1)
        proposal_margin = proposal_margin.reshape(N, -1)

        candidate_inds = box_cls > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

        # multiply the classification scores with centerness scores
        box_cls = box_cls * centerness[:, :, None]

        results = []
        for i in range(N):
            per_box_cls = box_cls[i]
            per_candidate_inds = candidate_inds[i]
            per_box_cls = per_box_cls[per_candidate_inds]

            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1] + 1

            per_box_regression = box_regression[i]
            per_box_regression = per_box_regression[per_box_loc]
            per_locations = locations[per_box_loc]

            per_proposal_embed = proposal_embed[i]
            per_proposal_embed = per_proposal_embed[per_box_loc]
            per_proposal_margin = proposal_margin[i][per_box_loc]

            per_pre_nms_top_n = pre_nms_top_n[i]

            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_locations = per_locations[top_k_indices]
                per_proposal_embed = per_proposal_embed[top_k_indices]
                per_proposal_margin = per_proposal_margin[top_k_indices]

            detections = torch.stack([
                per_locations[:, 0] - per_box_regression[:, 0],
                per_locations[:, 1] - per_box_regression[:, 1],
                per_locations[:, 0] + per_box_regression[:, 2],
                per_locations[:, 1] + per_box_regression[:, 3],
            ], dim=1)

            h, w = image_sizes[i]
            boxlist = BoxList(detections, (int(w), int(h)), mode="xyxy")
            boxlist.add_field("labels", per_class)
            boxlist.add_field("scores", per_box_cls)
            boxlist.add_field("proposal_embed", per_proposal_embed)
            boxlist.add_field("proposal_margin", per_proposal_margin)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size)
            results.append(boxlist)

        return results
    
    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            # multiclass nms
            result = boxlist_ml_nms(boxlists[i], self.nms_thresh)
            number_of_detections = len(result)

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.fpn_post_nms_top_n > 0:
                cls_scores = result.get_field("scores")
                image_thresh, _ = torch.kthvalue(
                    cls_scores.cpu(),
                    number_of_detections - self.fpn_post_nms_top_n + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        return results
        
    def predict_proposals(self):
        sampled_boxes = []
        
        bundle = (self.locations, self.box_cls,
                  self.box_regression, self.centerness,
                )
        for i (l,o, b, c) in enumerate(zip(*bundle)):
            em = self.proposal_embedp[i]
            mar = self.proposal_margin[i]
            
            if self.fix_margin:
                mar = torch.ones_like(mar) * self.init_margin
            sampled_boxes.append(
                self.forward_for_single_feature_map(
                    l, o, b, c, em, mar, image_sizes
                )
            )
        boxlists = list(zip(*sampled_boxes))
        boxlists = [Instances.cat(boxlist) for  boxlist in boxlists]
        boxlists = self.select_over_all_levels(boxlists)
        
        # resize pixel embedding for higher resolution
        N, dim, m_h, m_w = pixel_embed.shape
        o_h = m_h * self.mask_scale_factor
        o_w = m_w * self.mask_scale_factor
        pixel_embed = interpolate(pixel_embed, size=(o_h, o_w), mode='bilinear', align_corners=False)

        boxlists = self.forward_for_mask(boxlists, pixel_embed) 
        
        return boxlists
    
    