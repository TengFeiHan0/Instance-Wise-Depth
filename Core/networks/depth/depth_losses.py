

import torch
import torch.nn as nn 
import numpy as np
import torch.nn.functional as F 

from Core.layers.layers import SSIM,compute_depth_errors, get_smooth_loss



def compute_losses(cfg, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in cfg.MODEL.DEPTH.SCALES:
            loss = 0
            reprojection_losses = []

            
            source_scale = scale
           

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in cfg.MODEL.DEPTH.FRAME_IDS[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(compute_reprojection_loss(cfg,pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            
            identity_reprojection_losses = []
            for frame_id in cfg.MODEL.DEPTH.FRAME_IDS[1:]:
                pred = inputs[("color", frame_id, source_scale)]
                identity_reprojection_losses.append(
                    compute_reprojection_loss(cfg,pred, target))

            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

            if cfg.MODEL.DEPTH.AVG_REPROJECTION:
                identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
            else:
                # save both images, and do min all at once below
                identity_reprojection_loss = identity_reprojection_losses

            if cfg.MODEL.DEPTH.AVG_REPROJECTION:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            
            # add random numbers to break ties
            identity_reprojection_loss += torch.randn(
                identity_reprojection_loss.shape).cuda() * 0.00001

            combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
        
            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

           
            outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += cfg.MODEL.DEPTH.SMOOTH_WEIGHT* smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= cfg.MODEL.DEPTH.NUM_SCALES
        losses["loss"] = total_loss
        return losses

def compute_reprojection_loss(cfg,pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)
       
        SSIM().to(cfg.MODEL.DEVICE)
        
        ssim_loss = SSIM(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

def compute_depth_losses(inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]
        
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())    