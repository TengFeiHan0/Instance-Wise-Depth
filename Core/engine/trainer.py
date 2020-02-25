# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
import sys
sys.path.insert(0, './')
import numpy as np
import time
import datetime
import logging
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import os
import json

from Core.utils.utils import *
from Core.utils.kitti_utils import *
from Core.utils.embedmask_utils import *
from tools.metric_logger import MetricLogger
from Core.layers.layers import *
from Core.config import cfg
from Core.data.build import make_data_loader
from Core.networks.pose_decoder import PoseDecoder
from Core.networks.depth import compute_depth_losses, compute_losses, DepthDecoder
from Core.networks.embedmask import EMBED_MASK
from Core.networks.backbone import FPN, FPNTopP6P7, resnet50, resnet101
from IPython import embed
from tools.logger import *


class Trainer:
    def __init__(self, cfg,args):
        self.log_path = os.path.join(args.log_dir, cfg.MODEL.NAME)

        self.models = {}
        self.parameters_to_train = []
        self.parameters_mask = []        
        
        self.device = cfg.MODEL.DEVICE
        self.num_scales = len(cfg.MODEL.DEPTH.SCALES)
        self.num_input_frames = len(cfg.MODEL.DEPTH.FRAME_IDS) 
        self.num_pose_frames = 2 if cfg.MODEL.DEPTH.POSE_FRAMES_INPUT == "pairs" else self.num_input_frames

        assert cfg.MODEL.DEPTH.FRAME_IDS[0] == 0, "frame_ids must start with 0"

        self.models["encoder"] = resnet50(pretrained=False)
        self.parameters_to_train += list(self.models["encoder"].parameters())
        self.models["encoder"].to(self.device)
       
        self.models["depth"] = DepthDecoder(cfg)
        self.parameters_to_train += list(self.models["depth"].parameters())  
        self.models["depth"].to(self.device) 
        
        fpn_top = FPNTopP6P7(cfg)
        self.fpn = FPN(cfg, fpn_top)
        self.models["mask"] = EMBED_MASK(cfg)
        self.parameters_mask += list(self.models["mask"].parameters())  
        self.models["mask"].to(self.device)
        
        if cfg.MODEL.DEPTH.POSE_MODEL_TYPE == "shared":
            self.models["pose"] = PoseDecoder(cfg, self.num_pose_frames)

        
        self.models["pose"].to(self.device)    
        self.parameters_to_train += list(self.models["pose"].parameters())
        
        self.depth_optimizer = optim.Adam(self.parameters_to_train, cfg.MODEL.DEPTH.LEARNING_RATE)
        self.depth_lr_scheduler = optim.lr_scheduler.StepLR(
            self.depth_optimizer, cfg.MODEL.DEPTH.SCHEDULER_STEP_SIZE, 0.1)
        
        self.mask_optimizer = optim.SGD(
        self.parameters_mask,lr=cfg.MODEL.MASK_LR, momentum=0.9,weight_decay=cfg.MODEL.MASK_WEIGHT_DECAY,
        nesterov=True,)     
        self.mask_scheduler = optim.lr_scheduler.MultiStepLR(self.mask_optimizer, milestones=[16, 22], gamma=0.1)
  
        if args.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", args.model_name)
        print("Models and tensorboard events files are saved to:\n  ", args.log_dir)
        print("Training is using:\n  ", self.device)

        
       
        self.data_loader = make_data_loader(
        cfg,is_train=True,is_distributed=args.distributed)
        
        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        # self.checkpoint_period= cfg.SOLVER.CHECKPOINT_PERIOD
        # output_dir = cfg.OUTPUT_DIR
        # save_to_disk = get_rank() == 0
        # self.checkpoint = DetectronCheckpointer(
        # cfg, self.models, self.mask_optimizer, self.mask_scheduler, output_dir, save_to_disk) 
    
        self.backproject_depth = {}
        self.project_3d = {}
        for scale in cfg.MODEL.DEPTH.SCALES:
            h = cfg.INPUT.HEIGHT // (2 ** scale)
            w = cfg.INPUT.WIDTH // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(cfg.SOLVER.IMS_PER_BATCH, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(cfg.SOLVER.IMS_PER_BATCH, h, w)
            self.project_3d[scale].to(self.device)

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        
        self.start_time = time.time()
        for self.epoch in range(cfg.MODEL.NUM_EPOCHS):
            self.run_epoch()
            if (self.epoch + 1) % args.save_frequency == 0:
                self.save_model(cfg)
    
    def process_batch(self,cfg,inputs,targets):
        
        for key, ipt in inputs.items():
                inputs[key] = ipt.to(self.device)

        if cfg.MODEL.DEPTH.POSE_MODEL_TYPE == "shared":
                # If we are using a shared encoder for both depth and pose (as advocated
                # in monodepthv1), then all images are fed separately through the depth encoder.
                all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in cfg.MODEL.DEPTH.FRAME_IDS])
                all_features = self.models["encoder"](all_color_aug)
                all_features = [torch.split(f, cfg.SOLVER.IMS_PER_BATCH) for f in all_features]
                
                features = {}
                for i, k in enumerate(cfg.MODEL.DEPTH.FRAME_IDS):
                    features[k] = [f[i] for f in all_features]
                
                feats = self.fpn(features[0])
                proposals, loss_dict = self.models["mask"](inputs, feats,targets)
                                              
                outputs = self.models["depth"](features[0])                           
        else:
                # Otherwise, we only feed the image with frame_id 0 through the depth encoder
                features = self.models["encoder"](inputs["color_aug", 0, 0])
                feats = self.fpn(features[0])
                proposals, loss_dict = self.models["mask"](inputs, feats,targets)
                outputs = self.models["depth"](features)
                           
        mask_losses = sum(loss for loss in loss_dict.values())
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        self.meters.update(loss=losses_reduced, **loss_dict_reduced)
                    
        outputs.update(self.predict_poses(inputs, features))

        self.generate_images_pred(inputs, outputs)
        depth_losses = compute_losses(cfg,inputs, outputs)

        return outputs, depth_losses, proposals, mask_losses
    
    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        logger = logging.getLogger("Instance-wise depth.trainer")
        logger.info("Start training")
        
        self.depth_lr_scheduler.step()
        self.mask_scheduler.step()
        self.set_train()
        self.meters = MetricLogger(delimiter="  ")
        start_training_time = time.time()
        end = time.time()
        max_iter = len(self.data_loader)
        start_iter = 0
        for batch_idx, (inputs,targets) in enumerate(self.data_loader, start_iter):
            
            data_time = time.time() - end
            before_op_time = time.time()
            targets = [target.to(device) for target in targets]
            
            outputs, losses, proposals, mask_losses = self.process_batch(cfg,inputs,targets)     
            self.mask_optimizer.zero_grad()
            self.depth_optimizer.zero_grad()
            
            losses["loss"].backward()
            mask_losses.backward()
            
            self.depth_optimizer.step()
            self.mask_optimizer.step()
            
            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)
            eta_seconds = meters.time.global_avg * (max_iter - batch_idx)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                        
            duration = time.time() - before_op_time

            # log less frequent
            # ly after the first 2000 steps to save time & disk space
            early_phase = batch_idx % args.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0
            
            if early_phase or late_phase:
                log_time(cfg,batch_idx, duration, losses["loss"].cpu().data, 
                         self.start_time, self.step)

                if "depth_gt" in inputs:
                    compute_depth_losses(inputs, outputs, losses)

                log(cfg, self.writers,self.step,"train", inputs, outputs, losses)
                self.val()

            self.step += 1
            
            if batch_idx % 20 == 0 or batch_idx == max_iter:
                logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=batch_idx,
                    meters=str(meters),
                    lr=self.mask_optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
            # if batch_idx % self.checkpoint_period == 0:
            #     self.checkpointer.save("model_{:07d}".format(batch_idx))
            # if batch_idx == max_iter:
            #     self.checkpointer.save("model_final")

        total_training_time = time.time() - start_training_time
        total_time_str = str(datetime.timedelta(seconds=total_training_time))
        logger.info(
            "Total training time: {} ({:.4f} s / it)".format(
                total_time_str, total_training_time / (max_iter)
            )
        )
          
    def predict_poses(self, cfg, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if cfg.MODEL.DEPTH.POSE_MODEL_TYPE == "shared":
                pose_feats = {f_i: features[f_i] for f_i in cfg.MODEL.DEPTH.FRAME_IDS}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in cfg.MODEL.DEPTH.FRAME_IDS}

            for f_i in cfg.MODEL.DEPTH.FRAME_IDS[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if cfg.MODEL.DEPTH.POSE_MODEL_TYPE == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif cfg.MODEL.DEPTH.POSE_MODEL_TYPE == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if cfg.MODEL.DEPTH.POSE_MODEL_TYPE in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in cfg.MODEL.DEPTH.FRAME_IDS if i != "s"], 1)

                if cfg.MODEL.DEPTH.POSE_MODEL_TYPE == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif cfg.MODEL.DEPTH.POSE_MODEL_TYPE == "shared":
                pose_inputs = [features[i] for i in cfg.MODEL.DEPTH.FRAME_IDS if i != "s"]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(cfg.MODEL.DEPTH.FRAME_IDS[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs

    def generate_images_pred(self, cfg, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in cfg.MODEL.SCALES:
            disp = outputs[("disp", scale)]
            
            source_scale = scale
            

            _, depth = disp_to_depth(disp, cfg.MODEL.DEPTH.MIN_DEPTH, cfg.MODEL.DEPTH.MAX_DEPTH)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(cfg.MODEL.DEPTH.FRAME_IDS[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")

                
                outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]
    
    

    def save_model(self,cfg):
        """Save model weights to disk
        """
        save_folder = os.path.join(cfg.MODEL.LOG_PATH, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = cfg.MODEL.DEPTH.HEIGHT
                to_save['width'] = cfg.MODEL.DEPTH.WIDTH
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.depth_optimizer.state_dict(), save_path)
        
    def load_model(self,args):
        """Load model(s) from disk
        """
        self.load_weights_folder = os.path.expanduser(args.load_weights_folder)

        assert os.path.isdir(self.load_weights_folder), \
            "Cannot find folder {}".format(self.load_weights_folder)
        print("loading model from folder {}".format(self.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
   

    
