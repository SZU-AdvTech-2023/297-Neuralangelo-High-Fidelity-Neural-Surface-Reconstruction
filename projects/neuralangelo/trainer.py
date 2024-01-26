'''
-----------------------------------------------------------------------------
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
-----------------------------------------------------------------------------
'''

import torch
import torch.nn.functional as torch_F
import wandb
import numpy as np
import os

from imaginaire.utils.distributed import master_only
from imaginaire.utils.visualization import wandb_image
from projects.nerf.trainers.base import BaseTrainer
from projects.neuralangelo.utils.misc import get_scheduler, eikonal_loss, curvature_loss
from projects.neuralangelo.data import gen_pts_view
from torch.utils.tensorboard import SummaryWriter

class Trainer(BaseTrainer):

    def __init__(self, cfg, is_inference=True, seed=0):
        super().__init__(cfg, is_inference=is_inference, seed=seed)
        self.metrics = dict()
        self.warm_up_end = cfg.optim.sched.warm_up_end
        self.cfg_gradient = cfg.model.object.sdf.gradient
        # get tensorboard to show the loss
        writer = SummaryWriter(os.path.join(cfg.logdir, "loss"))
        self.writer = writer
        # new_code
        # get the sfm points
        # 读取colmap生成的稀疏点云
        pts_dir = f"{cfg.data.root}/sfm_pts/points.npy"
        view_id_dir = f"{cfg.data.root}/sfm_pts/view_id.npy"
        self.pts = torch.from_numpy(np.load(pts_dir)).cuda()
        self.pts_view_id = np.load(view_id_dir, allow_pickle=True)
        if cfg.model.object.sdf.encoding.type == "hashgrid" and cfg.model.object.sdf.encoding.coarse2fine.enabled:
            self.c2f_step = cfg.model.object.sdf.encoding.coarse2fine.step
            self.model.module.neural_sdf.warm_up_end = self.warm_up_end

    def _init_loss(self, cfg):
        self.criteria["render"] = torch.nn.L1Loss()

    def setup_scheduler(self, cfg, optim):
        return get_scheduler(cfg.optim, optim)

    def _compute_loss(self, data, mode=None):
        if mode == "train":
            # Compute loss only on randomly sampled rays.
            self.losses["render"] = self.criteria["render"](data["rgb"], data["image_sampled"]) * 3  # FIXME:sumRGB?!
            self.writer.add_scalar("render_losss", self.losses["render"].detach(), self.current_epoch)
            self.metrics["psnr"] = -10 * torch_F.mse_loss(data["rgb"], data["image_sampled"]).log10()
            self.writer.add_scalar("psnr", self.metrics["psnr"].detach(), self.current_epoch)
            if "eikonal" in self.weights.keys():
                self.losses["eikonal"] = eikonal_loss(data["gradients"], outside=data["outside"])
                self.writer.add_scalar("eikonal_losss", self.losses["eikonal"].detach(), self.current_epoch)
            if "curvature" in self.weights:
                self.losses["curvature"] = curvature_loss(data["hessians"], outside=data["outside"])
                self.writer.add_scalar("curvature_losss", self.losses["curvature"].detach(), self.current_epoch)
           # new_code
            if "mask" in self.weights:
                self.losses["mask"] = self.criteria["render"](data["opacity"], data["mask_sampled"])
                self.writer.add_scalar("mask_losss", self.losses["mask"].detach(), self.current_epoch)
        else:
            if "sdf" in self.weights:
                batch_idx = data["idx"]
                idx_0 = batch_idx[0]
                idx_1= batch_idx[1]
                pts_view_0 = gen_pts_view(self, idx_0)
                pts_view_1 = gen_pts_view(self, idx_1)
                pts_view = torch.cat((pts_view_0, pts_view_1), dim=0)
                pts2sdf = self.model.module.neural_sdf.sdf(pts_view)
                self.losses["sdf"] = torch_F.l1_loss(pts2sdf, torch.zeros_like(pts2sdf), reduction='sum') / pts2sdf.shape[0]
                self.writer.add_scalar("sdf_losss", self.losses["sdf"].detach(), self.current_epoch)
            # Compute loss on the entire image.
            self.losses["render"] = self.criteria["render"](data["rgb_map"], data["image"])
            self.metrics["psnr"] = -10 * torch_F.mse_loss(data["rgb_map"], data["image"]).log10()

    def get_curvature_weight(self, current_iteration, init_weight, decay_factor):
        if "curvature" in self.weights:
            weight = (min(current_iteration / self.warm_up_end, 1.) if self.warm_up_end > 0 else 1.) * init_weight
            self.weights["curvature"] = weight / decay_factor

    def _start_of_iteration(self, data, current_iteration):
        model = self.model_module
        self.progress = model.progress = current_iteration / self.cfg.max_iter
        if self.cfg.model.object.sdf.encoding.coarse2fine.enabled:
            model.neural_sdf.set_active_levels(current_iteration)
            if self.cfg_gradient.mode == "numerical":
                model.neural_sdf.set_normal_epsilon()
                decay_factor = model.neural_sdf.growth_rate ** model.neural_sdf.add_levels  # TODO: verify?
                self.get_curvature_weight(current_iteration, self.cfg.trainer.loss_weight.curvature, decay_factor)
        elif self.cfg_gradient.mode == "numerical":
            model.neural_sdf.set_normal_epsilon()

        return super()._start_of_iteration(data, current_iteration)

    @master_only
    def log_wandb_scalars(self, data, mode=None):
        super().log_wandb_scalars(data, mode=mode)
        scalars = {
            f"{mode}/PSNR": self.metrics["psnr"].detach(),
            f"{mode}/s-var": self.model_module.s_var.item(),
        }
        if "curvature" in self.weights:
            scalars[f"{mode}/curvature_weight"] = self.weights["curvature"]
        if "eikonal" in self.weights:
            scalars[f"{mode}/eikonal_weight"] = self.weights["eikonal"]
        if mode == "train" and self.cfg_gradient.mode == "numerical":
            scalars[f"{mode}/epsilon"] = self.model.module.neural_sdf.normal_eps
        if self.cfg.model.object.sdf.encoding.coarse2fine.enabled:
            scalars[f"{mode}/active_levels"] = self.model.module.neural_sdf.active_levels
        wandb.log(scalars, step=self.current_iteration)

    @master_only
    def log_wandb_images(self, data, mode=None, max_samples=None):
        images = {"iteration": self.current_iteration, "epoch": self.current_epoch}
        if mode == "val":
            images_error = (data["rgb_map"] - data["image"]).abs()
            images.update({
                f"{mode}/vis/rgb_target": wandb_image(data["image"]),
                f"{mode}/vis/rgb_render": wandb_image(data["rgb_map"]),
                f"{mode}/vis/rgb_error": wandb_image(images_error),
                f"{mode}/vis/normal": wandb_image(data["normal_map"], from_range=(-1, 1)),
                f"{mode}/vis/inv_depth": wandb_image(1 / (data["depth_map"] + 1e-8) * self.cfg.trainer.depth_vis_scale),
                f"{mode}/vis/opacity": wandb_image(data["opacity_map"]),
            })
        wandb.log(images, step=self.current_iteration)

    def train(self, cfg, data_loader, single_gpu=False, profile=False, show_pbar=False):
        self.progress = self.model_module.progress = self.current_iteration / self.cfg.max_iter
        super().train(cfg, data_loader, single_gpu, profile, show_pbar)
