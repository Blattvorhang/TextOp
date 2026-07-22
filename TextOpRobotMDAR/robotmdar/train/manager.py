import os
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import errno
import shutil
import torch
import torch.nn as nn
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm
import copy

from robotmdar.dtype.motion import get_zero_feature, perturb_feature_v3, FeatureVersion
from robotmdar.dtype.rotation import rot6d_to_matrix, matrix_to_rot6d, quaternion_to_matrix, xyzw_to_wxyz
from isaac_utils.rotations import get_euler_xyz


class BaseManager(ABC):
    """
    Abstract base class for training managers.

    Defines common interfaces and functionality for training management,
    including EMA support, history selection, and training lifecycle management.
    """
    optimizer: torch.optim.Optimizer
    dataset: Any

    max_steps: int
    stages: List[int]
    stage_idx: int
    # -1 for not started, 0 for first stage, 1 for second stage, etc.
    use_rollout: bool

    use_static_pose: bool

    anneal_lr: bool
    learning_rate: float
    max_grad_norm: float
    loss_weight: Dict[str, float]
    extra: Dict[str, float]
    ckpt: DictConfig
    device: str
    platform: Any
    save_every: int
    eval_every: int
    eval_steps: int

    def __init__(self, **kwargs):
        # 自动保存所有参数为成员变量
        for k, v in kwargs.items():
            setattr(self, k, v)

        # 设置默认值
        self.stage_idx = -1
        self._stage_steps = torch.cumsum(torch.tensor(self.stages).int(), dim=0)
        # assert self._stage_steps[
        #     -1] == self.max_steps, "Stage steps must sum to max_steps"
        self.max_steps = int(self._stage_steps[-1])

        self.step = 0

        self._to_eval_steps = 0
        self._total_eval_loss_dict = defaultdict(lambda: torch.tensor(0.0))
        self._total_eval_extras_dict = defaultdict(lambda: torch.tensor(0.0))
        self.rec_criterion = nn.HuberLoss(reduction='mean', delta=1.0)

        self.save_dir = Path(self.save_dir)
        self._tqdm = None
        self.extra = {}

        # EMA相关
        self.use_ema = getattr(self, 'use_ema', False)
        self.ema_decay = getattr(self, 'ema_decay', 0.999)
        self.ema_models = {}

        # History选择相关
        self.static_prob = getattr(self, 'static_prob', 0.0)

    def pre_step(self, is_eval: bool = False) -> None:
        """每步训练前调用"""
        self.stage_idx = min(
            torch.searchsorted(
                self._stage_steps, self.step, right=True, out_int32=True
            ).item(),
            len(self.stages) - 1,
        )  # type:ignore
        self.extra['stage'] = self.stage_idx

        if not self._tqdm:
            self._tqdm = tqdm(total=self.max_steps, initial=self.step, ncols=120, desc="Training")
        if self.anneal_lr:
            frac = 1.0 - self.step / self.max_steps
            lrnow = frac * self.learning_rate
            self.extra['lr'] = lrnow
            self.optimizer.param_groups[0]["lr"] = lrnow
        else:
            lrnow = self.learning_rate
            self.extra['lr'] = lrnow
            self.optimizer.param_groups[0]["lr"] = lrnow

    def post_step(
        self,
        is_eval: bool = False,
        loss_dict: Dict[str, torch.Tensor] = {},
        extras: Optional[Dict[str, torch.Tensor]] = None
    ) -> None:
        """每步训练后调用"""
        if extras is None:
            extras = {}

        if is_eval:
            self._to_eval_steps -= 1
            for k, v in loss_dict.items():
                self._total_eval_loss_dict[k] += v
            for k, v in extras.items():
                self._total_eval_extras_dict[k] += v
            if self._to_eval_steps == 0:
                # 报告 loss 到 "loss" group
                for k, v in self._total_eval_loss_dict.items():
                    self.platform.report_scalar(
                        "eval_" + k, v.cpu().numpy() / self.eval_steps, self.step, group_name="loss"
                    )
                # 报告 extras 到 "extras" group
                for k, v in self._total_eval_extras_dict.items():
                    self.platform.report_scalar(
                        "eval_" + k, (v.cpu().numpy() / self.eval_steps) if isinstance(v, torch.Tensor) else
                        (v / self.eval_steps),
                        self.step,
                        group_name="extras"
                    )
                tqdm.write(
                    f"Eval finished at step {self.step} with loss * {self.eval_steps}: {dict(self._total_eval_loss_dict)}"
                )
                if self._total_eval_extras_dict:
                    tqdm.write(f"Eval extras * {self.eval_steps}: {dict(self._total_eval_extras_dict)}")
                self._total_eval_loss_dict = defaultdict(lambda: torch.tensor(0.0))
                self._total_eval_extras_dict = defaultdict(lambda: torch.tensor(0.0))
            return

        self.step += 1

        assert self._tqdm is not None
        self._tqdm.update(1)
        self._tqdm.set_postfix({'stage': self.stage_idx, 'loss': loss_dict["total"].item(), 'lr': self.extra['lr']})
        if self.step >= self.max_steps:
            self._tqdm.close()

        for k, v in loss_dict.items():
            self.platform.report_scalar("train_" + k, v.cpu().numpy(), self.step, group_name="loss")
        for k, v in extras.items():
            self.platform.report_scalar(
                "train_" + k, v.cpu().numpy() if isinstance(v, torch.Tensor) else v, self.step, group_name="extras"
            )
        for k, v in self.extra.items():
            self.platform.report_scalar(k, v, self.step, group_name="extras")

        # 更新EMA模型
        self.update_ema_models()

        if self.step % self.save_every == 0 or self.step == self.max_steps:
            self.save_model()

        if (self.step % self.eval_every == 0 or self.step == self.max_steps):
            self._to_eval_steps = self.eval_steps
            self._total_eval_loss_dict = defaultdict(lambda: torch.tensor(0.0))
            self._total_eval_extras_dict = defaultdict(lambda: torch.tensor(0.0))

    def should_eval(self) -> bool:
        """是否需要评估"""
        return self._to_eval_steps > 0

    def grad_clip(self, model):
        """
        Apply gradient clipping to model parameters.

        Args:
            model: PyTorch model to clip gradients for
        """
        if self.max_grad_norm > 0:
            norm = nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
            self.extra['grad_norm'] = norm.detach().cpu().numpy()

    def __bool__(self):
        """Check if training should continue."""
        return self.step < self.max_steps

    def register_ema_model(self, name: str, model: nn.Module):
        """
        Register a model for EMA (Exponential Moving Average) tracking.

        Args:
            name: Unique identifier for the model
            model: PyTorch model to track with EMA
        """
        if self.use_ema:
            ema_model = copy.deepcopy(model)
            for param in ema_model.parameters():
                param.requires_grad = False
            self.ema_models[name] = ema_model
            logger.info(f"Registered EMA model: {name}")

    def update_ema(self, name: str, model: nn.Module):
        """
        Update EMA model parameters using exponential moving average.

        Args:
            name: Model identifier
            model: Current model to update EMA from
        """
        if self.use_ema and name in self.ema_models:
            ema_model = self.ema_models[name]
            for param, ema_param in zip(model.parameters(), ema_model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

    def get_ema_model(self, name: str) -> Optional[nn.Module]:
        """
        Get EMA model by name.

        Args:
            name: Model identifier

        Returns:
            EMA model if exists, None otherwise
        """
        if self.use_ema and name in self.ema_models:
            return self.ema_models[name]
        return None

    @abstractmethod
    def hold_model(self, *args, **kwargs):
        """子类实现: 绑定模型和优化器"""
        pass

    @abstractmethod
    def save_model(self) -> None:
        ...

    @abstractmethod
    def load_model(self) -> None:
        ...

    @abstractmethod
    def calc_loss(self, *args, **kwargs) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """子类实现: 计算损失，返回 (terms, extras)"""
        pass

    def should_rollout(self) -> bool:
        """
        Determine whether to use rollout history instead of ground truth.

        Returns:
            True if should use rollout history, False otherwise
        """
        if not self.use_rollout:
            return False
        if self.stage_idx < 1:
            return False
        prob = min(1.0, (self.step - self.stages[0]) / max(float(self.stages[1]), 1e-6))
        return torch.rand(1).item() < prob

    def should_static_pose(self) -> bool:
        """
        Determine whether to use static pose with perturbation.

        Returns:
            True if should use static pose, False otherwise
        """
        if not self.use_static_pose or self.stage_idx < 2:
            return False
        stage_start = self.stages[0] + self.stages[1]
        prob = min(
            1.0,
            (self.step - stage_start) / max(float(self.stages[2]), 1e-6),
        ) * self.static_prob
        return torch.rand(1).item() < prob

    def choose_history(
        self,
        gt_history: torch.Tensor,
        prev_motion: Optional[torch.Tensor] = None,
        history_len: Optional[int] = None,
        return_rollout: bool = False,
    ):
        """
        统一的history选择函数
        Args:
            gt_history: 来自数据集的真实历史 [B, H, D]
            prev_motion: 前一个primitive的预测结果 [B, T, D]
            history_len: 历史长度，如果None则使用gt_history的长度
        Returns:
            选择的历史 [B, H, D]
        """
        if history_len is None:
            history_len = gt_history.shape[1]

        # 1. 检查是否使用rollout history
        used_rollout = prev_motion is not None and self.should_rollout()
        if used_rollout:
            history_motion = prev_motion[:, -history_len:, :]
        else:
            # 使用ground truth history
            history_motion = gt_history

        # 2. 检查是否使用static pose
        if self.should_static_pose():
            zero_feature = get_zero_feature().expand_as(history_motion).to(history_motion.device)
            # 添加扰动
            perturbation_scale = getattr(self, 'static_perturbation_scale', 0.0)
            if perturbation_scale > 0:
                # Not Test
                zero_feature = perturb_feature_v3(zero_feature, perturbation_scale)
            history_motion = self.dataset.normalize(zero_feature)

        if return_rollout:
            return history_motion, used_rollout
        return history_motion

    @abstractmethod
    def update_ema_models(self):
        """子类实现: 更新EMA模型"""
        pass


class GeometryLoss:

    dataset: Any
    rec_criterion: nn.HuberLoss

    @staticmethod
    def calc_jerk(joints):
        vel = joints[:, 1:] - joints[:, :-1]  # --> B x T-1 x 22 x 3
        acc = vel[:, 1:] - vel[:, :-1]  # --> B x T-2 x 22 x 3
        jerk = acc[:, 1:] - acc[:, :-1]  # --> B x T-3 x 22 x 3
        jerk = torch.abs(jerk).mean()  # --> B x T-3 x 22, compute L1 norm of jerk
        return jerk

    @staticmethod
    def quantization(tensor, bits=8):
        if bits == 8:
            scale = 127.0 / tensor.abs().max().clamp(min=1e-8)
            quantized = (tensor * scale).round() / scale
        elif bits == 16:
            quantized = tensor.half().float()
        else:
            quantized = tensor

        return quantized

    @staticmethod
    def quat_geodesic_loss(q_pred, q_target):
        # q_pred, q_target: shape (..., 4), normalized or will be normalized
        q_pred = torch.nn.functional.normalize(q_pred, dim=-1)
        q_target = torch.nn.functional.normalize(q_target, dim=-1)

        # 内积
        dot = torch.sum(q_pred * q_target, dim=-1).abs()  # |q1·q2|

        # 防止数值问题
        dot = torch.clamp(dot, -1.0, 1.0)

        # geodesic distance
        angle = 2 * torch.acos(dot)  # shape (...,)

        return angle.mean()  # 或 angle^2.mean() 视具体任务而定

    def calc_geometry_loss(
        self,
        future_motion_pred,
        future_motion_gt,
        history_motion=None,
        smooth=False,
        quantize=False,
        drift=False
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """计算几何损失"""
        terms = {}
        extras = {}

        # future_motion_denorm = self.dataset.denormalize(future_motion_pred)
        # pred_dof_delta = future_motion_denorm[..., -23:]

        # if history_motion is not None:
        #     motion_denorm = self.dataset.denormalize(torch.cat((history_motion[:, -1, :].unsqueeze(1), future_motion_pred), dim=1))
        #     cal_dof_delta = motion_denorm[:, 1:, -46:-23] - motion_denorm[:, :-1, -46:-23]
        #     dof_delta_loss = self.rec_criterion(cal_dof_delta, pred_dof_delta)
        #     terms['dof_delta'] = dof_delta_loss

        if smooth and history_motion is not None:
            motion_tensor = torch.cat([history_motion, future_motion_pred], dim=1)
            diff = motion_tensor[:, 1:, :] - motion_tensor[:, :-1, :]
            terms['smooth'] = torch.abs(diff).mean()

        # Geometric loss
        future_motion_pred_fk = self.dataset.reconstruct_motion(future_motion_pred, need_denormalize=True, ret_fk=True)
        with torch.no_grad():
            future_motion_gt_fk = self.dataset.reconstruct_motion(future_motion_gt, need_denormalize=True, ret_fk=True)

        body_trans_loss = self.rec_criterion(
            future_motion_pred_fk['global_translation_extend'], future_motion_gt_fk['global_translation_extend']
        )  # [B=512, T=8, L=27, 3]
        body_rot_loss = self.rec_criterion(
            future_motion_pred_fk['global_rotation'], future_motion_gt_fk['global_rotation']
        )
        dof_pos_loss = self.rec_criterion(future_motion_pred_fk['dof_pos'], future_motion_gt_fk['dof_pos'])
        dof_vel_loss = self.rec_criterion(future_motion_pred_fk['dof_vel'], future_motion_gt_fk['dof_vel'])

        foot_should_contact = future_motion_gt_fk['contact_mask'].unsqueeze(-1)
        foot_trans_pred = future_motion_pred_fk['global_translation_extend'][:, :, self.dataset.skeleton.foot_id, :]
        foot_trans_gt = future_motion_gt_fk['global_translation_extend'][:, :, self.dataset.skeleton.foot_id, :]
        foot_contact_loss = self.rec_criterion(
            foot_trans_pred * foot_should_contact, foot_trans_gt * foot_should_contact
        )

        if quantize:
            quantize_pred_rot = self.quantization(future_motion_pred_fk['global_rotation'][:, -1, 0])
            quantize_gt_rot = self.quantization(future_motion_gt_fk['global_rotation'][:, -1, 0])

            quantize_pred_trans_xy = self.quantization(future_motion_pred_fk['global_translation_extend'][:, :, 0, :2])
            quantize_gt_trans_xy = self.quantization(future_motion_gt_fk['global_translation_extend'][:, :, 0, :2])
            terms['quantize_rot'] = self.rec_criterion(quantize_pred_rot, quantize_gt_rot)
            terms['quantize_trans'] = self.rec_criterion(quantize_pred_trans_xy, quantize_gt_trans_xy)

        drift_yaw_pred = get_euler_xyz(future_motion_pred_fk['global_rotation'][:, -1, 0], w_last=True)[2]  # (B,)
        drift_yaw_gt = get_euler_xyz(future_motion_gt_fk['global_rotation'][:, -1, 0], w_last=True)[2]  # (B,)

        drift_yaw_diff = (drift_yaw_pred - drift_yaw_gt) % (2 * torch.pi)
        drift_yaw_diff[drift_yaw_diff > torch.pi] -= 2 * torch.pi
        drift_yaw_loss = self.rec_criterion(drift_yaw_diff, torch.zeros_like(drift_yaw_diff))

        drift_xy_loss = self.rec_criterion(
            future_motion_pred_fk['global_translation_extend'][:, -1, 0, :2],
            future_motion_gt_fk['global_translation_extend'][:, -1, 0, :2]
        )

        if drift:
            terms['drift_yaw'] = drift_yaw_loss
            terms['drift_xy'] = drift_xy_loss

        terms['body_trans'] = body_trans_loss
        terms['body_rot'] = body_rot_loss
        terms['dof_pos'] = dof_pos_loss
        terms['dof_vel'] = dof_vel_loss
        terms['foot_contact'] = foot_contact_loss

        extras['drift_xy'] = drift_xy_loss
        extras['drift_yaw'] = drift_yaw_loss
        # if smooth:
        #     jerk_loss = self.calc_jerk(
        #         future_motion_pred_fk['global_translation_extend'])
        #     terms['smooth'] = jerk_loss

        return terms, extras

    def calc_geometry_loss_v2(self,
                              future_motion_pred,
                              future_motion_gt,
                              history_motion=None) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """计算几何损失"""

        terms = {}
        extras = {}

        future_motion_pred_fk = self.dataset.reconstruct_motion(
            future_motion_pred,
            # torch.cat((history_motion, future_motion_pred), dim=1),
            need_denormalize=True,
            ret_fk=True
        )

        with torch.no_grad():
            future_motion_gt_fk = self.dataset.reconstruct_motion(
                future_motion_gt,
                # torch.cat((history_motion, future_motion_gt), dim=1),
                need_denormalize=True,
                ret_fk=True
            )

        B, T = future_motion_pred_fk['root_trans_offset'].shape[:2]

        terms['fk_joints_rec'] = self.rec_criterion(
            future_motion_pred_fk['global_translation_extend'], future_motion_gt_fk['global_translation_extend']
        )
        # breakpoint()
        terms['joints_consistency'] = self.rec_criterion(
            future_motion_pred_fk['joints'].reshape(B, T, -1, 3), future_motion_pred_fk['global_translation_extend']
        )
        """temporal delta loss"""
        if history_motion is not None:
            pred_motion_tensor = torch.cat([history_motion[:, -1:, :], future_motion_pred], dim=1)
            pred_feature_dict = self.dataset.reconstruct_motion(pred_motion_tensor)

            pred_joints_delta = pred_feature_dict['joints_delta'][:, :-1, :]
            pred_transl_delta = pred_feature_dict['transl_delta'][:, :-1, :]
            pred_rot_delta = pred_feature_dict['rot_delta_6d'][:, :-1, :]
            calc_joints_delta = pred_feature_dict['joints'][:, 1:, :] - pred_feature_dict['joints'][:, :-1, :]
            calc_transl_delta = pred_feature_dict['root_trans_offset'][:, 1:, :] - pred_feature_dict['root_trans_offset'
                                                                                                    ][:, :-1, :]

            # breakpoint()
            pred_rot = quaternion_to_matrix(xyzw_to_wxyz(pred_feature_dict['root_rot']))
            calc_rot_delta_matrix = torch.matmul(pred_rot[:, 1:], pred_rot[:, :-1].permute(0, 1, 3, 2))
            calc_rot_delta_6d = matrix_to_rot6d(calc_rot_delta_matrix)

            terms["joints_delta"] = self.rec_criterion(calc_joints_delta, pred_joints_delta)
            terms["transl_delta"] = self.rec_criterion(calc_transl_delta, pred_transl_delta)
            terms["orient_delta"] = self.rec_criterion(calc_rot_delta_6d, pred_rot_delta)

        return terms, extras

    def calc_geometry_loss_v3(self,
                              future_motion_pred,
                              future_motion_gt,
                              history_motion=None) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """计算几何损失"""

        terms = {}
        extras = {}

        future_motion_pred_fk = self.dataset.reconstruct_motion(
            future_motion_pred,
            # torch.cat((history_motion, future_motion_pred), dim=1),
            need_denormalize=True,
            ret_fk=True
        )

        with torch.no_grad():
            future_motion_gt_fk = self.dataset.reconstruct_motion(
                future_motion_gt,
                # torch.cat((history_motion, future_motion_gt), dim=1),
                need_denormalize=True,
                ret_fk=True
            )

        body_trans_loss = self.rec_criterion(
            future_motion_pred_fk['global_translation_extend'], future_motion_gt_fk['global_translation_extend']
        )  # [B=512, T=8, L=27, 3]
        body_rot_loss = self.rec_criterion(
            future_motion_pred_fk['global_rotation'], future_motion_gt_fk['global_rotation']
        )
        dof_pos_loss = self.rec_criterion(future_motion_pred_fk['dof_pos'], future_motion_gt_fk['dof_pos'])
        dof_vel_loss = self.rec_criterion(future_motion_pred_fk['dof_vel'], future_motion_gt_fk['dof_vel'])

        foot_should_contact = future_motion_gt_fk['contact_mask'].unsqueeze(-1)
        foot_trans_pred = future_motion_pred_fk['global_translation_extend'][:, :, self.dataset.skeleton.foot_id, :]
        foot_trans_gt = future_motion_gt_fk['global_translation_extend'][:, :, self.dataset.skeleton.foot_id, :]
        foot_contact_loss = self.rec_criterion(
            foot_trans_pred * foot_should_contact, foot_trans_gt * foot_should_contact
        )
        '''temporal delta loss'''
        if history_motion is not None:
            pred_motion_tensor = torch.cat([history_motion[:, -1:, :], future_motion_pred], dim=1)
            pred_feature_dict = self.dataset.reconstruct_motion(pred_motion_tensor)
            pred_trans_delta = pred_feature_dict['delta_trans_world'][:, :-1, :]
            pred_joints_delta = pred_feature_dict['delta_joints_world'][:, :-1, :]
            pred_dof_delta = pred_feature_dict['delta_dof'][:, :-1, :]

            calc_trans_delta = pred_feature_dict['trans_pred'][:, 1:, :] - pred_feature_dict['trans_pred'][:, :-1, :]
            calc_joints_delta = pred_feature_dict['joints_pred'][:, 1:, :] - pred_feature_dict['joints_pred'][:, :-1, :]
            calc_dof_delta = pred_feature_dict['dof'][:, 1:, :] - pred_feature_dict['dof'][:, :-1, :]

            terms['trans_delta'] = self.rec_criterion(calc_trans_delta, pred_trans_delta)
            terms['joints_delta'] = self.rec_criterion(calc_joints_delta, pred_joints_delta)
            terms['dof_delta'] = self.rec_criterion(calc_dof_delta, pred_dof_delta)

        terms['body_trans'] = body_trans_loss
        terms['body_rot'] = body_rot_loss
        terms['dof_pos'] = dof_pos_loss
        terms['dof_vel'] = dof_vel_loss
        terms['foot_contact'] = foot_contact_loss

        return terms, extras


class MVAEManager(BaseManager, GeometryLoss):
    """
    Training manager for MVAE (Motion Variational AutoEncoder).

    Inherits from BaseManager for common training functionality and GeometryLoss
    for geometric loss computation capabilities.
    """

    vae: nn.Module
    optimizer: torch.optim.Optimizer

    static_prob: float

    def hold_model(self, vae, optimizer, dataset):
        self.vae = vae.to(self.device)
        self.optimizer = optimizer
        self.dataset = dataset
        logger.info("MVAEManager: Holding VAE model and optimizer")

        # 注册EMA模型
        self.register_ema_model('vae', self.vae)

        if self.ckpt.vae is not None:
            self.load_model(Path(self.ckpt.vae))

        # self.save_model()

    def calc_loss(self,
                  future_motion_gt,
                  future_motion_pred,
                  dist,
                  history_motion=None) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        terms = {}
        extras = {}

        # 重构损失
        rec_loss = self.rec_criterion(future_motion_pred, future_motion_gt)
        terms['rec'] = rec_loss

        # if self.loss_weight['smooth'] > 0.0:
        #     recon_diff = future_motion_pred[:, 1:, :] - future_motion_pred[:, :-1, :]
        #     true_diff = future_motion_gt[:, 1:, :] - future_motion_gt[:, :-1, :]
        #     terms['smooth'] = self.rec_criterion(recon_diff, true_diff)

        # KL损失
        mu_ref = torch.zeros_like(dist.loc)
        scale_ref = torch.ones_like(dist.scale)
        dist_ref = torch.distributions.Normal(mu_ref, scale_ref)
        kl_loss = torch.distributions.kl_divergence(dist, dist_ref)
        kl_loss = kl_loss.mean()
        terms['kl'] = kl_loss

        # 使用继承的几何损失计算方法
        if FeatureVersion == 4:
            geometry_terms, geometry_extras = self.calc_geometry_loss_v2(
                future_motion_pred, future_motion_gt, history_motion
            )
        elif FeatureVersion == 5:
            geometry_terms, geometry_extras = self.calc_geometry_loss_v3(
                future_motion_pred, future_motion_gt, history_motion
            )
        else:
            quantize = (self.loss_weight['quantize_rot'] > 0.0 or self.loss_weight['quantize_trans'] > 0.0)
            drift = (self.loss_weight['drift_xy'] > 0.0 or self.loss_weight['drift_yaw'] > 0.0)
            geometry_terms, geometry_extras = self.calc_geometry_loss(
                future_motion_pred,
                future_motion_gt,
                history_motion,
                smooth=self.loss_weight['smooth'] > 0.0,
                quantize=quantize,
                drift=drift
            )

        # geometry_terms = self.calc_geometry_loss_v2(future_motion_pred, future_motion_gt, history_motion)
        terms.update(geometry_terms)
        extras.update(geometry_extras)

        total_loss = sum(self.loss_weight[k] * v for k, v in terms.items())
        terms['total'] = total_loss
        return terms, extras

    def save_model(self) -> None:
        save_path = self.save_dir / f"ckpt_{self.step}.pth"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            'vae': self.vae.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': self.step,
        }

        # 保存EMA模型
        if self.use_ema and self.ema_models:
            save_dict['ema_models'] = {name: model.state_dict() for name, model in self.ema_models.items()}

        torch.save(save_dict, save_path)
        logger.info(f"Saved model & optimizer to {save_path}")
        logger.info(f"Current step: {self.step}")

    def load_model(self, ckpt_path: Path):
        state_dict = torch.load(ckpt_path)
        self.vae.load_state_dict(state_dict['vae'])
        if self.optimizer is not None:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        self.step = state_dict['step']

        # 加载EMA模型
        if self.use_ema and 'ema_models' in state_dict:
            for name, ema_state in state_dict['ema_models'].items():
                if name in self.ema_models:
                    self.ema_models[name].load_state_dict(ema_state)
                    logger.info(f"Loaded EMA model: {name}")

        logger.info(f"Loaded CKPT model & optimizer from {ckpt_path}")
        logger.info(f"CKPT step: {self.step}")

    def update_ema_models(self):
        """更新EMA模型"""
        self.update_ema('vae', self.vae)


class DARManager(BaseManager, GeometryLoss):
    """
    Training manager for DAR (Diffusion AutoRegressive) model.

    Manages both VAE and denoiser models, with support for full DDPM sampling
    and advanced training strategies.
    """
    vae: nn.Module
    denoiser: nn.Module

    static_prob: float
    use_full_sample: bool

    def hold_model(self, vae, denoiser, optimizer, dataset):
        self.vae = vae.to(self.device)
        self.denoiser = denoiser.to(self.device)
        self.optimizer = optimizer
        self.dataset = dataset

        logger.info("DARManager: Holding denoiser models and optimizer. VAE loaded from checkpoint.")

        # 注册EMA模型
        self.register_ema_model('denoiser', self.denoiser)

        if self.ckpt.dar is not None:
            self.load_model(Path(self.ckpt.dar))
            logger.info(f"Loaded DAR model & optimizer from {self.ckpt.dar}")

            old_vae_path = Path(self.ckpt.dar).parent / "vae.pth"
            assert old_vae_path.exists(), f"VAE checkpoint not found at {old_vae_path}"
            self.ckpt.vae = str(old_vae_path)

        # Search Logic
        # 1. Search cache path
        # 2. Search self.ckpt.vae
        # 3. Search nearby 'train-mvae-*'
        cache_vae_path = self.save_dir / "vae.pth"
        if not cache_vae_path.exists():
            if self.ckpt.vae is None:
                maybe_vae_path = self.try_search_vae_path()
                if not maybe_vae_path:
                    raise ValueError("VAE checkpoint path must be provided in ckpt.vae")
                self.ckpt.vae = str(maybe_vae_path)
                logger.warning(f"VAE checkpoint path not provided, using the searched one: {self.ckpt.vae}")
            if self.save_dir.exists():
                try:
                    # Hard Link, not soft link. It should be more safe
                    os.link(self.ckpt.vae, cache_vae_path)
                except OSError as e:
                    # If self.ckpt.vae and cache_vae_path lie in different filesystem,
                    # it will raise error.
                    if e.errno in (errno.EXDEV, errno.EPERM, errno.EACCES):
                        shutil.copy2(self.ckpt.vae, cache_vae_path)
                    else:
                        raise
                logger.info(f"VAE cached to {cache_vae_path}")
                vae_src_path = self.save_dir / "vae_src.log"
                with open(vae_src_path, "w") as f:
                    f.write(str(self.ckpt.vae))
            else:
                logger.warning(f"Save dir {self.save_dir} not exists, skip caching VAE")
                cache_vae_path = Path(self.ckpt.vae)

        self._load_vae_from_checkpoint(cache_vae_path)
        # self.save_model()

    def try_search_vae_path(self) -> Optional[Path]:
        exp_dir = self.save_dir.parent
        # 按修改时间排序（最新在前）
        mvae_dirs = sorted(exp_dir.glob("train-mvae-*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if len(mvae_dirs) > 1:
            logger.warning("Multiple MVAE checkpoints found, using the latest one")

        for mvae_dir in mvae_dirs:
            ckpt_files = sorted(mvae_dir.glob("ckpt_*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
            if ckpt_files:
                return ckpt_files[0]
        return None

    def _load_vae_from_checkpoint(self, ckpt_path: Path):
        """Load VAE from checkpoint following DART's approach"""
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        vae_state_dict = checkpoint['vae']

        if 'latent_mean' not in vae_state_dict:
            vae_state_dict['latent_mean'] = torch.tensor(0.0, device=self.device)
        if 'latent_std' not in vae_state_dict:
            vae_state_dict['latent_std'] = torch.tensor(1.0, device=self.device)

        self.vae.load_state_dict(vae_state_dict)

        self.vae.latent_mean = vae_state_dict['latent_mean']
        self.vae.latent_std = vae_state_dict['latent_std']

        # Freeze VAE parameters
        for param in self.vae.parameters():
            param.requires_grad = False
        self.vae.eval()

        logger.info(f"Loaded VAE from checkpoint: {ckpt_path}")
        logger.info(f"(latent_mean, latent_std): {self.vae.latent_mean}, {self.vae.latent_std}")

    def calc_loss(
        self,
        future_motion_gt,
        future_motion_pred,
        latent,
        dist=None,
        latent_pred=None,
        weights=None,
        history_motion=None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        terms = {}
        extras = {}

        # 重构损失
        rec_loss = self.rec_criterion(future_motion_pred, future_motion_gt)
        terms['rec'] = rec_loss

        # KL损失
        if dist is not None:
            mu_ref = torch.zeros_like(dist.loc)
            scale_ref = torch.ones_like(dist.scale)
            dist_ref = torch.distributions.Normal(mu_ref, scale_ref)
            kl_loss = torch.distributions.kl_divergence(dist, dist_ref)
            kl_loss = kl_loss.mean()
            terms['kl'] = kl_loss
        else:
            kl_loss = torch.tensor(0.0, device=future_motion_gt.device)
            terms['kl'] = kl_loss

        # latent重构损失
        if latent_pred is not None and latent is not None:
            latent_rec_loss = self.rec_criterion(latent_pred, latent)
            terms['latent_rec'] = latent_rec_loss
        else:
            latent_rec_loss = torch.tensor(0.0, device=future_motion_gt.device)
            terms['latent_rec'] = latent_rec_loss

        # 几何损失
        if FeatureVersion == 4:
            geometry_terms, geometry_extras = self.calc_geometry_loss_v2(
                future_motion_pred, future_motion_gt, history_motion
            )
        elif FeatureVersion == 5:
            geometry_terms, geometry_extras = self.calc_geometry_loss_v3(
                future_motion_pred, future_motion_gt, history_motion
            )
        else:
            quantize = (self.loss_weight['quantize_rot'] > 0.0 or self.loss_weight['quantize_trans'] > 0.0)
            drift = (self.loss_weight['drift_xy'] > 0.0 or self.loss_weight['drift_yaw'] > 0.0)
            geometry_terms, geometry_extras = self.calc_geometry_loss(
                future_motion_pred,
                future_motion_gt,
                history_motion,
                smooth=self.loss_weight['smooth'] > 0.0,
                quantize=quantize,
                drift=drift
            )

        # geometry_terms, geometry_extras = calc_geometry_loss(
        #     future_motion_pred, future_motion_gt, history_motion)
        # geometry_terms = self.calc_geometry_loss(future_motion_pred,
        #                                          future_motion_gt, history_motion=history_motion)

        # geometry_terms = self.calc_geometry_loss_v2(future_motion_pred, future_motion_gt, history_motion)
        terms.update(geometry_terms)
        extras.update(geometry_extras)

        total_loss = sum(self.loss_weight[k] * v for k, v in terms.items())

        # diffusion训练时可加权
        if weights is not None:
            total_loss = total_loss * weights.mean()

        terms['total'] = total_loss
        return terms, extras

    def save_model(self) -> None:
        save_path = self.save_dir / f"ckpt_{self.step}.pth"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            'denoiser': self.denoiser.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': self.step,
        }

        # 保存EMA模型
        if self.use_ema and self.ema_models:
            save_dict['ema_models'] = {name: model.state_dict() for name, model in self.ema_models.items()}

        torch.save(save_dict, save_path)
        logger.info(f"Saved DAR model & optimizer to {save_path}")
        logger.info(f"Current step: {self.step}")

    def load_model(self, ckpt_path: Path):
        state_dict = torch.load(ckpt_path, map_location=self.device)
        self.denoiser.load_state_dict(state_dict['denoiser'])
        if self.optimizer is not None:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        self.step = state_dict['step']

        # 加载EMA模型
        if self.use_ema and 'ema_models' in state_dict:
            for name, ema_state in state_dict['ema_models'].items():
                if name in self.ema_models:
                    self.ema_models[name].load_state_dict(ema_state)
                    logger.info(f"Loaded EMA model: {name}")

        logger.info(f"Loaded DAR model & optimizer from {ckpt_path}")
        logger.info(f"CKPT step: {self.step}")

    def update_ema_models(self):
        """更新EMA模型"""
        self.update_ema('denoiser', self.denoiser)

    def should_use_full_sample(self) -> bool:
        """判断是否使用完整DDPM采样"""
        if not self.use_full_sample:
            return False
        return self.stage_idx > 0
