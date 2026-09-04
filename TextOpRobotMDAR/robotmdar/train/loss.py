from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

import robotmdar.dtype.motion as motion_dtype
from robotmdar.dtype.motion import (
    G1_CORE_DOF_INDICES,
    G1_WRIST_DOF_INDICES,
)
from robotmdar.dtype.rotation import (
    matrix_to_rot6d,
    quaternion_to_matrix,
    rot6d_to_matrix,
    xyzw_to_wxyz,
)
from robotmdar.utils.goal import (
    GoalType,
    JOINT_STATE_GOAL_DIM,
    ROT_MAT_JOINT_STATE_GOAL_DIM,
    SPLIT_GOAL_DIM,
    SPLIT_HORIZONTAL_SLICE,
    SPLIT_JOINT_SLICE,
    SPLIT_ORIENTATION_SLICE,
    SPLIT_VELOCITY_SLICE,
    SPLIT_VERTICAL_HEIGHT_SLICE,
    V6_RAW_JOINT_SLICE,
    V6_RAW_ORIENTATION_SLICE,
    V6_RAW_POSITION_SLICE,
    V6_RAW_VELOCITY_SLICE,
)
from isaac_utils.rotations import get_euler_xyz

# Coarse motion classes for per-class diagnostics (doc §4.3.4/§4.3.6).
# The dataset's verb vocabulary is small and fixed (see data/g1_textop_29dof
# frame_ann); verbs outside these sets fall into 'unknown'.
_FAST_LOCOMOTION_VERBS = frozenset({'jog', 'run', 'sprint', 'jump', 'sport'})
_WALKING_SPEED_VERBS = frozenset(
    {'walk', 'step_over', 'carry', 'turn', 'push', 'injured'})
MOTION_CLASSES = ('walk', 'run', 'fall', 'getup', 'unknown')


def _motion_class_labels(action_label, is_recovery=None) -> List[str]:
    """Map per-sample BABEL verbs to coarse motion classes (doc §4.3.6).

    ``is_recovery`` marks dataset recovery-boosted segments (get-up
    dynamics), refining the get-up split regardless of the verb.
    """
    classes = []
    for i, verb in enumerate(action_label):
        if is_recovery is not None and bool(is_recovery[i]):
            classes.append('getup')
        else:
            verb = str(verb).lower()
            if verb == 'fall':
                classes.append('fall')
            elif verb in _FAST_LOCOMOTION_VERBS:
                classes.append('run')
            elif verb in _WALKING_SPEED_VERBS:
                classes.append('walk')
            else:
                classes.append('unknown')
    return classes


def _add_per_class_extras(extras, action_label, is_recovery, per_sample):
    """Log-only per-class masked means with a class suffix (doc §4.3.6).

    Per-sample error tensors are detached first: these extras never add
    graph nodes or gradient work (the cost is a handful of masked means
    per batch). Every class gets a key — classes absent from the batch
    log a zero — so the key set is deterministic per rank and the DDP
    eval-metrics all-reduce (ddp_reduce_mean) never diverges across
    ranks.
    """
    if not per_sample:
        return
    values = [v for v in per_sample.values() if v is not None]
    if not values:
        return
    labels = _motion_class_labels(action_label, is_recovery)
    device = values[0].device
    for key, err in per_sample.items():
        if err is None:
            continue
        err = err.detach()  # [B]
        if err.ndim != 1 or err.shape[0] != len(labels):
            raise ValueError(
                f'per-sample {key} must be [B] with B={len(labels)}, '
                f'got {tuple(err.shape)}'
            )
        for cls in MOTION_CLASSES:
            mask = torch.as_tensor(
                [label == cls for label in labels],
                dtype=torch.bool, device=device)
            extras[f'{key}__{cls}'] = (
                err[mask].mean() if mask.any()
                else torch.zeros((), device=device, dtype=err.dtype)
            )


def _standard_normal_kl_mean(dist) -> torch.Tensor:
    """KL(N(loc, scale) || N(0, 1)) without building reference tensors."""
    return 0.5 * (
        dist.loc.square() + dist.scale.square()
        - 1.0 - 2.0 * dist.scale.log()
    ).mean()


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

    def calc_foot_sliding_loss(self, foot_positions, contact_mask, fps,
                               sliding_mask=None):
        """Penalize predicted world-foot velocity on ground-truth stance frames."""
        if fps is None or fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        if foot_positions.shape[1] < 2:
            return foot_positions.sum() * 0.0
        foot_velocity = (
            foot_positions[:, 1:] - foot_positions[:, :-1]
        ) * fps
        stance = contact_mask[:, 1:] > 0.5
        if sliding_mask is not None:
            stance = stance & ~(sliding_mask[:, 1:] > 0.5)
        stance = stance.unsqueeze(-1).to(dtype=foot_velocity.dtype)
        masked_velocity = foot_velocity * stance
        return self.rec_criterion(masked_velocity, torch.zeros_like(masked_velocity))

    @staticmethod
    def calc_sliding_ratio(contact_mask, sliding_mask):
        if sliding_mask is None:
            return contact_mask.new_zeros(())
        sliding = sliding_mask.sum()
        return sliding / (contact_mask.sum() + sliding + 1e-8)

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
        endpoint=False,
        sliding_mask=None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """计算几何损失"""
        terms = {}
        extras = {}

        if smooth and history_motion is not None:
            motion_tensor = torch.cat([history_motion, future_motion_pred], dim=1)
            diff = motion_tensor[:, 1:, :] - motion_tensor[:, :-1, :]
            terms['smooth'] = torch.abs(diff).mean()

        # Geometric loss
        future_motion_pred_fk = self.dataset.reconstruct_motion(future_motion_pred, need_denormalize=True, ret_fk=True)
        with torch.no_grad():
            future_motion_gt_fk = self.dataset.reconstruct_motion(
                future_motion_gt, need_denormalize=True, ret_fk=True,
                sliding_mask=sliding_mask)

        body_trans_loss = self.rec_criterion(
            future_motion_pred_fk['global_translation_extend'], future_motion_gt_fk['global_translation_extend']
        )
        body_rot_loss = self.rec_criterion(
            future_motion_pred_fk['global_rotation'], future_motion_gt_fk['global_rotation']
        )
        dof_dim = int(self.dataset.dof_dim)
        for label, fk_result in (
            ('prediction', future_motion_pred_fk),
            ('ground truth', future_motion_gt_fk),
        ):
            expected_shape = (*fk_result['dof_pos'].shape[:-1], dof_dim)
            if tuple(fk_result['dof_pos'].shape) != expected_shape:
                raise RuntimeError(
                    f"{label} dof_pos must be [B, T, {dof_dim}], got "
                    f"{tuple(fk_result['dof_pos'].shape)}"
                )
            if tuple(fk_result['dof_vel'].shape) != expected_shape:
                raise RuntimeError(
                    f"{label} dof_vel must match dof_pos shape "
                    f"{expected_shape}, got {tuple(fk_result['dof_vel'].shape)}"
                )
        dof_pos_loss = self.rec_criterion(future_motion_pred_fk['dof_pos'], future_motion_gt_fk['dof_pos'])
        dof_vel_loss = self.rec_criterion(future_motion_pred_fk['dof_vel'], future_motion_gt_fk['dof_vel'])

        if dof_dim == 29:
            core_ids = torch.as_tensor(
                G1_CORE_DOF_INDICES,
                device=future_motion_pred_fk['dof_pos'].device,
            )
            wrist_ids = torch.as_tensor(
                G1_WRIST_DOF_INDICES,
                device=future_motion_pred_fk['dof_pos'].device,
            )
            extras['dof_pos_core'] = self.rec_criterion(
                future_motion_pred_fk['dof_pos'].index_select(-1, core_ids),
                future_motion_gt_fk['dof_pos'].index_select(-1, core_ids),
            )
            extras['dof_pos_wrist'] = self.rec_criterion(
                future_motion_pred_fk['dof_pos'].index_select(-1, wrist_ids),
                future_motion_gt_fk['dof_pos'].index_select(-1, wrist_ids),
            )
            extras['dof_vel_core'] = self.rec_criterion(
                future_motion_pred_fk['dof_vel'].index_select(-1, core_ids),
                future_motion_gt_fk['dof_vel'].index_select(-1, core_ids),
            )
            extras['dof_vel_wrist'] = self.rec_criterion(
                future_motion_pred_fk['dof_vel'].index_select(-1, wrist_ids),
                future_motion_gt_fk['dof_vel'].index_select(-1, wrist_ids),
            )
        extras['hand_translation'] = self.rec_criterion(
            future_motion_pred_fk['global_translation_extend'][
                :, :, self.dataset.skeleton.hand_id, :
            ],
            future_motion_gt_fk['global_translation_extend'][
                :, :, self.dataset.skeleton.hand_id, :
            ],
        )

        foot_trans_pred = future_motion_pred_fk['global_translation_extend'][:, :, self.dataset.skeleton.foot_id, :]
        foot_contact_loss = self.calc_foot_sliding_loss(
            foot_trans_pred,
            future_motion_gt_fk['contact_mask'],
            fps=self.dataset.fps,
            sliding_mask=future_motion_gt_fk.get('sliding_mask'),
        )
        extras['sliding_ratio'] = self.calc_sliding_ratio(
            future_motion_gt_fk['contact_mask'],
            future_motion_gt_fk.get('sliding_mask'))

        if quantize:
            quantize_pred_rot = self.quantization(future_motion_pred_fk['global_rotation'][:, -1, 0])
            quantize_gt_rot = self.quantization(future_motion_gt_fk['global_rotation'][:, -1, 0])

            quantize_pred_trans_xy = self.quantization(future_motion_pred_fk['global_translation_extend'][:, :, 0, :2])
            quantize_gt_trans_xy = self.quantization(future_motion_gt_fk['global_translation_extend'][:, :, 0, :2])
            terms['quantize_rot'] = self.rec_criterion(quantize_pred_rot, quantize_gt_rot)
            terms['quantize_trans'] = self.rec_criterion(quantize_pred_trans_xy, quantize_gt_trans_xy)

        endpoint_yaw_pred = get_euler_xyz(future_motion_pred_fk['global_rotation'][:, -1, 0], w_last=True)[2]  # (B,)
        endpoint_yaw_gt = get_euler_xyz(future_motion_gt_fk['global_rotation'][:, -1, 0], w_last=True)[2]  # (B,)

        endpoint_yaw_diff = (endpoint_yaw_pred - endpoint_yaw_gt) % (2 * torch.pi)
        endpoint_yaw_diff[endpoint_yaw_diff > torch.pi] -= 2 * torch.pi
        endpoint_yaw_loss = self.rec_criterion(endpoint_yaw_diff, torch.zeros_like(endpoint_yaw_diff))

        endpoint_xy_loss = self.rec_criterion(
            future_motion_pred_fk['global_translation_extend'][:, -1, 0, :2],
            future_motion_gt_fk['global_translation_extend'][:, -1, 0, :2]
        )

        if endpoint:
            terms['endpoint_yaw'] = endpoint_yaw_loss
            terms['endpoint_xy'] = endpoint_xy_loss

        terms['body_trans'] = body_trans_loss
        terms['body_rot'] = body_rot_loss
        terms['dof_pos'] = dof_pos_loss
        terms['dof_vel'] = dof_vel_loss
        terms['foot_contact'] = foot_contact_loss

        extras['endpoint_xy'] = endpoint_xy_loss
        extras['endpoint_yaw'] = endpoint_yaw_loss
        # if smooth:
        #     jerk_loss = self.calc_jerk(
        #         future_motion_pred_fk['global_translation_extend'])
        #     terms['smooth'] = jerk_loss

        return terms, extras

    def _feature_v6_components(self, motion_feature):
        denorm = self.dataset.denormalize(motion_feature)
        dof_dim = int(self.dataset.dof_dim)
        gravity_raw = denorm[..., 1:4]
        gravity = torch.nn.functional.normalize(gravity_raw, dim=-1, eps=1e-8)
        delta_hor_raw = denorm[..., 4:7]
        rel_rot6d = denorm[..., 7:13]
        rel_rot = rot6d_to_matrix(rel_rot6d)
        return {
            'height': denorm[..., 0],
            'gravity_raw': gravity_raw,
            'gravity': gravity,
            'delta_hor_raw': delta_hor_raw,
            'rel_rot6d': rel_rot6d,
            'rel_rot': rel_rot,
            'dof': denorm[..., 13:13 + dof_dim],
        }

    def _foot_support_mask_from_contact(self, contact_mask, sliding_mask=None):
        if contact_mask is None:
            return None
        if contact_mask.ndim != 3:
            raise ValueError(
                "contact_mask must be [B, T, C], got "
                f"{tuple(contact_mask.shape)}"
            )
        support = contact_mask > 0.5
        if sliding_mask is not None:
            if sliding_mask.shape != contact_mask.shape:
                raise ValueError(
                    "sliding_mask must match contact_mask shape: "
                    f"{tuple(sliding_mask.shape)} != {tuple(contact_mask.shape)}"
                )
            support = support & ~(sliding_mask > 0.5)
        return support

    def _hand_support_mask_from_fk(
        self,
        fk_result,
        is_recovery=None,
        height_thresh: float = 0.12,
        speed_thresh: float = 0.35,
    ):
        hand_ids = getattr(self.dataset.skeleton, 'hand_id', None)
        if not hand_ids:
            return None
        if is_recovery is None:
            return None
        recovery = torch.as_tensor(
            is_recovery,
            device=fk_result['global_translation_extend'].device,
            dtype=torch.bool,
        ).reshape(-1)
        if not recovery.any():
            return None
        hand_ids = torch.as_tensor(
            hand_ids,
            device=fk_result['global_translation_extend'].device,
            dtype=torch.long,
        )
        hand_pos = fk_result['global_translation_extend'].index_select(2, hand_ids)
        hand_height = hand_pos[..., 2]
        hand_speed = hand_pos.new_zeros(hand_pos.shape[:-1])
        if hand_pos.shape[1] > 1:
            hand_speed[:, 1:] = (
                hand_pos[:, 1:] - hand_pos[:, :-1]
            ).norm(dim=-1) * float(self.dataset.fps)
        support = (
            (hand_height < height_thresh)
            & (hand_speed < speed_thresh)
        )
        support = support & recovery[:, None, None]
        return support

    def _support_component_from_fk(self, fk_result, support_body_ids,
                                   support_mask):
        """Return support residual loss plus diagnostic summaries.

        support_mask is expected to be [B, T, K]; the active support frames are
        the adjacent pairs where the mask stays on for both frames.
        """
        zero = fk_result['global_translation_extend'].sum() * 0.0
        zero_metric = zero.detach()
        zero_per_sample = fk_result['global_translation_extend'].sum(
            dim=(1, 2, 3)
        ) * 0.0
        zero_per_sample = zero_per_sample.detach()
        if support_mask is None:
            return zero, zero_metric, zero_per_sample, zero_metric
        if support_mask.ndim != 3:
            raise ValueError(
                "support_mask must be [B, T, K], got "
                f"{tuple(support_mask.shape)}"
            )
        if fk_result['global_translation_extend'].shape[1] < 2:
            return zero, zero_metric, zero_per_sample, zero_metric

        global_translation = fk_result['global_translation_extend']
        global_rotation = fk_result['global_rotation']
        support_mask = support_mask.to(
            device=global_translation.device, dtype=torch.bool)
        support_body_ids = torch.as_tensor(
            support_body_ids,
            device=global_translation.device,
            dtype=torch.long,
        )
        if support_mask.shape[0] != global_translation.shape[0]:
            raise ValueError(
                "support_mask batch size must match FK result: "
                f"{support_mask.shape[0]} != {global_translation.shape[0]}"
            )
        if support_mask.shape[1] != global_translation.shape[1]:
            raise ValueError(
                "support_mask time dimension must match FK result: "
                f"{support_mask.shape[1]} != {global_translation.shape[1]}"
            )
        if support_mask.shape[-1] != support_body_ids.numel():
            raise ValueError(
                "support_mask channel count must match support_body_ids: "
                f"{support_mask.shape[-1]} != {support_body_ids.numel()}"
            )

        active = support_mask[:, 1:] & support_mask[:, :-1]
        if not active.any():
            return zero, zero_metric, zero_per_sample, zero_metric

        root_pos = global_translation[:, :, 0, :]
        root_rot = quaternion_to_matrix(
            xyzw_to_wxyz(global_rotation[:, :, 0, :])
        )
        support_pos = global_translation.index_select(2, support_body_ids)
        local_pos = torch.matmul(
            root_rot.transpose(-1, -2).unsqueeze(-3),
            (support_pos - root_pos.unsqueeze(-2)).unsqueeze(-1),
        ).squeeze(-1)

        root_delta_local = torch.matmul(
            root_rot[:, :-1].transpose(-1, -2),
            (root_pos[:, 1:] - root_pos[:, :-1]).unsqueeze(-1),
        ).squeeze(-1)
        rel_rot = torch.matmul(
            root_rot[:, :-1].transpose(-1, -2),
            root_rot[:, 1:],
        )
        transported = torch.matmul(
            rel_rot.unsqueeze(-3),
            local_pos[:, 1:].unsqueeze(-1),
        ).squeeze(-1)
        residual = root_delta_local.unsqueeze(-2) - (
            local_pos[:, :-1] - transported
        )

        active_vectors = residual[active]
        if active_vectors.numel() == 0:
            return zero, zero_metric, zero_per_sample, zero_metric

        loss = self.rec_criterion(active_vectors, torch.zeros_like(active_vectors))
        metric = active_vectors.norm(dim=-1).mean()

        residual_norm = residual.detach().norm(dim=-1)
        active_float = active.to(dtype=residual_norm.dtype)
        per_sample = (
            residual_norm * active_float
        ).sum(dim=(1, 2)) / active_float.sum(dim=(1, 2)).clamp_min(1.0)
        active_ratio = active_float.mean()
        return loss, metric, per_sample, active_ratio

    @staticmethod
    def _transport_gravity(prev_g, rel_rot):
        return torch.matmul(
            rel_rot.transpose(-1, -2),
            prev_g.unsqueeze(-1),
        ).squeeze(-1)

    @staticmethod
    def _project_to_tangent(vector, gravity):
        gravity = torch.nn.functional.normalize(gravity, dim=-1, eps=1e-8)
        return vector - (vector * gravity).sum(dim=-1, keepdim=True) * gravity

    @staticmethod
    def _v6_identity(batch_size, device, dtype):
        return torch.eye(3, device=device, dtype=dtype).expand(
            batch_size, 3, 3).clone()

    def _future_goal_state_v6(self, future_motion_pred, history_motion,
                              goal_time_frame=None):
        if history_motion is None:
            raise ValueError(
                "history_motion is required for v6 goal-state losses")
        pred = self._feature_v6_components(future_motion_pred)
        future_motion = self.dataset.denormalize(future_motion_pred)
        hist = self._feature_v6_components(history_motion[:, -1:])

        B, T = future_motion.shape[:2]
        device, dtype = future_motion.device, future_motion.dtype
        ref_g = hist['gravity'][:, 0]
        ref_height = hist['height'][:, 0]
        dep_gravity = torch.cat(
            (ref_g[:, None], pred['gravity'][:, :-1]), dim=1)
        delta_hor = self._project_to_tangent(
            pred['delta_hor_raw'], dep_gravity)

        prev_R = self._v6_identity(B, device, dtype)
        prev_height = ref_height
        displacement = torch.zeros(B, 3, device=device, dtype=dtype)
        trajectory = [displacement]
        rotations = []
        velocities = []

        for t in range(T):
            dep_R = prev_R
            step_delta_hor = delta_hor[:, t]
            displacement = displacement + torch.matmul(
                dep_R, step_delta_hor.unsqueeze(-1)).squeeze(-1)
            trajectory.append(self._project_to_tangent(displacement, ref_g))

            dh = pred['height'][:, t] - prev_height
            delta_local = step_delta_hor - dh.unsqueeze(-1) * dep_gravity[:, t]
            velocity_ref = torch.matmul(
                dep_R, delta_local.unsqueeze(-1)).squeeze(-1)
            velocity_ref = velocity_ref * float(self.dataset.fps)
            velocity_hor = self._project_to_tangent(velocity_ref, ref_g)
            velocity_vert = -(velocity_ref * ref_g).sum(dim=-1, keepdim=True)
            velocities.append(torch.cat((velocity_hor, velocity_vert), dim=-1))

            provisional_R = torch.matmul(dep_R, pred['rel_rot'][:, t])
            integrated_gravity = torch.matmul(
                provisional_R.transpose(-1, -2),
                ref_g.unsqueeze(-1),
            ).squeeze(-1)
            correction = motion_dtype._shortest_arc_right_correction(
                integrated_gravity, pred['gravity'][:, t])
            prev_R = torch.matmul(provisional_R, correction)
            rotations.append(prev_R)
            prev_height = pred['height'][:, t]

        trajectory = torch.stack(trajectory, dim=1)
        rotations = torch.stack(rotations, dim=1)
        velocities = torch.stack(velocities, dim=1)
        goal_step = self._future_step_from_goal_time(
            future_motion_pred, goal_time_frame)
        batch_idx = torch.arange(B, device=device)
        return {
            'future_motion': future_motion,
            'goal_step': goal_step,
            'batch_idx': batch_idx,
            'selected': future_motion[batch_idx, goal_step],
            'height_at_goal': pred['height'][batch_idx, goal_step],
            'trajectory': trajectory,
            'displacement_at_goal': trajectory[
                batch_idx, (goal_step + 1).clamp(max=trajectory.shape[1] - 1)],
            'rel_rot_at_goal': rotations[batch_idx, goal_step],
            'velocity_at_goal': velocities[batch_idx, goal_step],
        }

    def _history_trajectory_ego_v6(self, history_motion):
        history = self._feature_v6_components(history_motion)
        B, H = history_motion.shape[:2]
        device, dtype = history_motion.device, history_motion.dtype
        if H == 0:
            return torch.zeros(B, 0, 3, device=device, dtype=dtype)

        first_g = history['gravity'][:, 0]
        prev_g = torch.matmul(
            history['rel_rot'][:, 0],
            first_g.unsqueeze(-1),
        ).squeeze(-1)
        prev_R = self._v6_identity(B, device, dtype)
        delta_hor = self._project_to_tangent(
            history['delta_hor_raw'],
            torch.cat((prev_g[:, None], history['gravity'][:, :-1]), dim=1),
        )

        displacement = torch.zeros(B, 3, device=device, dtype=dtype)
        positions = [displacement]
        rotations = []
        for t in range(H):
            dep_R = prev_R
            displacement = displacement + torch.matmul(
                dep_R, delta_hor[:, t].unsqueeze(-1)).squeeze(-1)
            positions.append(displacement)

            provisional_R = torch.matmul(dep_R, history['rel_rot'][:, t])
            integrated_gravity = torch.matmul(
                provisional_R.transpose(-1, -2),
                prev_g.unsqueeze(-1),
            ).squeeze(-1)
            correction = motion_dtype._shortest_arc_right_correction(
                integrated_gravity, history['gravity'][:, t])
            prev_R = torch.matmul(provisional_R, correction)
            rotations.append(prev_R)

        positions = torch.stack(positions, dim=1)
        ref_R = torch.stack(rotations, dim=1)[:, -1]
        d_pos = positions[:, 1:] - positions[:, -1:]
        ego_pos = torch.matmul(
            ref_R.transpose(-1, -2).unsqueeze(1),
            d_pos.unsqueeze(-1),
        ).squeeze(-1)
        ref_g = history['gravity'][:, -1]
        return self._project_to_tangent(ego_pos, ref_g.unsqueeze(1))

    @staticmethod
    def _quat_chordal_loss(q_pred, q_gt):
        q_pred = torch.nn.functional.normalize(q_pred, dim=-1, eps=1e-8)
        q_gt = torch.nn.functional.normalize(q_gt, dim=-1, eps=1e-8)
        dot = (q_pred * q_gt).sum(dim=-1).clamp(-1.0, 1.0)
        return (8.0 * (1.0 - dot.square())).mean()

    def calc_geometry_loss_v6(
        self,
        future_motion_pred,
        future_motion_gt,
        history_motion=None,
        smooth=False,
        sliding_mask=None,
        action_label=None,
        is_recovery=None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Geometry losses for the gravity + relative-rotation v6 feature."""
        terms = {}
        extras = {}

        if smooth and history_motion is not None:
            motion_tensor = torch.cat([history_motion, future_motion_pred], dim=1)
            diff = motion_tensor[:, 1:, :] - motion_tensor[:, :-1, :]
            terms['smooth'] = torch.abs(diff).mean()

        zero = future_motion_pred.sum() * 0.0
        zero_metric = zero.detach()
        per_sample = {}
        future_motion_pred_fk = self.dataset.reconstruct_motion(
            future_motion_pred, need_denormalize=True, ret_fk=True
        )
        with torch.no_grad():
            future_motion_gt_fk = self.dataset.reconstruct_motion(
                future_motion_gt, need_denormalize=True, ret_fk=True,
                sliding_mask=sliding_mask,
            )

        body_trans_loss = self.rec_criterion(
            future_motion_pred_fk['global_translation_extend'],
            future_motion_gt_fk['global_translation_extend'],
        )
        body_rot_loss = self._quat_chordal_loss(
            future_motion_pred_fk['global_rotation'],
            future_motion_gt_fk['global_rotation'],
        )

        dof_dim = int(self.dataset.dof_dim)
        for label, fk_result in (
            ('prediction', future_motion_pred_fk),
            ('ground truth', future_motion_gt_fk),
        ):
            expected_shape = (*fk_result['dof_pos'].shape[:-1], dof_dim)
            if tuple(fk_result['dof_pos'].shape) != expected_shape:
                raise RuntimeError(
                    f"{label} dof_pos must be [B, T, {dof_dim}], got "
                    f"{tuple(fk_result['dof_pos'].shape)}"
                )
            if tuple(fk_result['dof_vel'].shape) != expected_shape:
                raise RuntimeError(
                    f"{label} dof_vel must match dof_pos shape "
                    f"{expected_shape}, got {tuple(fk_result['dof_vel'].shape)}"
                )
        dof_pos_loss = self.rec_criterion(
            future_motion_pred_fk['dof_pos'], future_motion_gt_fk['dof_pos']
        )
        dof_vel_loss = self.rec_criterion(
            future_motion_pred_fk['dof_vel'], future_motion_gt_fk['dof_vel']
        )

        if dof_dim == 29:
            core_ids = torch.as_tensor(
                G1_CORE_DOF_INDICES,
                device=future_motion_pred_fk['dof_pos'].device,
            )
            wrist_ids = torch.as_tensor(
                G1_WRIST_DOF_INDICES,
                device=future_motion_pred_fk['dof_pos'].device,
            )
            with torch.no_grad():
                pred_dof_pos = future_motion_pred_fk['dof_pos'].detach()
                gt_dof_pos = future_motion_gt_fk['dof_pos']
                pred_dof_vel = future_motion_pred_fk['dof_vel'].detach()
                gt_dof_vel = future_motion_gt_fk['dof_vel']
                extras['dof_pos_core'] = self.rec_criterion(
                    pred_dof_pos.index_select(-1, core_ids),
                    gt_dof_pos.index_select(-1, core_ids),
                )
                extras['dof_pos_wrist'] = self.rec_criterion(
                    pred_dof_pos.index_select(-1, wrist_ids),
                    gt_dof_pos.index_select(-1, wrist_ids),
                )
                extras['dof_vel_core'] = self.rec_criterion(
                    pred_dof_vel.index_select(-1, core_ids),
                    gt_dof_vel.index_select(-1, core_ids),
                )
                extras['dof_vel_wrist'] = self.rec_criterion(
                    pred_dof_vel.index_select(-1, wrist_ids),
                    gt_dof_vel.index_select(-1, wrist_ids),
                )

        with torch.no_grad():
            extras['hand_translation'] = self.rec_criterion(
                future_motion_pred_fk['global_translation_extend'][
                    :, :, self.dataset.skeleton.hand_id, :
                ].detach(),
                future_motion_gt_fk['global_translation_extend'][
                    :, :, self.dataset.skeleton.hand_id, :
                ],
            )

        foot_trans_pred = future_motion_pred_fk['global_translation_extend'][
            :, :, self.dataset.skeleton.foot_id, :
        ]
        foot_contact_loss = self.calc_foot_sliding_loss(
            foot_trans_pred,
            future_motion_gt_fk['contact_mask'],
            fps=self.dataset.fps,
            sliding_mask=future_motion_gt_fk.get('sliding_mask'),
        )
        extras['sliding_ratio'] = self.calc_sliding_ratio(
            future_motion_gt_fk['contact_mask'],
            future_motion_gt_fk.get('sliding_mask'),
        )

        foot_support_mask = self._foot_support_mask_from_contact(
            future_motion_gt_fk['contact_mask'],
            future_motion_gt_fk.get('sliding_mask'),
        )
        support_losses = []
        support_metrics = []
        support_per_sample = []
        support_ratios = {}
        foot_active = False
        hand_active = False

        if foot_support_mask is not None:
            foot_pred_loss, foot_pred_metric, foot_pred_per_sample, foot_ratio = (
                self._support_component_from_fk(
                    future_motion_pred_fk,
                    self.dataset.skeleton.foot_id,
                    foot_support_mask,
                )
            )
            with torch.no_grad():
                _, foot_gt_metric, _, _ = self._support_component_from_fk(
                    future_motion_gt_fk,
                    self.dataset.skeleton.foot_id,
                    foot_support_mask,
                )
            foot_active = float(foot_ratio) > 0.0
            if foot_active:
                support_losses.append(foot_pred_loss)
                support_metrics.append(foot_pred_metric.detach())
                support_per_sample.append(foot_pred_per_sample)
            support_ratios['foot_support_active_ratio'] = foot_ratio
            extras['e_support_foot'] = foot_pred_metric.detach()
            extras['e_support_foot_gt'] = foot_gt_metric

        hand_support_mask = self._hand_support_mask_from_fk(
            future_motion_gt_fk,
            is_recovery=is_recovery,
        )
        if hand_support_mask is not None:
            hand_pred_loss, hand_pred_metric, hand_pred_per_sample, hand_ratio = (
                self._support_component_from_fk(
                    future_motion_pred_fk,
                    self.dataset.skeleton.hand_id,
                    hand_support_mask,
                )
            )
            with torch.no_grad():
                _, hand_gt_metric, _, _ = self._support_component_from_fk(
                    future_motion_gt_fk,
                    self.dataset.skeleton.hand_id,
                    hand_support_mask,
                )
            hand_active = float(hand_ratio) > 0.0
            if hand_active:
                support_losses.append(hand_pred_loss)
                support_metrics.append(hand_pred_metric.detach())
                support_per_sample.append(hand_pred_per_sample)
            support_ratios['hand_support_active_ratio'] = hand_ratio
            extras['e_support_hand'] = hand_pred_metric.detach()
            extras['e_support_hand_gt'] = hand_gt_metric

        if support_losses:
            terms['support_consistency'] = (
                sum(support_losses) / float(len(support_losses))
            )
            extras['e_support_consistency'] = (
                sum(support_metrics) / float(len(support_metrics))
            )
            per_sample['e_support_consistency'] = (
                sum(support_per_sample) / float(len(support_per_sample))
            )
            with torch.no_grad():
                gt_support_metrics = []
                if foot_active:
                    gt_support_metrics.append(foot_gt_metric)
                if hand_active:
                    gt_support_metrics.append(hand_gt_metric)
                extras['e_support_consistency_gt'] = (
                    sum(gt_support_metrics) / float(len(gt_support_metrics))
                    if gt_support_metrics else zero_metric
                )
        else:
            terms['support_consistency'] = zero
            extras['e_support_consistency'] = zero_metric
            extras['e_support_consistency_gt'] = zero_metric
            per_sample['e_support_consistency'] = future_motion_pred.new_zeros(
                future_motion_pred.shape[0]
            )
        extras.update(support_ratios)

        pred = self._feature_v6_components(future_motion_pred)
        with torch.no_grad():
            gt = self._feature_v6_components(future_motion_gt)

        terms['rot_chord'] = (
            pred['rel_rot'] - gt['rel_rot']
        ).square().sum(dim=(-1, -2)).mean()

        residuals = []
        if history_motion is not None:
            hist = self._feature_v6_components(history_motion[:, -1:])
            boundary = (
                pred['gravity'][:, 0]
                - self._transport_gravity(hist['gravity'][:, 0], pred['rel_rot'][:, 0])
            )
            residuals.append(boundary.unsqueeze(1))
            extras['e_g_boundary'] = boundary.detach().norm(dim=-1).mean()
        if pred['gravity'].shape[1] > 1:
            interior = (
                pred['gravity'][:, 1:]
                - self._transport_gravity(
                    pred['gravity'][:, :-1],
                    pred['rel_rot'][:, :-1],
                )
            )
            residuals.append(interior)
            interior_norm = interior.detach().norm(dim=-1)  # [B, T-1]
            extras['e_g_cons'] = interior_norm.mean()
            per_sample['e_g_cons'] = interior_norm.mean(dim=1)  # [B]
        else:
            extras['e_g_cons'] = zero_metric
        terms['g_cons'] = (
            torch.cat(residuals, dim=1).square().sum(dim=-1).mean()
            if residuals else zero
        )

        if pred['height'].shape[1] > 1:
            fps = float(self.dataset.fps)
            pred_h_vel = (pred['height'][:, 1:] - pred['height'][:, :-1]) * fps
            gt_h_vel = (gt['height'][:, 1:] - gt['height'][:, :-1]) * fps
            terms['h_vel'] = self.rec_criterion(pred_h_vel, gt_h_vel)
            per_h = (
                pred_h_vel.detach() - gt_h_vel
            ).abs().mean(dim=1)  # [B]
            extras['e_h_vel'] = per_h.mean()

            pred_q_vel = (pred['dof'][:, 1:] - pred['dof'][:, :-1]) * fps
            gt_q_vel = (gt['dof'][:, 1:] - gt['dof'][:, :-1]) * fps
            per_q = (
                pred_q_vel.detach() - gt_q_vel
            ).norm(dim=-1).mean(dim=1)  # [B]
            extras['e_q_vel'] = per_q.mean()

            per_sample['e_h_vel'] = per_h
            per_sample['e_q_vel'] = per_q
        else:
            terms['h_vel'] = zero
            extras['e_h_vel'] = zero_metric
            extras['e_q_vel'] = zero_metric

        with torch.no_grad():
            extras['e_g_proj'] = (
                pred['gravity_raw'].detach() - pred['gravity'].detach()
            ).norm(dim=-1).mean()
            extras['e_R_proj'] = (
                pred['rel_rot6d'].detach()
                - matrix_to_rot6d(pred['rel_rot'].detach())
            ).norm(dim=-1).mean()
            if history_motion is not None:
                dep_gravity = torch.cat(
                    [
                        hist['gravity'][:, -1:].detach(),
                        pred['gravity'][:, :-1].detach(),
                    ],
                    dim=1,
                )
            else:
                dep_gravity = torch.cat(
                    [
                        pred['gravity'][:, :1].detach(),
                        pred['gravity'][:, :-1].detach(),
                    ],
                    dim=1,
                )
            extras['e_p_proj'] = (
                pred['delta_hor_raw'].detach() * dep_gravity
            ).sum(dim=-1).abs().mean()

            endpoint_xy_loss = self.rec_criterion(
                future_motion_pred_fk['global_translation_extend'][
                    :, -1, 0, :2
                ].detach(),
                future_motion_gt_fk['global_translation_extend'][:, -1, 0, :2],
            )
            extras['endpoint_xy'] = endpoint_xy_loss
            extras['e_R_endpoint'] = self._quat_chordal_loss(
                future_motion_pred_fk['global_rotation'][:, -1:, 0].detach(),
                future_motion_gt_fk['global_rotation'][:, -1:, 0],
            )

        if action_label is not None:
            _add_per_class_extras(
                extras, action_label, is_recovery, per_sample)

        terms['body_trans'] = body_trans_loss
        terms['body_rot'] = body_rot_loss
        terms['dof_pos'] = dof_pos_loss
        terms['dof_vel'] = dof_vel_loss
        terms['foot_contact'] = foot_contact_loss

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
                              history_motion=None,
                              sliding_mask=None) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
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
                ret_fk=True,
                sliding_mask=sliding_mask,
            )

        body_trans_loss = self.rec_criterion(
            future_motion_pred_fk['global_translation_extend'], future_motion_gt_fk['global_translation_extend']
        )  # [B=512, T=8, L=27, 3]
        body_rot_loss = self.rec_criterion(
            future_motion_pred_fk['global_rotation'], future_motion_gt_fk['global_rotation']
        )
        dof_pos_loss = self.rec_criterion(future_motion_pred_fk['dof_pos'], future_motion_gt_fk['dof_pos'])
        dof_vel_loss = self.rec_criterion(future_motion_pred_fk['dof_vel'], future_motion_gt_fk['dof_vel'])

        foot_trans_pred = future_motion_pred_fk['global_translation_extend'][:, :, self.dataset.skeleton.foot_id, :]
        foot_contact_loss = self.calc_foot_sliding_loss(
            foot_trans_pred,
            future_motion_gt_fk['contact_mask'],
            fps=self.dataset.fps,
            sliding_mask=future_motion_gt_fk.get('sliding_mask'),
        )
        extras['sliding_ratio'] = self.calc_sliding_ratio(
            future_motion_gt_fk['contact_mask'],
            future_motion_gt_fk.get('sliding_mask'))
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

    def calc_goal_root_position_loss(self, future_motion_pred, ego_goal,
                                     goal_condition_keep_mask=None,
                                     history_motion=None,
                                     goal_time_frame=None):
        """Match generated horizontal root displacement to the goal endpoint."""
        if (motion_dtype.FeatureVersion == 6
                and ego_goal.shape[-1]
                in (ROT_MAT_JOINT_STATE_GOAL_DIM, SPLIT_GOAL_DIM)):
            goal_state = self._future_goal_state_v6(
                future_motion_pred, history_motion,
                goal_time_frame=goal_time_frame)
            predicted = torch.cat(
                (
                    goal_state['height_at_goal'].unsqueeze(-1),
                    goal_state['displacement_at_goal'],
                ),
                dim=-1,
            )
            if ego_goal.shape[-1] == SPLIT_GOAL_DIM:
                vertical = ego_goal[..., SPLIT_VERTICAL_HEIGHT_SLICE]
                horizontal = ego_goal[..., SPLIT_HORIZONTAL_SLICE]
                target = torch.cat(
                    (vertical[..., :1], horizontal[..., :3]),
                    dim=-1,
                )
            else:
                target = ego_goal[..., V6_RAW_POSITION_SLICE]
            valid = torch.ones(
                target.shape[0], dtype=torch.bool, device=target.device)
            if goal_condition_keep_mask is not None:
                valid = valid & goal_condition_keep_mask.to(
                    device=valid.device, dtype=torch.bool)
            if not valid.any():
                return future_motion_pred.sum() * 0.0
            return self.rec_criterion(predicted[valid], target[valid])

        root_displacement = self.root_displacement_ego(
            future_motion_pred, history_motion,
            goal_time_frame=goal_time_frame)
        goal_root_position = ego_goal[..., :2]
        valid = torch.ones(
            goal_root_position.shape[0], dtype=torch.bool,
            device=goal_root_position.device)
        if goal_condition_keep_mask is not None:
            valid = valid & goal_condition_keep_mask.to(
                device=valid.device, dtype=torch.bool
            )
        if not valid.any():
            return future_motion_pred.sum() * 0.0
        return self.rec_criterion(
            root_displacement[valid], goal_root_position[valid])

    def _valid_goal_component_mask(self, batch_size, device, keep_mask=None):
        valid = torch.ones(batch_size, dtype=torch.bool, device=device)
        if keep_mask is not None:
            valid = valid & keep_mask.to(device=device, dtype=torch.bool)
        return valid

    def _future_step_from_goal_time(self, future_motion_pred,
                                    goal_time_frame=None):
        if goal_time_frame is None:
            return torch.full(
                (future_motion_pred.shape[0],),
                future_motion_pred.shape[1] - 1,
                dtype=torch.long,
                device=future_motion_pred.device,
            )
        goal_step = goal_time_frame.to(
            device=future_motion_pred.device, dtype=torch.long
        )
        if goal_step.ndim > 1:
            goal_step = goal_step.squeeze(-1)
        return (goal_step - 1).clamp(
            min=0, max=future_motion_pred.shape[1] - 1)

    def _future_goal_state(self, future_motion_pred, history_motion=None,
                           goal_time_frame=None, include_yaw=False):
        if motion_dtype.FeatureVersion == 6:
            return self._future_goal_state_v6(
                future_motion_pred, history_motion,
                goal_time_frame=goal_time_frame)
        future_motion = self.dataset.denormalize(future_motion_pred)
        goal_step = self._future_step_from_goal_time(
            future_motion, goal_time_frame)
        batch_idx = torch.arange(
            future_motion.shape[0], device=future_motion.device)
        state = {
            'future_motion': future_motion,
            'goal_step': goal_step,
            'batch_idx': batch_idx,
            'selected': future_motion[batch_idx, goal_step],
        }
        if include_yaw:
            if history_motion is None:
                raise ValueError(
                    "history_motion is required to integrate goal-frame yaw")
            history_last = self.dataset.denormalize(history_motion[:, -1:])
            delta_yaw = torch.cat(
                (history_last[..., 4], future_motion[..., 4]), dim=1)
            yaw_future = delta_yaw[:, :future_motion.shape[1]].cumsum(dim=1)
            state['yaw_at_goal'] = yaw_future[batch_idx, goal_step]
        return state

    def _future_feature_at_goal(self, future_motion_pred,
                                goal_time_frame=None,
                                goal_state=None):
        if goal_state is not None:
            return goal_state['selected']
        future_motion = self.dataset.denormalize(future_motion_pred)
        goal_step = self._future_step_from_goal_time(
            future_motion, goal_time_frame)
        batch_idx = torch.arange(
            future_motion.shape[0], device=future_motion.device)
        return future_motion[batch_idx, goal_step]

    def _future_yaw_ego_at_goal(self, future_motion_pred, history_motion,
                                goal_time_frame=None,
                                goal_state=None):
        if goal_state is not None and 'yaw_at_goal' in goal_state:
            return goal_state['yaw_at_goal']
        if history_motion is None:
            raise ValueError(
                "history_motion is required to integrate goal-frame yaw")
        future_motion = self.dataset.denormalize(future_motion_pred)
        history_last = self.dataset.denormalize(history_motion[:, -1:])
        delta_yaw = torch.cat(
            (history_last[..., 4], future_motion[..., 4]), dim=1)
        yaw_future = delta_yaw[:, :future_motion.shape[1]].cumsum(dim=1)
        goal_step = self._future_step_from_goal_time(
            future_motion, goal_time_frame)
        batch_idx = torch.arange(
            future_motion.shape[0], device=future_motion.device)
        return yaw_future[batch_idx, goal_step]

    def calc_goal_root_orientation_loss(
        self,
        future_motion_pred,
        ego_goal,
        goal_orientation_condition_keep_mask=None,
        history_motion=None,
        goal_time_frame=None,
        goal_state=None,
    ):
        """Match TextOp-style root orientation at the selected goal frame."""
        if (motion_dtype.FeatureVersion == 6
                and ego_goal.shape[-1]
                in (ROT_MAT_JOINT_STATE_GOAL_DIM, SPLIT_GOAL_DIM)):
            if goal_state is None:
                goal_state = self._future_goal_state_v6(
                    future_motion_pred, history_motion,
                    goal_time_frame=goal_time_frame)
            predicted = goal_state['rel_rot_at_goal']
            if ego_goal.shape[-1] == SPLIT_GOAL_DIM:
                target_rot6d = ego_goal[..., SPLIT_ORIENTATION_SLICE]
            else:
                target = ego_goal[..., V6_RAW_ORIENTATION_SLICE]
                target_rot6d = target[..., 3:9]
            target_R = rot6d_to_matrix(target_rot6d)
            valid = self._valid_goal_component_mask(
                target_rot6d.shape[0], target_rot6d.device,
                goal_orientation_condition_keep_mask)
            if not valid.any():
                return future_motion_pred.sum() * 0.0
            return (predicted[valid] - target_R[valid]).square().sum(
                dim=(-1, -2)).mean()

        selected = self._future_feature_at_goal(
            future_motion_pred, goal_time_frame, goal_state=goal_state)
        yaw = self._future_yaw_ego_at_goal(
            future_motion_pred, history_motion, goal_time_frame,
            goal_state=goal_state)
        predicted = torch.cat((selected[..., 0:4], yaw.unsqueeze(-1)), dim=-1)
        target = ego_goal[..., 3:8]
        error = predicted - target
        error[..., 4] = torch.atan2(
            torch.sin(error[..., 4]), torch.cos(error[..., 4]))
        valid = self._valid_goal_component_mask(
            target.shape[0], target.device,
            goal_orientation_condition_keep_mask)
        if not valid.any():
            return future_motion_pred.sum() * 0.0
        return self.rec_criterion(error[valid], torch.zeros_like(error[valid]))

    def calc_goal_joint_angle_loss(
        self,
        future_motion_pred,
        ego_goal,
        goal_joint_condition_keep_mask=None,
        goal_time_frame=None,
        goal_state=None,
    ):
        """Match the 29-DOF joint angles at the selected goal frame."""
        dof_dim = int(self.dataset.dof_dim)
        if dof_dim != 29:
            raise ValueError(
                "joint_state goal losses require dataset.dof_dim=29, got "
                f"{dof_dim}"
            )
        if (motion_dtype.FeatureVersion == 6
                and ego_goal.shape[-1]
                in (ROT_MAT_JOINT_STATE_GOAL_DIM, SPLIT_GOAL_DIM)):
            selected = self._future_feature_at_goal(
                future_motion_pred, goal_time_frame, goal_state=goal_state)
            predicted = selected[..., 13:42]
            target = (
                ego_goal[..., SPLIT_JOINT_SLICE]
                if ego_goal.shape[-1] == SPLIT_GOAL_DIM
                else ego_goal[..., V6_RAW_JOINT_SLICE]
            )
            valid = self._valid_goal_component_mask(
                target.shape[0], target.device, goal_joint_condition_keep_mask)
            if not valid.any():
                return future_motion_pred.sum() * 0.0
            return self.rec_criterion(predicted[valid], target[valid])

        selected = self._future_feature_at_goal(
            future_motion_pred, goal_time_frame, goal_state=goal_state)
        predicted = selected[..., 11:40]
        target = ego_goal[..., 8:37]
        valid = self._valid_goal_component_mask(
            target.shape[0], target.device, goal_joint_condition_keep_mask)
        if not valid.any():
            return future_motion_pred.sum() * 0.0
        return self.rec_criterion(predicted[valid], target[valid])

    def calc_goal_root_velocity_loss(
        self,
        future_motion_pred,
        ego_goal,
        goal_velocity_condition_keep_mask=None,
        history_motion=None,
        goal_time_frame=None,
        goal_state=None,
    ):
        """Match reference-ego root velocity at the selected goal frame."""
        if (motion_dtype.FeatureVersion == 6
                and ego_goal.shape[-1]
                in (ROT_MAT_JOINT_STATE_GOAL_DIM, SPLIT_GOAL_DIM)):
            if goal_state is None:
                goal_state = self._future_goal_state_v6(
                    future_motion_pred, history_motion,
                    goal_time_frame=goal_time_frame)
            predicted = goal_state['velocity_at_goal']
            target = (
                ego_goal[..., SPLIT_VELOCITY_SLICE]
                if ego_goal.shape[-1] == SPLIT_GOAL_DIM
                else ego_goal[..., V6_RAW_VELOCITY_SLICE]
            )
            valid = self._valid_goal_component_mask(
                target.shape[0], target.device,
                goal_velocity_condition_keep_mask)
            if not valid.any():
                return future_motion_pred.sum() * 0.0
            return self.rec_criterion(predicted[valid], target[valid])

        selected = self._future_feature_at_goal(
            future_motion_pred, goal_time_frame, goal_state=goal_state)
        yaw = self._future_yaw_ego_at_goal(
            future_motion_pred, history_motion, goal_time_frame,
            goal_state=goal_state)
        delta = selected[..., 7:10] * float(self.dataset.fps)
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        predicted = torch.stack(
            (
                delta[..., 0] * cos_yaw - delta[..., 1] * sin_yaw,
                delta[..., 0] * sin_yaw + delta[..., 1] * cos_yaw,
                delta[..., 2],
            ),
            dim=-1,
        )
        target = ego_goal[..., 37:40]
        valid = self._valid_goal_component_mask(
            target.shape[0], target.device, goal_velocity_condition_keep_mask)
        if not valid.any():
            return future_motion_pred.sum() * 0.0
        return self.rec_criterion(predicted[valid], target[valid])

    def _trajectory_at_step(self, trajectory, goal_time_frame=None):
        if goal_time_frame is None:
            return trajectory[:, -1]
        goal_time_frame = goal_time_frame.to(
            device=trajectory.device, dtype=torch.long
        )
        if goal_time_frame.ndim > 1:
            goal_time_frame = goal_time_frame.squeeze(-1)
        goal_time_frame = goal_time_frame.clamp(
            min=0, max=trajectory.shape[1] - 1
        )
        batch_idx = torch.arange(trajectory.shape[0], device=trajectory.device)
        return trajectory[batch_idx, goal_time_frame]

    def root_displacement_ego(self, future_motion_pred, history_motion,
                              goal_time_frame=None):
        """Integrate reference-to-goal displacement in the reference ego frame."""
        trajectory = self.root_trajectory_ego(future_motion_pred, history_motion)
        return self._trajectory_at_step(trajectory, goal_time_frame)

    def root_trajectory_ego(self, future_motion_pred, history_motion):
        """Integrate root XY positions in the reference ego frame."""
        if motion_dtype.FeatureVersion == 6:
            return self._future_goal_state_v6(
                future_motion_pred, history_motion)['trajectory']
        if motion_dtype.FeatureVersion != 3:
            raise NotImplementedError(
                "root trajectory integration currently supports FeatureVersion 3 only"
            )
        if history_motion is None:
            raise ValueError(
                "history_motion is required to integrate displacement from "
                "the last history frame"
            )

        future_motion = self.dataset.denormalize(future_motion_pred)
        history_last = self.dataset.denormalize(history_motion[:, -1:])
        # Feature frame t stores the forward delta t -> t+1. The path starts
        # with the last history delta and ends with future[-2] -> future[-1].
        path_motion = torch.cat((history_last, future_motion[:, :-1]), dim=1)
        delta_yaw = path_motion[..., 4]
        relative_yaw = torch.cat(
            (torch.zeros_like(delta_yaw[:, :1]), delta_yaw[:, :-1]), dim=1
        ).cumsum(dim=1)
        cos_yaw = torch.cos(relative_yaw)
        sin_yaw = torch.sin(relative_yaw)
        delta_xy = path_motion[..., 7:9]
        delta_xy_start_frame = torch.stack(
            (
                delta_xy[..., 0] * cos_yaw - delta_xy[..., 1] * sin_yaw,
                delta_xy[..., 0] * sin_yaw + delta_xy[..., 1] * cos_yaw,
            ),
            dim=-1,
        )
        origin = torch.zeros_like(delta_xy_start_frame[:, :1])
        return torch.cat(
            (origin, delta_xy_start_frame.cumsum(dim=1)), dim=1
        )

    def history_trajectory_ego(self, history_motion):
        """Reverse-integrate history root XY in the reference ego frame.

        Traces where the character came from, with the origin at the last
        history frame (the reference frame). Each frame *t* encodes the
        forward delta t→t+1; we integrate forward first then transform all
        positions back to the reference frame.

        Returns:
            Tensor [B, history_len, 2] — ego XY positions of history
            frames.  The last frame (reference) is always (0, 0).
        """
        if motion_dtype.FeatureVersion == 6:
            return self._history_trajectory_ego_v6(history_motion)
        if motion_dtype.FeatureVersion != 3:
            raise NotImplementedError(
                "history trajectory integration currently supports "
                "FeatureVersion 3 or 6 only"
            )
        if history_motion is None:
            raise ValueError(
                "history_motion is required to integrate history trajectory"
            )
        history = self.dataset.denormalize(history_motion)
        B, H, _ = history.shape
        if H < 2:
            return torch.zeros(B, H, 2, device=history.device)

        delta_yaw = history[..., 4]       # [B, H]
        delta_xy = history[..., 7:9]      # [B, H, 2]

        # ── yaw of each frame relative to frame 0 ──
        relative_yaw = torch.cat(
            (torch.zeros_like(delta_yaw[:, :1]), delta_yaw[:, :-1]),
            dim=1,
        ).cumsum(dim=1)  # [B, H]

        cos_yaw = torch.cos(relative_yaw)
        sin_yaw = torch.sin(relative_yaw)

        # ── express every delta in frame-0 coordinates ──
        delta_in_frame0 = torch.stack(
            (
                delta_xy[..., 0] * cos_yaw - delta_xy[..., 1] * sin_yaw,
                delta_xy[..., 0] * sin_yaw + delta_xy[..., 1] * cos_yaw,
            ),
            dim=-1,
        )  # [B, H, 2]

        # ── cumulative positions relative to frame 0 ──
        origin_frame0 = torch.zeros_like(delta_in_frame0[:, :1])
        pos_frame0 = torch.cat(
            (origin_frame0, delta_in_frame0.cumsum(dim=1)), dim=1
        )  # [B, H+1, 2]  — frame 0 … frame H (first future) in frame-0 coords

        # ── transform to reference frame (last history = index H-1) ──
        ref_pos = pos_frame0[:, H - 1 : H]    # [B, 1, 2]
        ref_yaw = relative_yaw[:, H - 1 : H]  # [B, 1]
        d_pos = pos_frame0[:, :H] - ref_pos   # [B, H, 2]

        cos_ref = torch.cos(-ref_yaw)
        sin_ref = torch.sin(-ref_yaw)
        ego_pos = torch.stack(
            (
                d_pos[..., 0] * cos_ref - d_pos[..., 1] * sin_ref,
                d_pos[..., 0] * sin_ref + d_pos[..., 1] * cos_ref,
            ),
            dim=-1,
        )  # [B, H, 2]

        return ego_pos


def calc_mvae_loss(self,
              future_motion_gt,
              future_motion_pred,
              dist,
              history_motion=None,
              sliding_mask=None,
              action_label=None,
              is_recovery=None) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
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
    kl_loss = _standard_normal_kl_mean(dist)
    terms['kl'] = kl_loss

    # 使用继承的几何损失计算方法
    if motion_dtype.FeatureVersion == 4:
        geometry_terms, geometry_extras = self.calc_geometry_loss_v2(
            future_motion_pred, future_motion_gt, history_motion
        )
    elif motion_dtype.FeatureVersion == 5:
        geometry_terms, geometry_extras = self.calc_geometry_loss_v3(
            future_motion_pred, future_motion_gt, history_motion,
            sliding_mask=sliding_mask,
        )
    elif motion_dtype.FeatureVersion == 6:
        geometry_terms, geometry_extras = self.calc_geometry_loss_v6(
            future_motion_pred,
            future_motion_gt,
            history_motion,
            smooth=self.loss_weight.get('smooth', 0.0) > 0.0,
            sliding_mask=sliding_mask,
            action_label=action_label,
            is_recovery=is_recovery,
        )
    else:
        quantize = (self.loss_weight.get('quantize_rot', 0.0) > 0.0 or self.loss_weight.get('quantize_trans', 0.0) > 0.0)
        endpoint = (self.loss_weight.get('endpoint_xy', 0.0) > 0.0 or self.loss_weight.get('endpoint_yaw', 0.0) > 0.0)
        geometry_terms, geometry_extras = self.calc_geometry_loss(
            future_motion_pred,
            future_motion_gt,
            history_motion,
            smooth=self.loss_weight.get('smooth', 0.0) > 0.0,
            quantize=quantize,
            endpoint=endpoint,
            sliding_mask=sliding_mask,
        )

    # geometry_terms = self.calc_geometry_loss_v2(future_motion_pred, future_motion_gt, history_motion)
    terms.update(geometry_terms)
    extras.update(geometry_extras)

    total_loss = sum(self.loss_weight.get(k, 0.0) * v for k, v in terms.items())
    terms['total'] = total_loss
    return terms, extras


def calc_dar_loss(
    self,
    future_motion_gt,
    future_motion_pred,
    latent,
    dist=None,
    latent_pred=None,
    weights=None,
    history_motion=None,
    sliding_mask=None,
    ego_goal=None,
    goal_condition_keep_mask=None,
    goal_type: GoalType | str = GoalType.ROOT,
    goal_orientation_condition_keep_mask=None,
    goal_joint_condition_keep_mask=None,
    goal_velocity_condition_keep_mask=None,
    goal_time_frame=None,
    is_eval: bool = False,
    action_label=None,
    is_recovery=None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    terms = {}
    extras = {}

    # 重构损失
    rec_loss = self.rec_criterion(future_motion_pred, future_motion_gt)
    terms['rec'] = rec_loss

    # KL损失
    if dist is not None:
        kl_loss = _standard_normal_kl_mean(dist)
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
    if motion_dtype.FeatureVersion == 4:
        geometry_terms, geometry_extras = self.calc_geometry_loss_v2(
            future_motion_pred, future_motion_gt, history_motion
        )
    elif motion_dtype.FeatureVersion == 5:
        geometry_terms, geometry_extras = self.calc_geometry_loss_v3(
            future_motion_pred, future_motion_gt, history_motion,
            sliding_mask=sliding_mask,
        )
    elif motion_dtype.FeatureVersion == 6:
        geometry_terms, geometry_extras = self.calc_geometry_loss_v6(
            future_motion_pred,
            future_motion_gt,
            history_motion,
            smooth=self.loss_weight.get('smooth', 0.0) > 0.0,
            sliding_mask=sliding_mask,
            action_label=action_label,
            is_recovery=is_recovery,
        )
    else:
        quantize = (self.loss_weight.get('quantize_rot', 0.0) > 0.0 or self.loss_weight.get('quantize_trans', 0.0) > 0.0)
        endpoint = (self.loss_weight.get('endpoint_xy', 0.0) > 0.0 or self.loss_weight.get('endpoint_yaw', 0.0) > 0.0)
        geometry_terms, geometry_extras = self.calc_geometry_loss(
            future_motion_pred,
            future_motion_gt,
            history_motion,
            smooth=self.loss_weight.get('smooth', 0.0) > 0.0,
            quantize=quantize,
            endpoint=endpoint,
            sliding_mask=sliding_mask,
        )

    # geometry_terms, geometry_extras = calc_geometry_loss(
    #     future_motion_pred, future_motion_gt, history_motion)
    # geometry_terms = self.calc_geometry_loss(future_motion_pred,
    #                                          future_motion_gt, history_motion=history_motion)

    # geometry_terms = self.calc_geometry_loss_v2(future_motion_pred, future_motion_gt, history_motion)
    terms.update(geometry_terms)
    extras.update(geometry_extras)

    goal_type = GoalType.parse(goal_type)
    compute_goal_root_position = (
        self.loss_weight.get('goal_root_position', 0.0) > 0.0 or is_eval
    )
    if compute_goal_root_position:
        if ego_goal is None:
            raise ValueError(
                "ego_goal is required when goal_root_position loss is enabled "
                "or during eval"
            )
        terms['goal_root_position'] = self.calc_goal_root_position_loss(
            future_motion_pred, ego_goal, goal_condition_keep_mask,
            history_motion=history_motion,
            goal_time_frame=goal_time_frame,
        )

    joint_goal_dims = (
        JOINT_STATE_GOAL_DIM,
        ROT_MAT_JOINT_STATE_GOAL_DIM,
        SPLIT_GOAL_DIM,
    )
    if ego_goal is not None and ego_goal.shape[-1] in joint_goal_dims:
        compute_goal_root_orientation = (
            self.loss_weight.get('goal_root_orientation', 0.0) > 0.0
            or is_eval
        )
        compute_goal_joint_angle = (
            self.loss_weight.get('goal_joint_angle', 0.0) > 0.0
            or is_eval
        )
        compute_goal_root_velocity = (
            self.loss_weight.get('goal_root_velocity', 0.0) > 0.0
            or is_eval
        )
        goal_state = None
        if (compute_goal_root_orientation or compute_goal_joint_angle
                or compute_goal_root_velocity):
            goal_state = self._future_goal_state(
                future_motion_pred,
                history_motion=history_motion,
                goal_time_frame=goal_time_frame,
                include_yaw=(
                    compute_goal_root_orientation
                    or compute_goal_root_velocity
                ),
            )
        if (self.loss_weight.get('goal_root_orientation', 0.0) > 0.0
                or is_eval):
            terms['goal_root_orientation'] = (
                self.calc_goal_root_orientation_loss(
                    future_motion_pred,
                    ego_goal,
                    goal_orientation_condition_keep_mask,
                    history_motion=history_motion,
                    goal_time_frame=goal_time_frame,
                    goal_state=goal_state,
                )
            )
        if (self.loss_weight.get('goal_joint_angle', 0.0) > 0.0
                or is_eval):
            terms['goal_joint_angle'] = (
                self.calc_goal_joint_angle_loss(
                    future_motion_pred,
                    ego_goal,
                    goal_joint_condition_keep_mask,
                    goal_time_frame=goal_time_frame,
                    goal_state=goal_state,
                )
            )
        if (self.loss_weight.get('goal_root_velocity', 0.0) > 0.0
                or is_eval):
            terms['goal_root_velocity'] = (
                self.calc_goal_root_velocity_loss(
                    future_motion_pred,
                    ego_goal,
                    goal_velocity_condition_keep_mask,
                    history_motion=history_motion,
                    goal_time_frame=goal_time_frame,
                    goal_state=goal_state,
                )
            )

    total_loss = sum(
        self.loss_weight.get(k, 0.0) * v for k, v in terms.items()
    )

    # diffusion训练时可加权
    if weights is not None:
        total_loss = total_loss * weights.mean()

    terms['total'] = total_loss
    return terms, extras
