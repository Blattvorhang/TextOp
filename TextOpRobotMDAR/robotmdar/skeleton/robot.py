from functools import cached_property
import torch
from typing import Optional
from omegaconf import DictConfig
from isaac_utils.rotations import axis_angle_to_quaternion, quaternion_to_matrix
from robotmdar.dtype.motion import MotionDict
from robotmdar.dtype.rotation import xyzw_to_wxyz
from robotmdar.skeleton.forward_kinematics import ForwardKinematics
from scipy.spatial.transform import Rotation as sRot


class RobotSkeleton:

    def __init__(self, device: str = "cpu", cfg: Optional[DictConfig] = None):
        self.fk = ForwardKinematics(
            cfg=cfg,
            device=torch.device(device),
        )
        self.device = device

    @property
    def num_bodies(self):
        return self.fk.num_bodies

    @property
    def body_names(self):
        return self.fk.body_names

    @cached_property
    def foot_id(self):
        return self.fk.get_foot_id()

    @cached_property
    def hand_id(self):
        return self.fk.get_hand_id()

    @cached_property
    def goal_keypoint_id(self):
        return [0, *self.foot_id, *self.hand_id]

    @property
    def parent_indices(self):
        return self.fk.parent_indices

    @property
    def local_translations(self):
        return self.fk.local_translations

    @property
    def local_rotations(self):
        return self.fk.local_rotations

    @property
    def num_extend_dof(self):
        return self.fk.num_extend_dof

    def forward_kinematics(self,
                           motion_dict: MotionDict,
                           return_full: bool = False) -> dict:
        """
        输入: motion_dict (root_trans_offset, root_rot, dof, contact_mask)
        输出: FK后的全局位姿信息（dict，含global_translation, global_rotation等）
        支持有batch和无batch的输入
        """
        dof = motion_dict['dof']
        root_trans = motion_dict['root_trans_offset']
        root_rot = motion_dict['root_rot']

        # # 判断是否有batch维度
        if dof.ndim == 3:
            # (batch, seq, joints)
            dof_batch = dof
            joint_angles = self.fk.dof_to_axis_angle(dof_batch)
            root_translation = root_trans
        elif dof.ndim == 2:
            # (seq, joints)
            dof_batch = dof.unsqueeze(0)
            joint_angles = self.fk.dof_to_axis_angle(dof_batch)
            root_translation = root_trans.unsqueeze(0)
            root_rot = root_rot.unsqueeze(0)
        else:
            raise ValueError(f"dof shape not supported: {dof.shape}")

        # fk_batch's axis-angle path treats every 3-vector as a rotation
        # vector. A root Euler triple is not a rotation vector and corrupts
        # combined roll/pitch/yaw. Construct matrices explicitly so the root
        # quaternion is preserved exactly.
        root_rot_mat = quaternion_to_matrix(
            xyzw_to_wxyz(root_rot)).unsqueeze(-3)
        joint_rot_mat = quaternion_to_matrix(
            axis_angle_to_quaternion(joint_angles))
        num_extended = self.num_extend_dof - joint_angles.shape[-2]
        extended_rot_mat = torch.eye(
            3, dtype=joint_angles.dtype, device=joint_angles.device
        ).reshape(1, 1, 1, 3, 3).expand(
            *joint_angles.shape[:-2], num_extended, 3, 3)
        pose_mat = torch.cat(
            (root_rot_mat, joint_rot_mat, extended_rot_mat), dim=-3)

        fk_result = self.fk.fk_batch(pose_mat,
                                     root_translation,
                                     convert_to_mat=False,
                                     return_full=return_full)
        # fk_batch normally recovers scalar joints by summing axis-angle
        # vectors. Its matrix-input path cannot do that, so retain the exact
        # source angles instead.
        fk_result.dof_pos = dof_batch
        fk_result.update(motion_dict)
        return fk_result
