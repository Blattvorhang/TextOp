from functools import cached_property
import torch
from typing import Optional
from omegaconf import DictConfig
from robotmdar.dtype.motion import MotionDict, quaternion_to_euler_angles
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
        root_rot_aa = quaternion_to_euler_angles(root_rot)

        # # 判断是否有batch维度
        if dof.ndim == 3:
            # (batch, seq, joints)
            joint_angles = self.fk.dof_to_axis_angle(dof)
            root_translation = root_trans
        elif dof.ndim == 2:
            # (seq, joints)
            joint_angles = self.fk.dof_to_axis_angle(dof.unsqueeze(0))
            root_translation = root_trans.unsqueeze(0)
            root_rot_aa = root_rot_aa.unsqueeze(0)
        else:
            raise ValueError(f"dof shape not supported: {dof.shape}")

        pose_aa = torch.cat(
            (root_rot_aa.unsqueeze(-2), joint_angles,
             torch.zeros((*joint_angles.shape[:-2],
                          self.num_extend_dof - joint_angles.shape[-2], 3),
                         device=joint_angles.device)),
            dim=-2)

        fk_result = self.fk.fk_batch(pose_aa,
                                     root_translation,
                                     return_full=return_full)
        fk_result.update(motion_dict)
        return fk_result
