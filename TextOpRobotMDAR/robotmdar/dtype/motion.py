from pathlib import Path
import numpy as np
from typing import Any, Callable, Dict, Optional, List
from typing import TypedDict, Union, Tuple
import robotmdar
from robotmdar.dtype.rotation import (
    euler_angles_to_quaternion, quaternion_to_euler_angles, quat_inverse, quaternion_to_matrix, matrix_to_quaternion,
    xyzw_to_wxyz, wxyz_to_xyzw, matrix_to_rot6d, rot6d_to_matrix, quat_apply
)
import torch
import joblib

# Define the dimensions of each component (change if needed)
ROOT_TRANS_OFFSET_DIM = 3  # xyz (m)
ROOT_ROT_DIM = 4  # Quat, xyzw
DOF_DIM = 29  # Full G1 skeleton, including both 3-DoF wrists.
CONTACT_MASK_DIM = 2  # 2 feet

G1_ROOT_HEIGHT = 0.77  # meters

G1_DEFAULT_DOF = (
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,       # left leg
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,       # right leg
    0.0, 0.0, 0.0,                          # waist
    0.2, 0.2, 0.0, 0.9,                    # left arm
    0.0, 0.0, 0.0,                          # left wrist
    0.2, -0.2, 0.0, 0.9,                   # right arm
    0.0, 0.0, 0.0,                          # right wrist
)
assert len(G1_DEFAULT_DOF) == DOF_DIM


# Define MotionDict with explicit component shapes

MotionKeys: List[str] = ['root_trans_offset', 'dof', 'root_rot', 'contact_mask']


class MotionDict_(TypedDict):
    root_trans_offset: torch.Tensor  # shape (..., 3)
    root_rot: torch.Tensor  # shape (..., 4) xyzw
    dof: torch.Tensor  # shape (..., 29)
    contact_mask: torch.Tensor  # shape (..., 2)


MotionDict = Union[MotionDict_, Dict[str, torch.Tensor]]


class AbsolutePose_(TypedDict):
    root_trans_offset: torch.Tensor  # shape (..., 3)
    root_rot: torch.Tensor  # shape (..., 4)


AbsolutePose = Union[AbsolutePose_, Dict[str, torch.Tensor]]

FeatureVersion: int = 3

QPos = torch.Tensor  # shape (..., 7 + DOF_DIM), MuJoCo qpos format
n_qpos = 7 + DOF_DIM

# MotionFeatureV0 is a tensor with last dim = 38 (3+4+29+2).
MotionFeatureV0 = torch.Tensor

MotionFeatureV1 = torch.Tensor
# (root_roll, root_pitch) x (sin, cos) : 4
# delta yaw: 1
# contact_mask: 2
# root invariant - delta root trans: 3
# height: 1
# dof: 29

MotionFeatureV2 = torch.Tensor
# (root_roll, root_pitch) x (sin, cos) : 4
# delta yaw (t+1 - t): 1
# contact_mask: 2
# root invariant - delta root trans (t+1 - t): 3
# height: 1
# dof: 29
# delta dof (t+1 - t): 29

MotionFeatureV3 = torch.Tensor


def motion_dict_to_abs_pose(motion_dict: MotionDict, idx: int = -1) -> AbsolutePose:
    " Pick the Last frame's absolute pose from the feature"
    batch_shape = motion_dict['root_trans_offset'].shape[:-1]
    if len(batch_shape) == 0:
        # (dof)
        ret = {
            'root_trans_offset': motion_dict['root_trans_offset'],
            'root_rot': motion_dict['root_rot'],
        }
    elif len(batch_shape) == 1:
        # (seq_len, dof)
        ret = {
            'root_trans_offset': motion_dict['root_trans_offset'][idx],
            'root_rot': motion_dict['root_rot'][idx],
        }
    elif len(batch_shape) == 2:
        # (batch_size, seq_len, dof)
        ret = {
            'root_trans_offset': motion_dict['root_trans_offset'][:, idx],
            'root_rot': motion_dict['root_rot'][:, idx],
        }
    else:
        raise ValueError(f"Unexpected batch shape {batch_shape}")
    return ret


def motion_dict_to_qpos(motion_dict: MotionDict) -> Tuple[QPos, torch.Tensor]:
    batch_shape = motion_dict['root_trans_offset'].shape[:-1]
    qpos = torch.zeros(batch_shape + (n_qpos, ), device=motion_dict['root_trans_offset'].device)
    qpos[..., :3] = motion_dict['root_trans_offset']
    qpos[..., 3:7] = motion_dict['root_rot']
    qpos[..., 7:] = motion_dict['dof']
    contact = motion_dict['contact_mask']
    return qpos, contact


def motion_dict_to_feature_v0(motion_dict: MotionDict) -> MotionFeatureV0:

    components = [
        motion_dict['root_trans_offset'], motion_dict['root_rot'], motion_dict['dof'], motion_dict['contact_mask']
    ]

    data_concat = torch.cat(components, axis=-1)  # type:ignore
    return data_concat


def motion_feature_to_dict_v0(motion_feature: MotionFeatureV0) -> MotionDict:
    """
    Splits a flat motion_feature array into its dictionary components.
    """
    dim0 = ROOT_TRANS_OFFSET_DIM
    dim1 = dim0 + ROOT_ROT_DIM
    dim2 = dim1 + DOF_DIM
    dim3 = dim2 + CONTACT_MASK_DIM

    assert motion_feature.shape[-1] == dim3, \
        f"Unexpected last dim {motion_feature.shape[-1]}, expected {dim3}"

    return {
        'root_trans_offset': motion_feature[..., :dim0],
        'root_rot': motion_feature[..., dim0:dim1],
        'dof': motion_feature[..., dim1:dim2],
        'contact_mask': motion_feature[..., dim2:dim3],
    }


def motion_dict_to_feature_v1(motion_dict: MotionDict) -> Tuple[MotionFeatureV1, AbsolutePose]:
    # 保持不变
    trans = motion_dict['root_trans_offset']  # (T, 3)
    rot = motion_dict['root_rot']  # (T, 4)
    dof = motion_dict['dof']  # (T, 29)
    contact = motion_dict['contact_mask']  # (T, 2)

    T, _ = trans.shape

    abs_pose = {'root_trans_offset': trans[0], 'root_rot': rot[0]}

    height = trans[:, 2]

    euler = quaternion_to_euler_angles(rot)
    roll = euler[..., 0]
    pitch = euler[..., 1]
    yaw = euler[..., 2]

    sincos = torch.stack([torch.sin(roll), torch.cos(roll), torch.sin(pitch), torch.cos(pitch)], dim=-1)

    # 关键修改：使用第一帧的绝对Yaw作为参考
    ref_yaw = yaw[0]  # 标量

    # 计算相对于第一帧的Yaw变化
    # delta_yaw = (yaw - ref_yaw).unsqueeze(-1)  # (T, 1)
    yaw_prev = torch.cat([yaw[:1], yaw[:-1]], dim=0)  # 前一帧的Yaw
    delta_yaw = (yaw - yaw_prev).unsqueeze(-1)  # (T,1)

    # 计算位置变化（相对于前一帧）
    trans_prev = torch.cat([trans[:1], trans[:-1]], dim=0)
    delta_trans_world = trans - trans_prev

    # 使用第一帧的Yaw构造旋转矩阵
    ref_yaw_quat = euler_angles_to_quaternion(torch.tensor([0, 0, ref_yaw], device=trans.device).unsqueeze(0))  # (1, 4)
    inv_ref_yaw_quat = quat_inverse(ref_yaw_quat, True)

    # 将位移转换到以第一帧Yaw为参考的局部坐标系
    delta_trans_local = quat_apply(inv_ref_yaw_quat, delta_trans_world, True)  # (T, 3)

    feature = torch.cat([sincos, delta_yaw, contact, delta_trans_local, height.unsqueeze(1), dof], dim=-1)
    return feature, AbsolutePose(**abs_pose)


def motion_feature_to_dict_v1(motion_feature: MotionFeatureV1, abs_pose: Optional[AbsolutePose] = None) -> MotionDict:
    T = motion_feature.shape[0]
    if not abs_pose:
        abs_pose = {
            'root_trans_offset': torch.zeros(3, device=motion_feature.device),
            'root_rot': torch.zeros(4, device=motion_feature.device)
        }

    # 提取特征组件
    sin_roll = motion_feature[:, 0]
    cos_roll = motion_feature[:, 1]
    sin_pitch = motion_feature[:, 2]
    cos_pitch = motion_feature[:, 3]
    delta_yaw = motion_feature[:, 4]  # 相对于第一帧的Yaw变化
    contact = motion_feature[:, 5:7]
    delta_trans_local = motion_feature[:, 7:10]
    height = motion_feature[:, 10]  # 高度
    dof = motion_feature[:, 11:]  # (T, 29)

    # 计算roll和pitch
    roll = torch.atan2(sin_roll, cos_roll)
    pitch = torch.atan2(sin_pitch, cos_pitch)

    # 从绝对位姿获取第一帧的绝对Yaw
    init_euler = quaternion_to_euler_angles(abs_pose['root_rot'].unsqueeze(0))
    ref_yaw = init_euler[0, 2]  # 标量

    # 重建Yaw序列（相对于第一帧的变化）
    yaw = torch.cumsum(delta_yaw, dim=0) + ref_yaw  # (T,)

    # 组合欧拉角并转换为四元数
    euler = torch.stack([roll, pitch, yaw], dim=-1)
    rot = euler_angles_to_quaternion(euler)

    # 重建位置
    # 使用第一帧的Yaw构造旋转矩阵
    ref_yaw_quat = euler_angles_to_quaternion(
        torch.tensor([0, 0, ref_yaw], device=motion_feature.device).unsqueeze(0)
    )  # (1, 4)

    # 将局部位移转换回世界坐标系
    delta_trans_world = quat_apply(ref_yaw_quat, delta_trans_local, True)  # (T, 3)

    # 重建位置
    trans = torch.zeros((T, 3), device=motion_feature.device)
    trans[0] = abs_pose['root_trans_offset']
    if T > 1:
        trans[1:] = torch.cumsum(delta_trans_world[:T - 1], dim=0) + abs_pose['root_trans_offset']
    trans[:, 2] = height

    return {'root_trans_offset': trans, 'root_rot': rot, 'dof': dof, 'contact_mask': contact}


@torch.jit.script
def __jitable_motion_dict_to_feature_v2__(
    trans: torch.Tensor, rot: torch.Tensor, dof: torch.Tensor, contact: torch.Tensor
) -> MotionFeatureV2:
    B, T_plus_1, _ = trans.shape
    T = T_plus_1 - 1  # 输出T帧
    # 只使用前T帧的高度
    height = trans[:, :T, 2]  #- 0.8

    # 计算前T帧的欧拉角
    euler = quaternion_to_euler_angles(rot[:, :T + 1])
    roll = euler[..., 0]
    pitch = euler[..., 1]
    yaw = euler[..., 2]

    sincos = torch.stack([torch.sin(roll), torch.cos(roll) - 1, torch.sin(pitch), torch.cos(pitch) - 1], dim=-1)[:, :T]

    # 计算delta yaw: (t+1) - t，使用前T帧
    # 对于第t帧，我们需要计算(t+1) - t的delta
    delta_yaw = (yaw[:, 1:T + 1] - yaw[:, :T]).unsqueeze(-1)  # (B, T, 1)

    # 计算位置变化（t+1） - t，使用前T帧
    delta_trans_world = trans[:, 1:T + 1] - trans[:, :T]  # (B, T, 3)

    # 使用第一帧的Yaw构造旋转矩阵
    ref_yaw = yaw[:, 0]  # (B,)
    ref_yaw_quat = euler_angles_to_quaternion(
        torch.stack([torch.zeros_like(ref_yaw), torch.zeros_like(ref_yaw), ref_yaw], dim=-1)
    )  # (B, 4)
    inv_ref_yaw_quat = quat_inverse(ref_yaw_quat, True)

    # 将位移转换到以第一帧Yaw为参考的局部坐标系
    delta_trans_local = quat_apply(
        inv_ref_yaw_quat.reshape(B, 1, 4).expand(B, T, 4), delta_trans_world, True
    )  # (B, T, 3)

    # 计算delta dof: (t+1) - t，使用前T帧
    delta_dof = dof[:, 1:T + 1] - dof[:, :T]  # (B, T, 29)

    # 只使用前T帧的contact
    contact_T = contact[:, :T]

    # 只使用前T帧的dof
    dof_T = dof[:, :T]

    # 确保所有tensor的维度都是(T, ...)
    assert sincos.shape[1] == T, f"sincos shape: {sincos.shape}"
    assert delta_yaw.shape[1] == T, f"delta_yaw shape: {delta_yaw.shape}"
    assert contact_T.shape[1] == T, f"contact_T shape: {contact_T.shape}"
    assert delta_trans_local.shape[1] == T, f"delta_trans_local shape: {delta_trans_local.shape}"
    assert height.shape[1] == T, f"height shape: {height.shape}"
    assert dof_T.shape[1] == T, f"dof_T shape: {dof_T.shape}"
    assert delta_dof.shape[1] == T, f"delta_dof shape: {delta_dof.shape}"

    feature = torch.cat(
        [sincos, delta_yaw, contact_T, delta_trans_local,
         height.unsqueeze(-1), dof_T, delta_dof], dim=-1
    )

    return feature


def motion_dict_to_feature_v2(motion_dict: MotionDict) -> Tuple[MotionFeatureV2, AbsolutePose]:
    # 输入N+1帧，返回N帧的feature
    trans = motion_dict['root_trans_offset']  # ([B,] T+1, 3)
    rot = motion_dict['root_rot']  # ([B], T+1, 4)
    dof = motion_dict['dof']  # ([B], T+1, 29)
    contact = motion_dict['contact_mask']  # ([B], T+1, 2)
    if len(trans.shape) == 2:
        expanded = True
        trans = trans.unsqueeze(0)
        rot = rot.unsqueeze(0)
        dof = dof.unsqueeze(0)
        contact = contact.unsqueeze(0)
    else:
        expanded = False
    B, T_plus_1, _ = trans.shape
    T = T_plus_1 - 1  # 输出T帧

    abs_pose = {'root_trans_offset': trans[:, 0], 'root_rot': rot[:, 0]}

    feature = __jitable_motion_dict_to_feature_v2__(trans, rot, dof, contact)

    if expanded:
        feature = feature[0]
        abs_pose = {'root_trans_offset': abs_pose['root_trans_offset'][0], 'root_rot': abs_pose['root_rot'][0]}
    return feature, abs_pose


def motion_feature_to_dict_v2(motion_feature: MotionFeatureV2, abs_pose: Optional[AbsolutePose] = None) -> MotionDict:
    if len(motion_feature.shape) == 2:
        expanded = True
        motion_feature = motion_feature.unsqueeze(0)
        if abs_pose:
            abs_pose = {
                'root_trans_offset': abs_pose['root_trans_offset'].unsqueeze(0),
                'root_rot': abs_pose['root_rot'].unsqueeze(0)
            }
    else:
        expanded = False

    B, T = motion_feature.shape[:2]
    if not abs_pose:
        abs_pose = {
            'root_trans_offset': torch.zeros(B, 3, device=motion_feature.device),
            'root_rot': torch.zeros(B, 4, device=motion_feature.device)
        }

    # 提取特征组件
    sin_roll = motion_feature[..., 0]
    cos_roll = motion_feature[..., 1] + 1
    sin_pitch = motion_feature[..., 2]
    cos_pitch = motion_feature[..., 3] + 1
    delta_yaw = motion_feature[..., 4]  # (t+1) - t
    contact = motion_feature[..., 5:7]
    delta_trans_local = motion_feature[..., 7:10]  # (t+1) - t
    height = motion_feature[..., 10]  #+ 0.8  # 高度
    dof = motion_feature[..., 11:11 + DOF_DIM]
    delta_dof = motion_feature[..., 11 + DOF_DIM:11 + 2 * DOF_DIM]

    # 计算roll和pitch
    roll = torch.atan2(sin_roll, cos_roll)
    pitch = torch.atan2(sin_pitch, cos_pitch)

    # 从绝对位姿获取第一帧的绝对Yaw
    init_euler = quaternion_to_euler_angles(abs_pose['root_rot'])
    ref_yaw = init_euler[..., 2]  # (B,)

    # 重建Yaw序列
    yaw = torch.zeros(B, T, device=delta_yaw.device)
    yaw[:, 0] = ref_yaw
    if T > 1:
        yaw[:, 1:] = torch.cumsum(delta_yaw[:, :T - 1], dim=1) + ref_yaw.unsqueeze(1)

    # 组合欧拉角并转换为四元数
    euler = torch.stack([roll, pitch, yaw], dim=-1)
    rot = euler_angles_to_quaternion(euler)

    # 位置重建
    ref_yaw_quat = euler_angles_to_quaternion(
        torch.stack([torch.zeros_like(ref_yaw), torch.zeros_like(ref_yaw), ref_yaw], dim=-1)
    )
    delta_trans_world = quat_apply(ref_yaw_quat.unsqueeze(1).expand(B, T, 4), delta_trans_local, True)
    trans = torch.zeros((B, T, 3), device=motion_feature.device)
    trans[:, 0] = abs_pose['root_trans_offset']
    if T > 1:
        trans[:, 1:] = torch.cumsum(delta_trans_world[:, :T - 1], dim=1) + abs_pose['root_trans_offset'].unsqueeze(1)
    trans[:, :, 2] = height

    # dof直接读取
    dof_out = dof
    if expanded:
        trans = trans[0]
        rot = rot[0]
        dof_out = dof_out[0]
        contact = contact[0]
        abs_pose = {'root_trans_offset': abs_pose['root_trans_offset'][0], 'root_rot': abs_pose['root_rot'][0]}
    return {'root_trans_offset': trans, 'root_rot': rot, 'dof': dof_out, 'contact_mask': contact}


# @torch.jit.script
def __jitable_motion_dict_to_feature_v3__(
    trans: torch.Tensor, rot: torch.Tensor, dof: torch.Tensor, contact: torch.Tensor
) -> MotionFeatureV3:
    B, T_plus_1, _ = trans.shape
    T = T_plus_1 - 1  # 输出T帧
    # 只使用前T帧的高度
    height = trans[:, :T, 2]  #- 0.8

    # 计算前T帧的欧拉角
    euler = quaternion_to_euler_angles(rot[:, :T + 1])
    roll = euler[..., 0]
    pitch = euler[..., 1]
    yaw = euler[..., 2]
    # print(f"DEBUG: {yaw=}")

    sincos = torch.stack([torch.sin(roll), torch.cos(roll) - 1, torch.sin(pitch), torch.cos(pitch) - 1], dim=-1)[:, :T]

    # 计算delta yaw: (t+1) - t，使用前T帧
    # 对于第t帧，我们需要计算(t+1) - t的delta
    delta_yaw = (yaw[:, 1:T + 1] - yaw[:, :T]).unsqueeze(-1)  # (B, T, 1)

    # 计算位置变化（t+1） - t，使用前T帧
    delta_trans_world = trans[:, 1:T + 1] - trans[:, :T]  # (B, T, 3)
    # print(f"DEBUG: {delta_trans_world=}")

    ref_yaw_quat = euler_angles_to_quaternion(
        torch.stack([torch.zeros_like(yaw), torch.zeros_like(yaw), yaw], dim=-1)
    )[:, :T]  # (B, T, 4)
    inv_ref_yaw_quat = quat_inverse(ref_yaw_quat, True)
    # print(f"DEBUG: {inv_ref_yaw_quat=}")

    # 将位移转换到以第一帧Yaw为参考的局部坐标系
    delta_trans_local = quat_apply(inv_ref_yaw_quat, delta_trans_world, True)  # (B, T, 3)
    # print(f"DEBUG: {delta_trans_local=}")

    # 计算delta dof: (t+1) - t，使用前T帧
    delta_dof = dof[:, 1:T + 1] - dof[:, :T]  # (B, T, 29)

    # 只使用前T帧的contact
    contact_T = contact[:, :T]

    # 只使用前T帧的dof
    dof_T = dof[:, :T]

    # 确保所有tensor的维度都是(T, ...)
    assert sincos.shape[1] == T, f"sincos shape: {sincos.shape}"
    assert delta_yaw.shape[1] == T, f"delta_yaw shape: {delta_yaw.shape}"
    assert contact_T.shape[1] == T, f"contact_T shape: {contact_T.shape}"
    assert delta_trans_local.shape[1] == T, f"delta_trans_local shape: {delta_trans_local.shape}"
    assert height.shape[1] == T, f"height shape: {height.shape}"
    assert dof_T.shape[1] == T, f"dof_T shape: {dof_T.shape}"
    assert delta_dof.shape[1] == T, f"delta_dof shape: {delta_dof.shape}"

    feature = torch.cat(
        [sincos, delta_yaw, contact_T, delta_trans_local,
         height.unsqueeze(-1), dof_T, delta_dof], dim=-1
    )

    return feature


def motion_dict_to_feature_v3(motion_dict: MotionDict, skeleton: None = None) -> Tuple[MotionFeatureV3, AbsolutePose]:
    # 输入N+1帧，返回N帧的feature
    trans = motion_dict['root_trans_offset']  # ([B,] T+1, 3)
    rot = motion_dict['root_rot']  # ([B], T+1, 4)
    dof = motion_dict['dof']  # ([B], T+1, 29)
    contact = motion_dict['contact_mask']  # ([B], T+1, 2)
    if dof.shape[-1] != DOF_DIM:
        raise ValueError(
            f"FeatureVersion 3 expects {DOF_DIM} DoFs, got {dof.shape[-1]}"
        )
    if len(trans.shape) == 2:
        expanded = True
        trans = trans.unsqueeze(0)
        rot = rot.unsqueeze(0)
        dof = dof.unsqueeze(0)
        contact = contact.unsqueeze(0)
    else:
        expanded = False
    B, T_plus_1, _ = trans.shape
    T = T_plus_1 - 1  # 输出T帧

    abs_pose = {'root_trans_offset': trans[:, 0], 'root_rot': rot[:, 0]}

    feature = __jitable_motion_dict_to_feature_v3__(trans, rot, dof, contact)

    if expanded:
        feature = feature[0]
        abs_pose = {'root_trans_offset': abs_pose['root_trans_offset'][0], 'root_rot': abs_pose['root_rot'][0]}
    return feature, abs_pose


def motion_feature_to_dict_v3(motion_feature: MotionFeatureV3, abs_pose: Optional[AbsolutePose] = None) -> MotionDict:
    if len(motion_feature.shape) == 2:
        expanded = True
        motion_feature = motion_feature.unsqueeze(0)
        if abs_pose:
            abs_pose = {
                'root_trans_offset': abs_pose['root_trans_offset'].unsqueeze(0),
                'root_rot': abs_pose['root_rot'].unsqueeze(0)
            }
    else:
        expanded = False

    B, T = motion_feature.shape[:2]
    if not abs_pose:
        abs_pose = {
            'root_trans_offset': torch.zeros(B, 3, device=motion_feature.device),
            'root_rot': torch.zeros(B, 4, device=motion_feature.device)
        }

    # print(f"DEBUG: {abs_pose=}")
    # 提取特征组件
    sin_roll = motion_feature[..., 0]
    cos_roll = motion_feature[..., 1] + 1
    sin_pitch = motion_feature[..., 2]
    cos_pitch = motion_feature[..., 3] + 1
    delta_yaw = motion_feature[..., 4]  # (t+1) - t
    contact = motion_feature[..., 5:7]
    delta_trans_local = motion_feature[..., 7:10]  # (t+1) - t
    height = motion_feature[..., 10]  #+ 0.8  # 高度
    expected_dim = 11 + 2 * DOF_DIM
    if motion_feature.shape[-1] != expected_dim:
        raise ValueError(
            f"FeatureVersion 3 expects {expected_dim} features for "
            f"{DOF_DIM} DoFs, got {motion_feature.shape[-1]}"
        )
    dof = motion_feature[..., 11:11 + DOF_DIM]
    delta_dof = motion_feature[..., 11 + DOF_DIM:expected_dim]

    # 计算roll和pitch
    roll = torch.atan2(sin_roll, cos_roll)
    pitch = torch.atan2(sin_pitch, cos_pitch)

    # print(f"DEBUG: {delta_trans_local=}")
    # 从绝对位姿获取第一帧的绝对Yaw
    init_euler = quaternion_to_euler_angles(abs_pose['root_rot'])
    # print(f"DEBUG: {init_euler=}")
    ref_yaw = init_euler[..., 2]  # (B,)

    # 重建Yaw序列
    yaw = torch.zeros(B, T, device=delta_yaw.device)
    yaw[:, 0] = ref_yaw
    if T > 1:
        if len(ref_yaw.shape) == 0:
            ref_yaw = ref_yaw.reshape(1)
        yaw[:, 1:] = torch.cumsum(delta_yaw[:, :T - 1], dim=1) + ref_yaw.unsqueeze(1)

    # print(f"DEBUG: {yaw=}")
    # 组合欧拉角并转换为四元数
    euler = torch.stack([roll, pitch, yaw], dim=-1)
    rot = euler_angles_to_quaternion(euler)

    # 位置重建
    ref_yaw_quat = euler_angles_to_quaternion(torch.stack([torch.zeros_like(yaw), torch.zeros_like(yaw), yaw], dim=-1))
    # print(f"DEBUG: {ref_yaw_quat=}")
    delta_trans_world = quat_apply(ref_yaw_quat, delta_trans_local, True)

    # breakpoint()
    # print(f"DEBUG: {delta_trans_world=}")

    trans = torch.zeros((B, T, 3), device=motion_feature.device)
    trans[:, 0] = abs_pose['root_trans_offset']
    if T > 1:
        if len(abs_pose['root_trans_offset'].shape) == 1:
            abs_pose['root_trans_offset'] = abs_pose['root_trans_offset'].reshape(1, 3)
        trans[:, 1:] = torch.cumsum(delta_trans_world[:, :T - 1], dim=1) + abs_pose['root_trans_offset'].unsqueeze(1)
    trans[:, :, 2] = height

    # dof直接读取
    dof_out = dof
    if expanded:
        trans = trans[0]
        rot = rot[0]
        dof_out = dof_out[0]
        contact = contact[0]
        abs_pose = {'root_trans_offset': abs_pose['root_trans_offset'][0], 'root_rot': abs_pose['root_rot'][0]}
    return {'root_trans_offset': trans, 'root_rot': rot, 'dof': dof_out, 'contact_mask': contact}


def get_new_coordinate(jts):
    # right_hip_pitch_link - left_hip_pitch_link
    x_axis = jts[:, 7, :] - jts[:, 1, :]
    x_axis[:, -1] = 0
    x_axis = x_axis / torch.norm(x_axis, dim=-1, keepdim=True)
    z_axis = torch.FloatTensor([[0, 0, 1]]).to(jts.device).repeat(x_axis.shape[0], 1)
    # breakpoint()
    y_axis = torch.cross(z_axis, x_axis, dim=-1)
    y_axis = y_axis / torch.norm(y_axis, dim=-1, keepdim=True)
    new_rotmat = torch.stack([x_axis, y_axis, z_axis], dim=-1)  # [b,3,3]
    new_transl = jts[:, :1]  # [b,1,3]
    return new_rotmat, new_transl


def update_global_transform(old_rotmat, old_transl, new_rotmat, new_transl):
    transf_rotmat = torch.einsum('bij,bjk->bik', old_rotmat, new_rotmat)  # [b,3,3]
    transf_transl = torch.einsum('bij,btj->bti', old_rotmat, new_transl) + old_transl  # [b,1,3]

    return transf_rotmat, transf_transl


def canonicalize(transl, root_rot, joints, return_transf=False):
    transf_rotmat, transf_transl = get_new_coordinate(joints[:, 0])

    root_rot_new = torch.einsum('bij,btjk->btik', transf_rotmat.permute(0, 2, 1), root_rot)
    transl_new = torch.einsum('bij,btj->bti', transf_rotmat.permute(0, 2, 1), transl - transf_transl)

    joints_new = torch.einsum('bij,btkj->btki', transf_rotmat.permute(0, 2, 1), joints - transf_transl.unsqueeze(1))

    if not return_transf:
        return transl_new, root_rot_new, joints_new
    else:
        device = transl.device
        batch_size = transl.shape[0]
        old_rotmat = torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1)
        old_transl = torch.zeros(3, device=device, dtype=torch.float32).reshape(1, 1, 3).repeat(batch_size, 1, 1)
        transf_rotmat, transf_transl = update_global_transform(old_rotmat, old_transl, transf_rotmat, transf_transl)
        return transf_rotmat, transf_transl, transl_new, root_rot_new, joints_new


def __jitable_motion_dict_to_feature_v4__(trans, rot, dof, contact, skeleton, ret_fk_full=False):
    '''
    DART:{
    'transl': 3,
    'poses_6d': 22 * 6,
    'transl_delta': 3,
    'global_orient_delta_6d': 6,
    'joints': 22 * 3,
    'joints_delta': 22 * 3
    }
    '''
    B, T, _ = trans.shape

    rot_matrix = quaternion_to_matrix(xyzw_to_wxyz(rot))  # [B, T, 3, 3]

    motion_dict = {'root_trans_offset': trans, 'root_rot': rot, 'dof': dof, 'contact_mask': contact}

    fk_res = skeleton.forward_kinematics(motion_dict, return_full=ret_fk_full)

    transl_new, root_rot_new, joints_new = canonicalize(trans, rot_matrix, fk_res['global_translation_extend'])
    transl = transl_new  # [B, T, 3]
    transl_delta = transl_new[:, 1:, :] - transl_new[:, :-1, :]  # [B, T-1, 3]
    joints = joints_new.reshape(B, T, -1)  # [B, T, J, 3]
    joints_delta = (joints_new[:, 1:, :] - joints_new[:, :-1, :]).reshape(B, T - 1, -1)  # [B, T-1, J*3]

    rot_6d = matrix_to_rot6d(root_rot_new)  # [B, T, 6]
    # delta rotation
    rot_delta_rotmat = torch.matmul(root_rot_new[:, 1:], root_rot_new[:, :-1].permute(0, 1, 3, 2))
    rot_delta_6d = matrix_to_rot6d(rot_delta_rotmat)  # [B, t-1, 6]

    # dof: [B, T, 29]
    dof = dof[:, :-1, :]  # [B, T-1, 29]
    transl_feature = transl[:, :-1, :]  # [B, T-1, 3]
    rot_6d_feature = rot_6d[:, :-1, :]  # [B, T-1, 6]
    joints_feature = joints[:, :-1, :]  # [B, T-1, J*3]

    # breakpoint()
    feature = torch.cat(
        [
            transl_feature,  # 3
            rot_6d_feature,  # 6
            dof,  # 29
            transl_delta,  # 3
            rot_delta_6d,  # 6
            joints_feature,  # J*3
            joints_delta,  # J*3
            contact[:, :-1]  # 2
        ],
        dim=-1
    )

    return feature


def motion_dict_to_feature_v4(motion_dict, skeleton):
    trans = motion_dict['root_trans_offset']  # ([B,] T+1, 3)
    rot = motion_dict['root_rot']  # ([B], T+1, 4)
    dof = motion_dict['dof']  # ([B], T+1, 29)
    contact = motion_dict['contact_mask']  # ([B], T+1, 2)
    if len(trans.shape) == 2:
        expanded = True
        trans = trans.unsqueeze(0)
        rot = rot.unsqueeze(0)
        dof = dof.unsqueeze(0)
        contact = contact.unsqueeze(0)
    else:
        expanded = False
    B, T_plus_1, _ = trans.shape
    T = T_plus_1 - 1  # 输出T帧

    abs_pose = {'root_trans_offset': trans[:, 0], 'root_rot': rot[:, 0]}

    feature = __jitable_motion_dict_to_feature_v4__(trans, rot, dof, contact, skeleton)

    if expanded:
        feature = feature[0]
        abs_pose = {'root_trans_offset': abs_pose['root_trans_offset'][0], 'root_rot': abs_pose['root_rot'][0]}
    return feature, abs_pose


def extract_yaw_from_rotation(rotmat):
    """
    从旋转矩阵中精确提取Yaw角度
    使用欧拉角分解：ZYX顺序（Yaw-Pitch-Roll）
    """
    # 使用旋转矩阵到欧拉角转换
    # R = Rz(yaw) * Ry(pitch) * Rx(roll)
    sin_pitch = -rotmat[:, 2, 0]
    sin_pitch = torch.clamp(sin_pitch, -1.0, 1.0)
    pitch = torch.asin(sin_pitch)

    # 避免万向锁情况
    mask = torch.abs(torch.cos(pitch)) > 1e-6
    yaw = torch.zeros_like(pitch)

    # 正常情况
    yaw[mask] = torch.atan2(rotmat[mask, 1, 0], rotmat[mask, 0, 0])

    # 万向锁情况（pitch ≈ ±90°）
    # 这种情况下Yaw和Roll耦合，我们选择一种合理的处理方式
    roll_mask = ~mask & (rotmat[:, 2, 2] > 0)
    yaw[roll_mask] = torch.atan2(-rotmat[roll_mask, 0, 1], rotmat[roll_mask, 1, 1])

    return yaw


def motion_feature_to_dict_v4(motion_feature, abs_pose):

    if len(motion_feature.shape) == 2:
        expanded = True
        motion_feature = motion_feature.unsqueeze(0)
        if abs_pose:
            abs_pose = {
                'root_trans_offset': abs_pose['root_trans_offset'].unsqueeze(0),
                'root_rot': abs_pose['root_rot'].unsqueeze(0)
            }
    else:
        expanded = False

    B, T = motion_feature.shape[:2]
    if not abs_pose:
        abs_pose = {
            'root_trans_offset': torch.zeros(B, 3, device=motion_feature.device),
            'root_rot': torch.zeros(B, 4, device=motion_feature.device)
        }

    transl_feature = motion_feature[..., :3]  # [B, T, 3]
    rot_6d_feature = motion_feature[..., 3:9]  # [B, T, 6]
    dof_end = 9 + DOF_DIM
    transl_delta_end = dof_end + 3
    rot_delta_end = transl_delta_end + 6
    joints_end = rot_delta_end + (DOF_DIM + 4) * 3
    dof = motion_feature[..., 9:dof_end]
    transl_delta = motion_feature[..., dof_end:transl_delta_end]
    rot_delta_6d = motion_feature[..., transl_delta_end:rot_delta_end]
    joints_feature = motion_feature[..., rot_delta_end:joints_end]
    joints_delta = motion_feature[..., joints_end:-2]
    contact = motion_feature[..., -2:]  # [B, T, 2]

    assert joints_feature.shape[
        -1] == (DOF_DIM + 4) * 3, f"joints_feature shape: {joints_feature.shape}, expected last dim {DOF_DIM*3}"
    assert joints_delta.shape[
        -1] == (DOF_DIM + 4) * 3, f"joints_delta shape: {joints_delta.shape}, expected last dim {DOF_DIM*3}"

    # root_rot = wxyz_to_xyzw(matrix_to_quaternion(rot6d_to_matrix(rot_6d_feature)))
    root_rot_mat = rot6d_to_matrix(rot_6d_feature)
    prev_rot_mat = quaternion_to_matrix(xyzw_to_wxyz(abs_pose['root_rot']))

    prev_yaw = extract_yaw_from_rotation(prev_rot_mat)  # [b]
    curr_yaw = extract_yaw_from_rotation(root_rot_mat[:, 0])  # [b]
    yaw_diff = prev_yaw - curr_yaw  # [b]

    # 4. 创建仅包含Yaw旋转的对齐矩阵
    def create_yaw_rotation(yaw_angles):
        cos_yaw = torch.cos(yaw_angles)
        sin_yaw = torch.sin(yaw_angles)
        zeros = torch.zeros_like(yaw_angles)
        ones = torch.ones_like(yaw_angles)

        # 仅Yaw旋转的矩阵
        yaw_rotmat = torch.stack(
            [
                torch.stack([cos_yaw, -sin_yaw, zeros], dim=-1),
                torch.stack([sin_yaw, cos_yaw, zeros], dim=-1),
                torch.stack([zeros, zeros, ones], dim=-1)
            ],
            dim=-2
        )  # [b, 3, 3]

        return yaw_rotmat

    align_yaw_rotmat = create_yaw_rotation(yaw_diff)  # [b, 3, 3]

    # align_rotmat = torch.matmul(prev_rot_mat, root_rot_mat[:, 0].transpose(1, 2))  # [b, 3, 3]
    # root_rot_mat = torch.matmul(align_rotmat.unsqueeze(1), root_rot_mat)  # [b, T, 3, 3]
    root_rot_mat = torch.matmul(align_yaw_rotmat.unsqueeze(1), root_rot_mat)
    root_rot = wxyz_to_xyzw(matrix_to_quaternion(root_rot_mat))

    # 4. 同时对平移进行对齐
    # 获取关键帧的位置差
    prev_transl = abs_pose['root_trans_offset'][:, :2]  # [b, 2]
    curr_transl = transl_feature[:, 0, :2]  # [b, 2]

    align_yaw_rotmat_2d = align_yaw_rotmat[:, :2, :2]  # [b, 2, 2]

    # 计算位置偏移（在当前序列的坐标系下）
    transl_offset = prev_transl - torch.matmul(align_yaw_rotmat_2d, curr_transl.unsqueeze(-1)).squeeze(-1)

    current_transl_aligned = transl_feature.clone()
    # 应用位置偏移
    current_transl_aligned[:, :, :2] = torch.matmul(
        align_yaw_rotmat_2d.unsqueeze(1), transl_feature[:, :, :2].unsqueeze(-1)
    ).squeeze(-1) + transl_offset.unsqueeze(1)
    transl_feature[:, :, :2] = current_transl_aligned[:, :, :2]

    # transl_feature[:, 0, [0,1]] = abs_pose['root_trans_offset'][..., [0, 1]]
    # if T > 1:
    #     transl_feature[:, 1:, [0,1]] += abs_pose['root_trans_offset'].unsqueeze(1)[..., [0, 1]]

    delta_T = torch.zeros(B, 1, 3, device=transl_feature.device)
    delta_T[:, :, 2] = G1_ROOT_HEIGHT
    transl_feature += delta_T

    if expanded:
        trans = trans[0]
        rot = rot[0]
        dof_out = dof_out[0]
        contact = contact[0]
        abs_pose = {'root_trans_offset': abs_pose['root_trans_offset'][0], 'root_rot': abs_pose['root_rot'][0]}

    return {
        "root_trans_offset": transl_feature,
        "root_rot": root_rot,
        "dof": dof,
        "contact_mask": contact,
        "transl_delta": transl_delta,
        "rot_delta_6d": rot_delta_6d,
        "joints": joints_feature,
        "joints_delta": joints_delta
    }


def get_blended_feature(feature_dict, skeleton, ret_fk_full=False):
    trans = feature_dict['root_trans_offset']
    rot = feature_dict['root_rot']
    dof = feature_dict['dof']
    contact = feature_dict['contact_mask']

    B, T, _ = trans.shape

    joints = feature_dict['joints'].reshape(B, T, (DOF_DIM + 4), 3)

    rot_matrix = quaternion_to_matrix(xyzw_to_wxyz(rot))  # [B, T, 3, 3]

    motion_dict = {'root_trans_offset': trans, 'root_rot': rot, 'dof': dof, 'contact_mask': contact}

    fk_res = skeleton.forward_kinematics(motion_dict, return_full=ret_fk_full)

    transf_rotmat, transf_transl, transl_new, root_rot_new, joints_new = canonicalize(
        trans, rot_matrix, fk_res['global_translation_extend'], return_transf=True
    )
    # transf_rotmat, transf_transl, transl_new, root_rot_new, joints_new = canonicalize(trans, rot_matrix, joints, return_transf=True)

    rot_matrix = torch.matmul(transf_rotmat.permute(0, 2, 1).unsqueeze(1), rot_matrix)
    root_rot = matrix_to_rot6d(rot_matrix)

    rot_delta_6d = feature_dict['rot_delta_6d']  # [b, T, 6], not change
    rot_delta_matrix = rot6d_to_matrix(rot_delta_6d)
    rot_delta_matrix = torch.matmul(
        torch.matmul(transf_rotmat.permute(0, 2, 1).unsqueeze(1), rot_delta_matrix), transf_rotmat.unsqueeze(1)
    )

    rot_delta_6d = matrix_to_rot6d(rot_delta_matrix)  # [b, T, 6]
    transl = transl_new
    joints = joints_new.reshape(B, T, (DOF_DIM + 4) * 3)
    transl_delta = feature_dict['transl_delta']  # [b, T, 3]
    joints_delta = feature_dict['joints_delta'].reshape(B, T, (DOF_DIM + 4), 3)
    transl_delta = torch.einsum('bij,btj->bti', transf_rotmat.permute(0, 2, 1), transl_delta)  # [b,3]
    joints_delta = torch.einsum('bij,btkj->btki', transf_rotmat.permute(0, 2, 1),
                                joints_delta).reshape(B, T, (DOF_DIM + 4) * 3)

    feature_dict_new = {
        "root_trans_offset": transl,
        "root_rot": root_rot,
        "dof": dof,
        "contact_mask": contact,
        "transl_delta": transl_delta,
        "rot_delta_6d": rot_delta_6d,
        "joints": joints,
        "joints_delta": joints_delta
    }

    return transf_rotmat, transf_transl, feature_dict_new


def transform_feature_to_world(feature_dict):
    transf_rotmat, transf_transl = feature_dict['transf_rotmat'], feature_dict['transf_transl']
    device = transf_rotmat.device
    batch_size = transf_rotmat.shape[0]
    dtype = transf_rotmat.dtype

    B, T, _ = feature_dict['root_trans_offset'].shape

    delta_T = torch.zeros(B, 1, 3, device=device, dtype=dtype)
    delta_T[:, :, 2] = G1_ROOT_HEIGHT

    root_rot_quat = feature_dict['root_rot']  # [b, T, 4]
    # breakpoint()
    global_orient_rotmat = quaternion_to_matrix(xyzw_to_wxyz(root_rot_quat))  # [b, T, 3, 3]
    global_orient_rotmat = torch.matmul(transf_rotmat.unsqueeze(1), global_orient_rotmat)
    root_rot = wxyz_to_xyzw(matrix_to_quaternion(global_orient_rotmat))  # [b, T, 4]
    # new_poses_6d = torch.cat([global_orient_6d, poses_6d[:, :, 6:]], dim=-1)  # [b, T, 22*6]

    global_orient_delta_6d = feature_dict['rot_delta_6d']  # [b, T, 6], not change
    global_orient_delta_rotmat = rot6d_to_matrix(global_orient_delta_6d)  # [b, T, 3, 3]
    global_orient_delta_rotmat = torch.matmul(
        torch.matmul(transf_rotmat.unsqueeze(1), global_orient_delta_rotmat),
        transf_rotmat.permute(0, 2, 1).unsqueeze(1)
    )
    global_orient_delta_6d = matrix_to_rot6d(global_orient_delta_rotmat)  # [b, T, 6]

    root_trans_offset = feature_dict['root_trans_offset']  # [b, T, 3]
    joints = feature_dict['joints'].reshape(B, T, (DOF_DIM + 4), 3)  # [b, T, 27, 3]
    root_trans_offset = torch.einsum('bij,btj->bti', transf_rotmat, root_trans_offset + delta_T) + transf_transl
    joints = torch.einsum('bij,btkj->btki', transf_rotmat, joints) + transf_transl.unsqueeze(1)
    joints = joints.reshape(B, T, (DOF_DIM + 4) * 3)
    transl_delta = feature_dict['transl_delta']  # [b, T, 3]
    joints_delta = feature_dict['joints_delta'].reshape(B, T, (DOF_DIM + 4), 3)  # [b, T, 22*3]
    transl_delta = torch.einsum('bij,btj->bti', transf_rotmat, transl_delta)  # [b,3]
    joints_delta = torch.einsum('bij,btkj->btki', transf_rotmat, joints_delta).reshape(B, T, (DOF_DIM + 4) * 3)

    world_feature_dict = {
        'transf_rotmat': torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1),
        'transf_transl': torch.zeros(3, device=device, dtype=dtype).reshape(1, 1, 3).repeat(batch_size, 1, 1),
        'root_trans_offset': root_trans_offset,
        'root_rot': root_rot,
        'dof': feature_dict['dof'],
        'transl_delta': transl_delta,
        'joints': joints,
        'joints_delta': joints_delta,
        'contact_mask': feature_dict['contact_mask']
    }
    return world_feature_dict


def dict_concat(history_dict, future_dict):
    concat_dict = {}
    for key in future_dict.keys():
        if key in history_dict:
            concat_dict[key] = torch.cat([history_dict[key], future_dict[key]], dim=1)
    return concat_dict


def dict_to_tensor(motion_dict):
    motion_repr = {
        'root_trans_offset': 3,
        'root_rot': 6,
        'dof': DOF_DIM,
        'transl_delta': 3,
        'rot_delta_6d': 6,
        'joints': (DOF_DIM + 4) * 3,
        'joints_delta': (DOF_DIM + 4) * 3,
        'contact_mask': 2
    }
    tensors = [motion_dict[key] for key in motion_repr]
    merged_tensor = torch.cat(tensors, dim=-1)
    return merged_tensor


# @torch.jit.script
def __jitable_motion_dict_to_feature_v5__(trans, rot, dof, contact, joints):
    '''
    trans_local: (B, T, 3)
    delta_trans_local: (B, T, 3)
    height: (B, T, 1)
    sincos: (B, T, 1)
    delta_yaw: (B, T, 1)
    contact_T: (B, T, 2)
    dof_T: (B, T, 29)
    delta_dof: (B, T, 29)
    joints_local: (B, T, (DoF_DIM+4)*3)
    delta_joints_local: (B, T, (DoF_DIM+4)*3)
    '''

    B, T_plus_1, _ = trans.shape
    T = T_plus_1 - 1  # 输出T帧
    # 只使用前T帧的高度
    height = trans[:, :T, 2]  #- 0.8

    # 计算前T帧的欧拉角
    euler = quaternion_to_euler_angles(rot[:, :T + 1])
    roll = euler[..., 0]
    pitch = euler[..., 1]
    yaw = euler[..., 2]
    # print(f"DEBUG: {yaw=}")

    sincos = torch.stack([torch.sin(roll), torch.cos(roll) - 1, torch.sin(pitch), torch.cos(pitch) - 1], dim=-1)[:, :T]

    # 计算delta yaw: (t+1) - t，使用前T帧
    # 对于第t帧，我们需要计算(t+1) - t的delta
    delta_yaw = (yaw[:, 1:T + 1] - yaw[:, :T]).unsqueeze(-1)  # (B, T, 1)

    # 计算位置变化（t+1） - t，使用前T帧
    delta_trans_world = trans[:, 1:T + 1] - trans[:, :T]  # (B, T, 3)
    # print(f"DEBUG: {delta_trans_world=}")

    ref_yaw_quat = euler_angles_to_quaternion(
        torch.stack([torch.zeros_like(yaw), torch.zeros_like(yaw), yaw], dim=-1)
    )[:, :T]  # (B, T, 4)
    inv_ref_yaw_quat = quat_inverse(ref_yaw_quat, True)
    # print(f"DEBUG: {inv_ref_yaw_quat=}")

    trans_local = quat_apply(inv_ref_yaw_quat, trans[:, :T], True)  # (B, T, 3)

    # 将位移转换到以第一帧Yaw为参考的局部坐标系
    delta_trans_local = quat_apply(inv_ref_yaw_quat, delta_trans_world, True)  # (B, T, 3)

    delta_joints_world = joints[:, 1:] - joints[:, :-1]  # (B, T, DoF_DIM+4, 3)

    # broadcast inv_ref_yaw_quat to match joints shape
    inv_ref_yaw_quat_expand = inv_ref_yaw_quat.unsqueeze(-2).expand(
        -1, -1, joints.shape[-2], -1
    )  # (B, T, DoF_DIM+4, 4)
    joints_local = quat_apply(inv_ref_yaw_quat_expand, joints[:, :T], True).reshape(B, T, -1)  # (B, T, (DoF_DIM+4)*3)
    delta_joints_local = quat_apply(inv_ref_yaw_quat_expand, delta_joints_world,
                                    True).reshape(B, T, -1)  # (B, T, (DoF_DIM+4)*3)

    # 计算delta dof: (t+1) - t，使用前T帧
    delta_dof = dof[:, 1:T + 1] - dof[:, :T]  # (B, T, 29)

    # breakpoint()

    # 只使用前T帧的contact
    contact_T = contact[:, :T]

    # 只使用前T帧的dof
    dof_T = dof[:, :T]

    # 确保所有tensor的维度都是(T, ...)
    assert sincos.shape[1] == T, f"sincos shape: {sincos.shape}"
    assert delta_yaw.shape[1] == T, f"delta_yaw shape: {delta_yaw.shape}"
    assert contact_T.shape[1] == T, f"contact_T shape: {contact_T.shape}"
    assert trans_local.shape[1] == T, f"trans_local shape: {trans_local.shape}"
    assert delta_trans_local.shape[1] == T, f"delta_trans_local shape: {delta_trans_local.shape}"
    assert joints_local.shape[1] == T, f"joints_local shape: {joints_local.shape}"
    assert delta_joints_local.shape[1] == T, f"delta_joints_local shape: {delta_joints_local.shape}"
    assert height.shape[1] == T, f"height shape: {height.shape}"
    assert dof_T.shape[1] == T, f"dof_T shape: {dof_T.shape}"
    assert delta_dof.shape[1] == T, f"delta_dof shape: {delta_dof.shape}"

    feature = torch.cat(
        [
            sincos, delta_yaw, contact_T, trans_local, delta_trans_local, joints_local, delta_joints_local,
            height.unsqueeze(-1), dof_T, delta_dof
        ],
        dim=-1
    )

    return feature


def motion_dict_to_feature_v5(motion_dict, skeleton):
    # 输入N+1帧，返回N帧的feature
    trans = motion_dict['root_trans_offset']  # ([B,] T+1, 3)
    rot = motion_dict['root_rot']  # ([B], T+1, 4)
    dof = motion_dict['dof']  # ([B], T+1, 29)
    contact = motion_dict['contact_mask']  # ([B], T+1, 2)
    if len(trans.shape) == 2:
        expanded = True
        trans = trans.unsqueeze(0)
        rot = rot.unsqueeze(0)
        dof = dof.unsqueeze(0)
        contact = contact.unsqueeze(0)
    else:
        expanded = False
    B, T_plus_1, _ = trans.shape
    T = T_plus_1 - 1  # 输出T帧

    abs_pose = {'root_trans_offset': trans[:, 0], 'root_rot': rot[:, 0]}

    motion_dict = {'root_trans_offset': trans, 'root_rot': rot, 'dof': dof, 'contact_mask': contact}

    fk_res = skeleton.forward_kinematics(motion_dict, return_full=False)
    joints = fk_res['global_translation_extend']  # (B, T+1, DoF_DIM+4, 3)

    feature = __jitable_motion_dict_to_feature_v5__(trans, rot, dof, contact, joints)

    if expanded:
        feature = feature[0]
        abs_pose = {'root_trans_offset': abs_pose['root_trans_offset'][0], 'root_rot': abs_pose['root_rot'][0]}
    return feature, abs_pose


def motion_feature_to_dict_v5(motion_feature, abs_pose):
    if len(motion_feature.shape) == 2:
        expanded = True
        motion_feature = motion_feature.unsqueeze(0)
        if abs_pose:
            abs_pose = {
                'root_trans_offset': abs_pose['root_trans_offset'].unsqueeze(0),
                'root_rot': abs_pose['root_rot'].unsqueeze(0)
            }
    else:
        expanded = False

    B, T = motion_feature.shape[:2]
    if not abs_pose:
        abs_pose = {
            'root_trans_offset': torch.zeros(B, 3, device=motion_feature.device),
            'root_rot': torch.zeros(B, 4, device=motion_feature.device)
        }

    # print(f"DEBUG: {abs_pose=}")
    # 提取特征组件
    joints_feature_dim = (DOF_DIM + 4) * 3
    sin_roll = motion_feature[..., 0]
    cos_roll = motion_feature[..., 1] + 1
    sin_pitch = motion_feature[..., 2]
    cos_pitch = motion_feature[..., 3] + 1
    delta_yaw = motion_feature[..., 4]  # (t+1) - t
    contact = motion_feature[..., 5:7]
    trans_local = motion_feature[..., 7:10]  # (B, T, 3)
    delta_trans_local = motion_feature[..., 10:13]  # (t+1) - t
    joints_local = motion_feature[..., 13:13 + joints_feature_dim]  # (B, T, (DoF_DIM+4)*3)
    delta_joints_local = motion_feature[..., 13 + joints_feature_dim:13 + 2 * joints_feature_dim]
    height = motion_feature[..., 13 + 2 * joints_feature_dim]  #+ 0.8  # 高度
    dof = motion_feature[..., -2 * DOF_DIM:-DOF_DIM]
    delta_dof = motion_feature[..., -DOF_DIM:]

    # 计算roll和pitch
    roll = torch.atan2(sin_roll, cos_roll)
    pitch = torch.atan2(sin_pitch, cos_pitch)

    # print(f"DEBUG: {delta_trans_local=}")
    # 从绝对位姿获取第一帧的绝对Yaw
    init_euler = quaternion_to_euler_angles(abs_pose['root_rot'])
    # print(f"DEBUG: {init_euler=}")
    ref_yaw = init_euler[..., 2]  # (B,)

    # 重建Yaw序列
    yaw = torch.zeros(B, T, device=delta_yaw.device)
    yaw[:, 0] = ref_yaw
    if T > 1:
        yaw[:, 1:] = torch.cumsum(delta_yaw[:, :T - 1], dim=1) + ref_yaw.unsqueeze(1)

    # print(f"DEBUG: {yaw=}")
    # 组合欧拉角并转换为四元数
    euler = torch.stack([roll, pitch, yaw], dim=-1)
    rot = euler_angles_to_quaternion(euler)

    # 位置重建
    ref_yaw_quat = euler_angles_to_quaternion(torch.stack([torch.zeros_like(yaw), torch.zeros_like(yaw), yaw], dim=-1))
    # print(f"DEBUG: {ref_yaw_quat=}")
    delta_trans_world = quat_apply(ref_yaw_quat, delta_trans_local, True)

    trans_pred = quat_apply(ref_yaw_quat, trans_local, True)

    ref_yaw_quat_expand = ref_yaw_quat.unsqueeze(-2).expand(-1, -1, DOF_DIM + 4, -1)  # (B, T, DoF_DIM+4, 4)
    joints_pred = quat_apply(ref_yaw_quat_expand, joints_local.reshape(B, -1, (DOF_DIM + 4), 3),
                             True).reshape(B, -1, (DOF_DIM + 4) * 3)
    delta_joints_world = quat_apply(ref_yaw_quat_expand, delta_joints_local.reshape(B, -1, (DOF_DIM + 4), 3),
                                    True).reshape(B, -1, (DOF_DIM + 4) * 3)

    # breakpoint()
    # print(f"DEBUG: {delta_trans_world=}")

    trans = torch.zeros((B, T, 3), device=motion_feature.device)
    trans[:, 0] = abs_pose['root_trans_offset']
    if T > 1:
        trans[:, 1:] = torch.cumsum(delta_trans_world[:, :T - 1], dim=1) + abs_pose['root_trans_offset'].unsqueeze(1)
    trans[:, :, 2] = height

    # dof直接读取
    dof_out = dof
    if expanded:
        trans = trans[0]
        rot = rot[0]
        dof_out = dof_out[0]
        contact = contact[0]
        abs_pose = {'root_trans_offset': abs_pose['root_trans_offset'][0], 'root_rot': abs_pose['root_rot'][0]}
    return {
        'root_trans_offset': trans,
        'root_rot': rot,
        'dof': dof_out,
        'contact_mask': contact,
        'trans_pred': trans_pred,
        'delta_trans_world': delta_trans_world,
        'joints_pred': joints_pred,
        'delta_joints_world': delta_joints_world,
        'delta_dof': delta_dof
    }


motion_feature_dim_v1 = (4 + 1 + 2 + 3 + 1 + DOF_DIM)
motion_feature_dim_v2 = (4 + 1 + 2 + 3 + 1 + DOF_DIM + DOF_DIM)
motion_feature_dim_v3 = (4 + 1 + 2 + 3 + 1 + DOF_DIM + DOF_DIM)
motion_feature_dim_v4 = (3 + 6 + DOF_DIM + 3 + 6 + (DOF_DIM + 4) * 3 + (DOF_DIM + 4) * 3 + 2)  # 247
motion_feature_dim_v5 = (4 + 1 + 2 + 3 + 3 + (DOF_DIM + 4) * 3 + (DOF_DIM + 4) * 3 + 1 + DOF_DIM + DOF_DIM)  # 270


def get_zero_abs_pose(batch_shape: Tuple[int, ...], device: str = 'cuda') -> AbsolutePose:
    root_rot = torch.zeros(batch_shape + (4, ), device=device)
    root_rot[..., 3] = 1.0
    # root_rot[..., 2] = 1.0
    # root_rot[..., 2] = 0.6
    # root_rot[..., 3] = 0.8

    return {'root_trans_offset': torch.zeros(batch_shape + (3, ), device=device), 'root_rot': root_rot}


def get_zero_feature_v1() -> MotionFeatureV1:
    """
    Returns a zero-initialized motion feature tensor.
    """
    feat = torch.zeros((1, motion_feature_dim_v1), dtype=torch.float32)
    feat[0, 3] = 1.0  # Set the w component of the quaternion to 1
    feat[0, 10] = 0.8  # Set height
    feat[0, 11:] = torch.tensor(G1_DEFAULT_DOF)
    return feat


def get_zero_feature_v2() -> MotionFeatureV2:
    """
    Returns a zero-initialized motion feature tensor for v2.
    """
    feat = torch.zeros((1, motion_feature_dim_v2), dtype=torch.float32)
    feat[0, 5:7] = 1.0  # contact mask
    feat[0, 10] = 0.75  # Set height
    feat[0, 11:11 + DOF_DIM] = torch.tensor(G1_DEFAULT_DOF)
    # delta dof部分保持为0
    return feat


get_zero_feature_v3 = get_zero_feature_v2


def get_zero_feature_v4(skeleton):
    root_rot = torch.zeros((1, 4))
    root_rot[..., 3] = 1.0
    dof = torch.tensor(G1_DEFAULT_DOF)

    motion_dict = {
        'root_trans_offset': torch.tensor([0.0, 0.0, G1_ROOT_HEIGHT]).unsqueeze(0),
        'root_rot': root_rot,
        'dof': dof.unsqueeze(0),
        'contact_mask': torch.tensor([1.0, 1.0]).unsqueeze(0)
    }
    fk_res = skeleton.forward_kinematics(motion_dict, return_full=False)

    feat = torch.zeros((1, motion_feature_dim_v4), dtype=torch.float32)
    # feat[0, 2] = G1_ROOT_HEIGHT  # root_rot 6d
    # feat[0, 9:32] = dof
    # feat[0, 41:41 + (DOF_DIM+4)*3] = fk_res['global_translation_extend'].reshape(-1)
    # feat[0, -2:] = torch.tensor([1.0, 1.0])  # contact mask
    return feat


def get_zero_feature_v5():
    """
    Returns a zero-initialized motion feature tensor for v5.
    """
    # return STATIC_POSE
    raise NotImplementedError


def perturb_feature_v3(motion_feature: MotionFeatureV3, scale: float) -> MotionFeatureV3:
    """
    Add random perturbation to motion features with component-specific strategies.

    Args:
        motion_feature: Input motion feature tensor [B, T, D]
        scale: Perturbation scale factor

    Returns:
        Perturbed motion feature tensor
    """
    if scale <= 0:
        return motion_feature

    # Generate random noise
    noise = torch.randn_like(motion_feature) * scale

    # Apply different perturbation strategies for different feature components
    perturbed = motion_feature.clone()

    # 对sin/cos部分添加小扰动后重新归一化
    sincos_noise = noise[..., :4] * 0.5  # 较小的扰动
    perturbed[..., :4] = motion_feature[..., :4] + sincos_noise
    # 重新归一化sin/cos对
    for i in range(0, 4, 2):
        norm = torch.sqrt(perturbed[..., i]**2 + (perturbed[..., i + 1] + 1)**2)
        perturbed[..., i] /= norm
        perturbed[..., i + 1] = (perturbed[..., i + 1] + 1) / norm - 1

    # 对其他特征添加正常扰动
    if motion_feature.shape[-1] > 4:
        perturbed[..., 4:] = motion_feature[..., 4:] + noise[..., 4:]

    perturbed[..., 5:7] = torch.clamp(perturbed[..., 5:7], 0, 1)

    return perturbed


motion_dict_to_feature: Callable[[MotionDict, Optional[Any]], Any]
if FeatureVersion == 0:
    motion_dict_to_feature = motion_dict_to_feature_v0
    motion_feature_to_dict = motion_feature_to_dict_v0
    motion_feature_dim = ROOT_TRANS_OFFSET_DIM + ROOT_ROT_DIM + DOF_DIM + CONTACT_MASK_DIM
elif FeatureVersion == 1:
    motion_dict_to_feature = motion_dict_to_feature_v1
    motion_feature_to_dict = motion_feature_to_dict_v1
    motion_feature_dim = motion_feature_dim_v1
elif FeatureVersion == 2:
    motion_dict_to_feature = motion_dict_to_feature_v2
    motion_feature_to_dict = motion_feature_to_dict_v2
    motion_feature_dim = motion_feature_dim_v2
elif FeatureVersion == 3:
    motion_dict_to_feature = motion_dict_to_feature_v3
    motion_feature_to_dict = motion_feature_to_dict_v3
    motion_feature_dim = motion_feature_dim_v3
    get_zero_feature = get_zero_feature_v3
elif FeatureVersion == 4:
    motion_dict_to_feature = motion_dict_to_feature_v4
    motion_feature_to_dict = motion_feature_to_dict_v4
    motion_feature_dim = motion_feature_dim_v4
    get_zero_feature = get_zero_feature_v4
elif FeatureVersion == 5:
    motion_dict_to_feature = motion_dict_to_feature_v5
    motion_feature_to_dict = motion_feature_to_dict_v5
    motion_feature_dim = motion_feature_dim_v5
    get_zero_feature = get_zero_feature_v5
else:
    raise ValueError(f"Unsupported FeatureVersion: {FeatureVersion}")

if __name__ == "__main__":
    # Example usage
    motion_dict = MotionDict(
        root_trans_offset=torch.zeros(11, ROOT_TRANS_OFFSET_DIM),  # 11帧用于测试v2
        root_rot=torch.zeros(11, ROOT_ROT_DIM),
        dof=torch.zeros(11, DOF_DIM),
        contact_mask=torch.zeros(11, CONTACT_MASK_DIM)
    )
    motion_dict['root_trans_offset'] = torch.arange(33).reshape(11, 3).float()
    motion_dict['root_rot'][:, 3] = np.sqrt(0.5)
    motion_dict['root_rot'][:, 1] = 0.5
    motion_dict['root_rot'][:, 2] = 0.5
    motion_dict['root_rot'][::2, 3] = 1.0
    motion_dict['root_rot'][::2, 2] = 0.0
    motion_dict['root_rot'][::2, 1] = 0.0

    # 测试v1
    print("Testing FeatureVersion 1:")
    feature_v1, abs_pose_v1 = motion_dict_to_feature_v1(motion_dict)
    reconstructed_motion_dict_v1 = motion_feature_to_dict_v1(feature_v1, abs_pose_v1)
    print("Feature v1 shape:", feature_v1.shape)
    for key in motion_dict:
        assert torch.allclose(motion_dict[key], reconstructed_motion_dict_v1[key], atol=1e-4), \
            f"Mismatch in {key} for FeatureVersion 1"
    print("v1 test passed!")

    # 测试v2
    print("\nTesting FeatureVersion 2:")
    feature_v2, abs_pose_v2 = motion_dict_to_feature_v2(motion_dict)
    reconstructed_motion_dict_v2 = motion_feature_to_dict_v2(feature_v2, abs_pose_v2)
    print("Feature v2 shape:", feature_v2.shape)
    print("Input frames:", motion_dict['root_trans_offset'].shape[0])
    print("Output frames:", reconstructed_motion_dict_v2['root_trans_offset'].shape[0])

    # v2应该输出比输入少一帧
    assert reconstructed_motion_dict_v2['root_trans_offset'].shape[0] == motion_dict['root_trans_offset'].shape[0] - 1, \
        "v2 should output one frame less than input"

    # 检查前N帧是否匹配（除了最后一帧）
    for key in motion_dict:
        original_data = motion_dict[key][:-1]  # 去掉最后一帧
        reconstructed_data = reconstructed_motion_dict_v2[key]
        assert torch.allclose(original_data, reconstructed_data, atol=1e-4), \
            f"Mismatch in {key} for FeatureVersion 2"
    print("v2 test passed!")

    print(f"\nFeatureVersion {FeatureVersion} is active")
    breakpoint()
