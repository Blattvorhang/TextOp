import torch
import numpy as np
import xml.etree.ElementTree as ETree
from pathlib import Path
from isaac_utils.rotations import (quaternion_to_matrix,
                                   axis_angle_to_quaternion,
                                   matrix_to_quaternion, wxyz_to_xyzw,
                                   quat_identity_like, quat_mul_norm,
                                   quat_inverse, quat_angle_axis)
from easydict import EasyDict
import scipy.ndimage.filters as filters
from typing import Optional
from omegaconf import DictConfig
from loguru import logger


class ForwardKinematics:
    """
    Forward kinematics calculator for humanoid robots.
    Maintains full compatibility with original interface including extend joints.
    """

    def __init__(self,
                 device: torch.device = torch.device("cpu"),
                 cfg: Optional[DictConfig] = None):
        """
        Initialize FK calculator from MJCF file.
        
        Args:
            mjcf_file: Path to MJCF file
            device: Target device for tensors
            cfg: Configuration object with extend_config
        """
        self.device = device
        self.dof_axis: torch.Tensor  # type annotation for linter
        self.mjcf_file = Path(cfg.asset.assetRoot) / cfg.asset.assetFileName
        self.cfg = cfg

        # Parse skeleton structure
        skeleton_data = self._parse_mjcf()

        # Extract skeleton information
        self.body_names = skeleton_data['node_names']
        self.parent_indices = skeleton_data['parent_indices'].to(device)
        # Add batch dimension to match original implementation
        self.local_translations = skeleton_data['local_translations'][
            None,
        ].to(device)  # [1, num_bodies, 3]
        self.local_rotations = skeleton_data['local_rotations'][
            None,
        ].to(device)  # [1, num_bodies, 4]

        # Initialize extended configuration
        self.body_names_augment = self.body_names.copy()
        self.num_extend_dof = len(
            self.body_names) - 1  # Assuming root has no DOF

        # Handle extended configuration if provided
        if cfg and hasattr(cfg, 'extend_config'):
            for extend_config in cfg.extend_config:
                self.body_names_augment += [extend_config.joint_name]
                self.parent_indices = torch.cat([
                    self.parent_indices,
                    torch.tensor([
                        self.body_names.index(extend_config.parent_name)
                    ]).to(device)
                ],
                                                dim=0)
                self.local_translations = torch.cat([
                    self.local_translations,
                    torch.tensor([[extend_config.pos]]).to(device)
                ],
                                                    dim=1)
                self.local_rotations = torch.cat([
                    self.local_rotations,
                    torch.tensor([[extend_config.rot]]).to(device)
                ],
                                                 dim=1)
                self.num_extend_dof += 1

        # Convert local rotations to matrices
        self.local_rotation_matrices = quaternion_to_matrix(
            self.local_rotations).float()

        # Skeleton info
        self.num_bodies = len(self.body_names)
        self.num_bodies_augment = len(self.body_names_augment)
        self.actuated_joints_idx = torch.arange(1,
                                                self.num_bodies)  # Skip root

    def _parse_mjcf(self):
        """Parse MJCF file to extract skeleton structure"""
        tree = ETree.parse(self.mjcf_file)
        xml_world_body = tree.getroot().find("worldbody")

        if xml_world_body is None:
            raise ValueError("Invalid MJCF file: no worldbody found")

        xml_body_root = xml_world_body.find("body")
        if xml_body_root is None:
            raise ValueError("Invalid MJCF file: no body found")

        # Initialize data structures
        node_names = []
        parent_indices = []
        local_translations = []
        local_rotations = []

        def _parse_node(xml_node, parent_idx, node_idx):
            """Recursively parse XML node"""
            node_name = xml_node.attrib.get("name", f"body_{node_idx}")

            # Parse position and rotation
            pos_str = xml_node.attrib.get("pos", "0 0 0")
            quat_str = xml_node.attrib.get("quat", "1 0 0 0")

            pos = np.fromstring(pos_str, dtype=float, sep=" ")
            quat = np.fromstring(quat_str, dtype=float, sep=" ")  # wxyz format

            # Store data
            node_names.append(node_name)
            parent_indices.append(parent_idx)
            local_translations.append(pos)
            local_rotations.append(quat)

            current_idx = node_idx
            node_idx += 1

            # Recursively parse child bodies
            for child_body in xml_node.findall("body"):
                node_idx = _parse_node(child_body, current_idx, node_idx)

            return node_idx

        # Start parsing from root
        _parse_node(xml_body_root, -1, 0)

        # Calculate DOF axis
        dof_axis_list = []
        actuator = tree.getroot().find("actuator")
        if actuator is None:
            raise ValueError("Invalid MJCF file: no actuator found")
        motors = sorted(
            [m.attrib['name'] for m in list(actuator) if 'name' in m.attrib])
        assert len(motors) > 0, "No motors found in the mjcf file"

        self.num_dof = len(motors)
        self.num_extend_dof = self.num_dof

        joint_nodes = xml_world_body.findall('.//joint')
        if not joint_nodes or len(joint_nodes) == 0:
            raise ValueError("Invalid MJCF file: no joints found")
        joints = joint_nodes

        # Defensive: check axis exists in attrib
        def get_axis(j):
            axis_str = j.attrib.get('axis', None)
            if axis_str is None:
                raise ValueError(
                    f"Joint {j.attrib.get('name', 'unknown')} missing axis attribute"
                )
            return [int(i) for i in axis_str.split(" ")]

        if "type" in joints[0].attrib and joints[0].attrib['type'] == "free":
            for j in joints[1:]:
                dof_axis_list.append(get_axis(j))
            self.has_freejoint = True
        elif "type" not in joints[0].attrib:
            for j in joints:
                dof_axis_list.append(get_axis(j))
            self.has_freejoint = True
        else:
            for j in joints[6:]:
                dof_axis_list.append(get_axis(j))
            self.has_freejoint = False

        self.dof_axis = torch.tensor(dof_axis_list)

        return {
            'node_names':
            node_names,
            'parent_indices':
            torch.from_numpy(np.array(parent_indices, dtype=np.int32)),
            'local_translations':
            torch.from_numpy(np.array(local_translations, dtype=np.float32)),
            'local_rotations':
            torch.from_numpy(np.array(local_rotations, dtype=np.float32))
        }

    def dof_to_axis_angle(self, dof: torch.Tensor) -> torch.Tensor:
        """Convert DOF to axis-angle"""
        return dof.unsqueeze(-1) * self.dof_axis.to(dof.device)

    def fk_batch(self,
                 pose: torch.Tensor,
                 trans: torch.Tensor,
                 convert_to_mat=True,
                 return_full=False,
                 dt=1 / 30):
        """
        Compute forward kinematics with full compatibility to original interface.
        
        Args:
            pose: Joint angles in axis-angle format [batch, seq_len, num_joints, 3]
            trans: Root translation [batch, seq_len, 3]
            convert_to_mat: Whether to convert to rotation matrices
            return_full: Whether to return full information including velocities
            dt: Time delta for velocity computation
            
        Returns:
            EasyDict with all original keys
        """
        device, dtype = pose.device, pose.dtype
        pose_input = pose.clone()
        B, seq_len = pose.shape[:2]
        pose = pose[..., :len(self.parent_indices), :]

        if convert_to_mat:
            pose_quat = axis_angle_to_quaternion(pose.clone())
            pose_mat = quaternion_to_matrix(pose_quat)
        else:
            pose_mat = pose

        if pose_mat.shape != 5:
            pose_mat = pose_mat.reshape(B, seq_len, -1, 3, 3)
        J = pose_mat.shape[2] - 1
        wbody_pos, wbody_mat = self.forward_kinematics_batch(
            pose_mat[:, :, 1:], pose_mat[:, :, 0:1], trans)

        return_dict = EasyDict()

        wbody_rot = wxyz_to_xyzw(matrix_to_quaternion(wbody_mat))

        # Handle extended configuration exactly like original
        if self.cfg and hasattr(self.cfg, 'extend_config') and len(
                self.cfg.extend_config) > 0:
            if return_full:
                return_dict.global_velocity_extend = self._compute_velocity(
                    wbody_pos, dt)
                return_dict.global_angular_velocity_extend = self._compute_angular_velocity(
                    wbody_rot, dt)

            return_dict.global_translation_extend = wbody_pos.clone()
            return_dict.global_rotation_mat_extend = wbody_mat.clone()
            return_dict.global_rotation_extend = wbody_rot

            wbody_pos = wbody_pos[..., :self.num_bodies, :]
            wbody_mat = wbody_mat[..., :self.num_bodies, :, :]
            wbody_rot = wbody_rot[..., :self.num_bodies, :]

        return_dict.global_translation = wbody_pos
        return_dict.global_rotation_mat = wbody_mat
        return_dict.global_rotation = wbody_rot  # (B, T, N, 4), xyzw

        # DOF position calculation exactly like original
        if self.cfg and hasattr(self.cfg, 'extend_config') and len(
                self.cfg.extend_config) > 0:
            return_dict.dof_pos = pose.sum(dim=-1)[..., 1:self.num_bodies]
        else:
            if not len(self.actuated_joints_idx) == len(self.body_names):
                return_dict.dof_pos = pose.sum(
                    dim=-1)[..., self.actuated_joints_idx]
            else:
                return_dict.dof_pos = pose.sum(dim=-1)[..., 1:]

        dof_vel = ((return_dict.dof_pos[:, 1:] - return_dict.dof_pos[:, :-1]) /
                   dt)
        return_dict.dof_vel = torch.cat([dof_vel, dof_vel[:, -2:-1]], dim=1)
        return_dict.fps = int(1 / dt)

        if return_full:
            rigidbody_linear_velocity = self._compute_velocity(wbody_pos, dt)
            rigidbody_angular_velocity = self._compute_angular_velocity(
                wbody_rot, dt)
            return_dict.local_rotation = wxyz_to_xyzw(pose_quat)
            return_dict.global_root_velocity = rigidbody_linear_velocity[...,
                                                                         0, :]
            return_dict.global_root_angular_velocity = rigidbody_angular_velocity[
                ..., 0, :]
            return_dict.global_angular_velocity = rigidbody_angular_velocity
            return_dict.global_velocity = rigidbody_linear_velocity

        return return_dict

    def forward_kinematics_batch(self, rotations, root_rotations,
                                 root_positions):
        """Perform forward kinematics using the given trajectory and local rotations"""
        device, dtype = root_rotations.device, root_rotations.dtype
        B, seq_len = rotations.size()[0:2]
        J = self.local_translations.shape[1]  # Now shape is [1, num_bodies, 3]
        positions_world = []
        rotations_world = []

        expanded_offsets = (self.local_translations.expand(
            B, seq_len, J, 3).to(device).type(dtype))

        for i in range(J):
            if self.parent_indices[i] == -1:
                positions_world.append(root_positions)
                rotations_world.append(root_rotations)
            else:
                jpos = (torch.matmul(
                    rotations_world[self.parent_indices[i]][:, :, 0],
                    expanded_offsets[:, :, i, :, None]).squeeze(-1) +
                        positions_world[self.parent_indices[i]])
                rot_mat = torch.matmul(
                    rotations_world[self.parent_indices[i]],
                    torch.matmul(
                        self.local_rotation_matrices[0, i:i + 1, :, :].expand(
                            B, seq_len, 1, 3, 3).to(device),
                        rotations[:, :, (i - 1):i, :]))

                positions_world.append(jpos)
                rotations_world.append(rot_mat)

        positions_world = torch.stack(positions_world, dim=2)
        rotations_world = torch.cat(rotations_world, dim=2)
        return positions_world, rotations_world

    @staticmethod
    def _gaussian_smooth1d(x: torch.Tensor) -> torch.Tensor:
        """Apply 1-2-1 kernel smoothing along the time dimension (last-3rd axis)."""
        # x: [..., T, N, M]
        orig_shape = x.shape
        T = x.shape[-3]
        x_flat = x.reshape(-1, T)
        x_flat = x_flat.unsqueeze(1)  # (B*N*M, 1, T)
        kernel = torch.tensor([1., 2., 1.], device=x.device, dtype=x.dtype)
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, 3)
        pad = (1, 1)
        x_padded = torch.nn.functional.pad(x_flat, pad, mode='replicate')
        x_smoothed = torch.nn.functional.conv1d(x_padded, kernel)
        return x_smoothed.squeeze(1).reshape(orig_shape)

    @staticmethod
    def _compute_velocity(p, time_delta, gaussian_filter=True):
        """
        Compute velocity from positions (differentiable torch version).
        Args:
            p: Tensor of shape [..., T, N, M] (T is time/sequence dimension)
            time_delta: float, time step between frames
            gaussian_filter: bool, whether to apply smoothing
        Returns:
            velocity: Tensor of same shape as p
        """
        # Handle single frame case
        if p.shape[-3] == 1:
            return torch.zeros_like(p)

        # Central difference for interior, forward/backward for endpoints
        velocity = torch.zeros_like(p)
        velocity[..., 1:-1, :, :] = (p[..., 2:, :, :] -
                                     p[..., :-2, :, :]) / (2 * time_delta)
        velocity[...,
                 0, :, :] = (p[..., 1, :, :] - p[..., 0, :, :]) / time_delta
        velocity[...,
                 -1, :, :] = (p[..., -1, :, :] - p[..., -2, :, :]) / time_delta

        if gaussian_filter:
            velocity = ForwardKinematics._gaussian_smooth1d(velocity)
        return velocity

    @staticmethod
    def _compute_angular_velocity(r, time_delta: float, guassian_filter=True):
        """Compute angular velocity from rotations"""
        # Handle single frame case
        if r.shape[-3] == 1:
            # For single frame, return zero angular velocity
            return torch.zeros_like(r)

        diff_quat_data = torch.cat([
            quat_mul_norm(r[..., 1:, :, :],
                          quat_inverse(r[..., :-1, :, :], w_last=True),
                          w_last=True),
            quat_identity_like(r[..., :1, :, :]).to(r)
        ],
                                   dim=-3)
        diff_angle, diff_axis = quat_angle_axis(diff_quat_data, w_last=True)
        angular_velocity = diff_axis * diff_angle.unsqueeze(-1) / time_delta
        if guassian_filter:
            angular_velocity = ForwardKinematics._gaussian_smooth1d(
                angular_velocity)
        return angular_velocity

    def forward_kinematics(self,
                           joint_angles: torch.Tensor,
                           root_translation: torch.Tensor = None) -> dict:
        """
        Simplified forward kinematics interface.
        
        Args:
            joint_angles: Joint angles in axis-angle format [batch, seq_len, num_joints, 3]
            root_translation: Root translation [batch, seq_len, 3] (optional, defaults to zero)
            
        Returns:
            dict with global positions and rotations
        """
        if root_translation is None:
            root_translation = torch.zeros(joint_angles.shape[0],
                                           joint_angles.shape[1],
                                           3,
                                           device=self.device)

        result = self.fk_batch(joint_angles,
                               root_translation,
                               return_full=True)
        return result

    def get_body_names(self) -> list:
        """Get list of body names"""
        return self.body_names.copy()

    def get_num_bodies(self) -> int:
        """Get number of bodies in skeleton"""
        return self.num_bodies

    def get_foot_id(self) -> list:
        """Get foot id"""
        body_names = self.body_names_augment
        foot_names: list = self.cfg.foot_names
        return [body_names.index(foot_name) for foot_name in foot_names]
        # return bodyt_names.index('left_foot')

    def get_hand_id(self) -> list:
        """Return left/right hand IDs in the augmented body array."""
        hand_names = ("left_hand_link", "right_hand_link")
        return [self.body_names_augment.index(name) for name in hand_names]
