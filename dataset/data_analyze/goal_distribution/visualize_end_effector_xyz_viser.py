#!/usr/bin/env python3
"""Interactively visualize cached end-effector points and the 29-DoF G1."""

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np
import viser


END_EFFECTOR_NAMES = ("left_hand", "right_hand", "left_foot", "right_foot")
DEFAULT_HAND_OFFSET = 0.08
DEFAULT_FOOT_OFFSET = 0.08
WAIST_BODY_NAME = "waist_yaw_link"
WAIST_Z_OFFSET = 0.044
POINT_COLORS = {
    "left_hand": (254, 167, 53),
    "right_hand": (0, 195, 255),
    "left_foot": (254, 114, 53),
    "right_foot": (0, 119, 255),
}
DEFAULT_POINTS = Path(__file__).resolve().parents[1] / "analysis_output/end_effector_xyz_64f.npz"
DEFAULT_STAND_POSE = (
    Path(__file__).resolve().parents[3]
    / "TextOpRobotMDAR/assets/ref_poses/stand.npz"
)
VITRUVIAN_ARM_JOINTS = {
    "left_shoulder_roll_joint": 1.00,
    "right_shoulder_roll_joint": -1.00,
    "left_elbow_joint": 1.323,
    "right_elbow_joint": 1.323,
}
DEFAULT_MJCF = (
    Path(__file__).resolve().parents[3]
    / "TextOpRobotMDAR/description/robots/g1/g1_29dof.xml"
)


def load_points(path):
    """Load end effectors and their matching shoulder/hip centers."""
    with np.load(path, allow_pickle=False) as data:
        required = set(END_EFFECTOR_NAMES) | {
            f"{name}_center" for name in END_EFFECTOR_NAMES
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(
                f"{path} is missing arrays: {sorted(missing)}; regenerate it with "
                "analyze_64f_end_effector_xyz.py"
            )
        removed_offset = (
            float(data["waist_z_offset_removed"])
            if "waist_z_offset_removed" in data.files
            else None
        )
        if removed_offset is None or not np.isclose(removed_offset, WAIST_Z_OFFSET):
            raise ValueError(
                f"{path} predates the waist offset correction; regenerate it with "
                "analyze_64f_end_effector_xyz.py"
            )
        points = {
            name: np.asarray(data[name], dtype=np.float32)
            for name in END_EFFECTOR_NAMES
        }
        centers = {
            name: np.asarray(data[f"{name}_center"], dtype=np.float32)
            for name in END_EFFECTOR_NAMES
        }
    for name in END_EFFECTOR_NAMES:
        xyz = points[name]
        center = centers[name]
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"{name} must have shape [N, 3], got {xyz.shape}")
        if center.shape != xyz.shape:
            raise ValueError(
                f"{name}_center must have shape {xyz.shape}, got {center.shape}"
            )
        if not np.isfinite(xyz).all() or not np.isfinite(center).all():
            raise ValueError(f"{name} or its center contains non-finite coordinates")
    return points, centers


def offset_from_centers(points, centers, distance):
    """Move each point radially away from its matching shoulder/hip center."""
    radial = points - centers
    lengths = np.linalg.norm(radial, axis=1, keepdims=True)
    directions = np.divide(
        radial,
        lengths,
        out=np.zeros_like(radial),
        where=lengths > 1e-8,
    )
    return points + np.float32(distance) * directions


def make_unit_icosphere():
    """Return a compact unit-radius icosphere mesh for point instances."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.asarray(
        [
            (-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
            (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
            (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1),
        ],
        dtype=np.float32,
    )
    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)
    faces = np.asarray(
        [
            (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
            (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
            (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
            (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
        ],
        dtype=np.uint32,
    )
    return vertices, faces


def apply_reference_pose(model, data, pose_path, joint_overrides=None):
    """Apply a 23-DoF MuJoCo-order reference pose to a 29-DoF model."""
    with np.load(pose_path, allow_pickle=False) as pose:
        if "joint_angles" not in pose.files:
            raise ValueError(f"{pose_path} has no joint_angles array")
        joint_angles = np.asarray(pose["joint_angles"], dtype=np.float32)
    actuator_joint_ids = model.actuator_trnid[:, 0].astype(int)
    actuator_joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        for joint_id in actuator_joint_ids
    ]
    core_joint_names = [
        name for name in actuator_joint_names if "_wrist_" not in name
    ]
    if joint_angles.shape != (len(core_joint_names),):
        raise ValueError(
            f"Expected {len(core_joint_names)} core joint angles in {pose_path}, "
            f"got {joint_angles.shape}"
        )
    pose_by_name = dict(zip(core_joint_names, joint_angles))
    if joint_overrides is not None:
        unknown = set(joint_overrides) - set(actuator_joint_names)
        if unknown:
            raise ValueError(f"Unknown pose override joints: {sorted(unknown)}")
        pose_by_name.update(joint_overrides)
    for name, angle in pose_by_name.items():
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        lower, upper = model.jnt_range[joint_id]
        if not lower <= angle <= upper:
            raise ValueError(
                f"Pose angle {angle} for {name} is outside [{lower}, {upper}]"
            )
        data.qpos[model.jnt_qposadr[joint_id]] = angle


def add_g1_meshes(server, mjcf_path, pose_path=None, joint_overrides=None):
    """Add colored 29-DoF G1 meshes, optionally using a reference pose."""
    model = mujoco.MjModel.from_xml_path(str(mjcf_path.resolve()))
    if model.nu != 29:
        raise ValueError(f"Expected a 29-DoF MJCF, but {mjcf_path} has {model.nu} actuators")
    waist_body = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, WAIST_BODY_NAME
    )
    waist_offset = model.body_pos[waist_body]
    if not np.isclose(waist_offset[2], WAIST_Z_OFFSET):
        raise ValueError(
            f"Expected {WAIST_BODY_NAME} Z offset {WAIST_Z_OFFSET}, "
            f"got {waist_offset[2]}"
        )
    model.body_pos[waist_body, 2] = 0.0
    data = mujoco.MjData(model)
    if pose_path is not None:
        apply_reference_pose(model, data, pose_path, joint_overrides)
    mujoco.mj_forward(model, data)

    root_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    root_position = data.xpos[root_body].copy()
    mesh_count = 0
    for geom_id in range(model.ngeom):
        if (
            model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH
            or model.geom_group[geom_id] != 1
        ):
            continue
        mesh_id = model.geom_dataid[geom_id]
        vertex_start = model.mesh_vertadr[mesh_id]
        vertex_end = vertex_start + model.mesh_vertnum[mesh_id]
        face_start = model.mesh_faceadr[mesh_id]
        face_end = face_start + model.mesh_facenum[mesh_id]
        vertices = np.asarray(model.mesh_vert[vertex_start:vertex_end], dtype=np.float32)
        faces = np.asarray(model.mesh_face[face_start:face_end], dtype=np.uint32)
        rotation = np.asarray(data.geom_xmat[geom_id], dtype=np.float32).reshape(3, 3)
        vertices = vertices @ rotation.T + data.geom_xpos[geom_id] - root_position
        color = tuple(np.clip(model.geom_rgba[geom_id, :3] * 255, 0, 255).astype(int))
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        server.scene.add_mesh_simple(
            f"/g1/{geom_name or f'geom_{geom_id}'}",
            vertices=vertices,
            faces=faces,
            color=color,
            side="double",
            cast_shadow=False,
            receive_shadow=False,
        )
        mesh_count += 1
    if mesh_count == 0:
        raise RuntimeError(f"No visual meshes found in {mjcf_path}")
    return mesh_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=Path, default=DEFAULT_POINTS)
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF)
    parser.add_argument("--stand-pose", type=Path, default=DEFAULT_STAND_POSE)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--density", type=int, default=25,
                        help="Initial point density percentage (default: 25)")
    parser.add_argument("--opacity", type=int, default=25,
                        help="Initial point opacity percentage (default: 25)")
    parser.add_argument(
        "--point-radius", "--point-size", dest="point_radius",
        type=float, default=0.005,
        help="Sphere radius in meters (default: 0.005)",
    )
    parser.add_argument(
        "--hand-offset",
        type=float,
        default=DEFAULT_HAND_OFFSET,
        help="Initial radial offset beyond wrists in meters (default: 0.08)",
    )
    parser.add_argument(
        "--foot-offset",
        type=float,
        default=DEFAULT_FOOT_OFFSET,
        help="Initial radial offset beyond ankles in meters (default: 0.08)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 1 <= args.density <= 100:
        raise ValueError(f"--density must be in [1, 100], got {args.density}")
    if not 0 <= args.opacity <= 100:
        raise ValueError(f"--opacity must be in [0, 100], got {args.opacity}")
    if args.point_radius <= 0:
        raise ValueError(
            f"--point-radius must be positive, got {args.point_radius}"
        )
    if not 0 <= args.hand_offset <= 0.25:
        raise ValueError(
            f"--hand-offset must be in [0, 0.25], got {args.hand_offset}"
        )
    if not 0 <= args.foot_offset <= 0.25:
        raise ValueError(
            f"--foot-offset must be in [0, 0.25], got {args.foot_offset}"
        )

    points, centers = load_points(args.points)
    rng = np.random.default_rng(args.seed)
    point_order = {name: rng.permutation(len(xyz)) for name, xyz in points.items()}
    sphere_vertices, sphere_faces = make_unit_icosphere()

    server = viser.ViserServer(host=args.host, port=args.port, label="G1 end effectors")
    server.scene.set_background_image(np.full((2, 2, 3), 255, dtype=np.uint8))
    server.scene.world_axes.visible = False
    server.scene.set_up_direction("+z")
    server.initial_camera.position = (3.0, 0.0, 0.0)
    server.initial_camera.look_at = (0.0, 0.0, 0.0)
    server.initial_camera.up_direction = (0.0, 0.0, 1.0)

    mesh_count = add_g1_meshes(
        server, args.mjcf, args.stand_pose, VITRUVIAN_ARM_JOINTS
    )
    density = server.gui.add_slider(
        "Point density (%)", min=1, max=100, step=1, initial_value=args.density
    )
    opacity = server.gui.add_slider(
        "Point opacity (%)", min=0, max=100, step=1, initial_value=args.opacity
    )
    hand_offset = server.gui.add_slider(
        "Hand offset (m)",
        min=0.0,
        max=0.25,
        step=0.005,
        initial_value=args.hand_offset,
    )
    foot_offset = server.gui.add_slider(
        "Foot offset (m)",
        min=0.0,
        max=0.25,
        step=0.005,
        initial_value=args.foot_offset,
    )
    point_count = server.gui.add_text("Points shown", "", disabled=True)

    def update_points(_) -> None:
        total = 0
        for name in END_EFFECTOR_NAMES:
            count = max(1, round(len(points[name]) * density.value / 100))
            sample_indices = point_order[name][:count]
            sampled = offset_from_centers(
                points[name][sample_indices],
                centers[name][sample_indices],
                hand_offset.value if name.endswith("_hand") else foot_offset.value,
            )
            orientations = np.zeros((count, 4), dtype=np.float32)
            orientations[:, 0] = 1.0
            server.scene.add_batched_meshes_simple(
                f"/end_effectors/{name}",
                vertices=sphere_vertices,
                faces=sphere_faces,
                batched_wxyzs=orientations,
                batched_positions=sampled,
                batched_scales=np.full(
                    count, args.point_radius, dtype=np.float32
                ),
                batched_colors=POINT_COLORS[name],
                opacity=opacity.value / 100.0,
                flat_shading=False,
                side="double",
                cast_shadow=False,
                receive_shadow=False,
            )
            total += count
        point_count.value = f"{total:,}"

    density.on_update(update_points)
    opacity.on_update(update_points)
    hand_offset.on_update(update_points)
    foot_offset.on_update(update_points)
    update_points(None)
    print(f"Loaded {sum(map(len, points.values())):,} points and {mesh_count} G1 meshes")
    print(f"Open http://localhost:{args.port}")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
