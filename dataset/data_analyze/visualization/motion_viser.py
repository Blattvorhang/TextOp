#!/usr/bin/env python3
"""Play G1 motion NPZs from explicit JSON/TXT file lists or --npz paths.

No directory traversal. List paths are relative to the list file; CLI/browser
paths are relative to the launch directory. Recorded body poses drive meshes.
"""

import argparse
from dataclasses import dataclass, field
from functools import lru_cache
import json
from pathlib import Path
import queue
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import viser


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FILE_LIST = Path(__file__).resolve().parents[1] / "analysis_output/hiphi_tpose/full_scan_endpoints_adaptive/review_10/motion_files.json"
DEFAULT_MJCF = ROOT / "TextOpRobotMDAR/description/robots/g1/g1_29dof.xml"
LINK_NAMES = (
    "pelvis", "left_hip_pitch_link", "right_hip_pitch_link", "waist_yaw_link",
    "left_hip_roll_link", "right_hip_roll_link", "waist_roll_link",
    "left_hip_yaw_link", "right_hip_yaw_link", "torso_link",
    "left_knee_link", "right_knee_link", "left_shoulder_pitch_link",
    "right_shoulder_pitch_link", "left_ankle_pitch_link", "right_ankle_pitch_link",
    "left_shoulder_roll_link", "right_shoulder_roll_link", "left_ankle_roll_link",
    "right_ankle_roll_link", "left_shoulder_yaw_link", "right_shoulder_yaw_link",
    "left_elbow_link", "right_elbow_link", "left_wrist_roll_link", "right_wrist_roll_link",
    "left_wrist_pitch_link", "right_wrist_pitch_link", "left_wrist_yaw_link", "right_wrist_yaw_link",
)
PARENTS = np.asarray([0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 10, 11,
                      12, 13, 14, 15, 16, 17, 20, 21, 22, 23, 24, 25, 26, 27])

@dataclass
class Motion:
    path: Path
    fps: float
    positions: np.ndarray
    quaternions: np.ndarray
    joints: np.ndarray | None
    start_frame: int = 0

    @property
    def frames(self):
        return len(self.positions)

    def crop(self, start, end):
        if not 0 <= start < end <= self.frames:
            raise ValueError(f"Invalid frame range [{start}, {end}) for {self.path}")
        return Motion(self.path, self.fps, self.positions[start:end], self.quaternions[start:end],
                      None if self.joints is None else self.joints[start:end], self.start_frame + start)


@lru_cache(maxsize=3)
def load_motion(path):
    with np.load(path, allow_pickle=False) as data:
        fps = float(data["fps"].item())
        positions = np.asarray(data["body_pos_w"], dtype=np.float32)
        quaternions = np.asarray(data["body_quat_w"], dtype=np.float32)
        joints = np.asarray(data["joint_pos"], dtype=np.float32) if "joint_pos" in data.files else None
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"{path}: invalid FPS {fps}")
    if positions.ndim != 3 or positions.shape[1:] != (30, 3) or len(positions) == 0:
        raise ValueError(f"{path}: expected nonempty body_pos_w [T,30,3], got {positions.shape}")
    if quaternions.shape != (len(positions), 30, 4):
        raise ValueError(f"{path}: invalid body_quat_w shape {quaternions.shape}")
    if not np.isfinite(positions).all() or not np.isfinite(quaternions).all():
        raise ValueError(f"{path}: non-finite body poses")
    lengths = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    if (lengths < 1e-6).any():
        raise ValueError(f"{path}: zero body quaternion")
    if joints is not None and (joints.shape != (len(positions), 29) or not np.isfinite(joints).all()):
        raise ValueError(f"{path}: invalid joint_pos")
    return Motion(path, fps, positions, quaternions / lengths, joints)



class G1Scene:
    def __init__(self, server, mjcf):
        self.server = server
        self.frames = [server.scene.add_frame(f"/robot/{name}", show_axes=False) for name in LINK_NAMES]
        model = mujoco.MjModel.from_xml_path(str(mjcf.resolve()))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        body_to_index = {}
        for index, name in enumerate(LINK_NAMES):
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body < 0:
                raise ValueError(f"{mjcf} is missing body {name}")
            body_to_index[body] = index
        mesh_count = 0
        for geom in range(model.ngeom):
            if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH or model.geom_group[geom] != 1:
                continue
            body = int(model.geom_bodyid[geom])
            while body and body not in body_to_index:
                body = int(model.body_parentid[body])
            if body not in body_to_index:
                continue
            # MuJoCo's compiled vertices include mesh frame corrections. Bring
            # them into the recorded ancestor link frame, including fixed hands.
            mesh = model.geom_dataid[geom]
            v0, nv = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            f0, nf = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
            vertices = model.mesh_vert[v0:v0 + nv]
            anchor_rotation = data.xmat[body].reshape(3, 3)
            relative_rotation = anchor_rotation.T @ data.geom_xmat[geom].reshape(3, 3)
            relative_position = anchor_rotation.T @ (data.geom_xpos[geom] - data.xpos[body])
            vertices = vertices @ relative_rotation.T + relative_position
            name = LINK_NAMES[body_to_index[body]]
            color = tuple(np.rint(model.geom_rgba[geom, :3] * 255).astype(np.uint8))
            server.scene.add_mesh_simple(
                f"/robot/{name}/geom_{geom}", vertices=vertices.astype(np.float32),
                faces=model.mesh_face[f0:f0 + nf].astype(np.uint32), color=color,
                cast_shadow=True, receive_shadow=False,
            )
            mesh_count += 1
        if not mesh_count:
            raise ValueError(f"No visual meshes found in {mjcf}")
        self.skeleton = server.scene.add_line_segments(
            "/skeleton", points=np.zeros((29, 2, 3), dtype=np.float32),
            colors=(235, 90, 40), thickness=3., thickness_units="screen", visible=False,
        )
        self.points = server.scene.add_point_cloud(
            "/joints", points=np.zeros((30, 3), dtype=np.float32),
            colors=(235, 90, 40), point_size=.015, visible=False,
        )
        self.mesh_count = mesh_count

    def update(self, motion, frame, offset, meshes, skeleton):
        positions = motion.positions[frame] - offset
        with self.server.atomic():
            for index, handle in enumerate(self.frames):
                handle.position = positions[index]
                handle.wxyz = motion.quaternions[frame, index]
                handle.visible = meshes
            self.skeleton.points = np.stack((positions[PARENTS[1:]], positions[1:]), axis=1)
            self.points.points = positions
            self.skeleton.visible = skeleton
            self.points.visible = skeleton


FULL = "完整 NPZ"


@dataclass
class MotionView:
    path: Path
    start: int = 0
    end: int | None = None
    source_start: int = 0

    def load(self):
        motion = load_motion(self.path)
        motion = motion.crop(self.start, motion.frames if self.end is None else self.end)
        motion.start_frame += self.source_start
        return motion


@dataclass
class MotionEntry:
    path: Path
    label: str = ""
    source: str = ""
    notes: str = ""
    annotations: list = field(default_factory=list)
    views: dict = field(default_factory=dict)

    def __post_init__(self):
        self.path = self.path.expanduser().resolve()
        if not self.path.is_file() or self.path.suffix.lower() != ".npz":
            raise ValueError(f"NPZ not found: {self.path}")
        self.label = self.label or self.path.name
        if not self.views:
            self.views = {FULL: MotionView(self.path)}
        for annotation in self.annotations:
            start, end = annotation["frames"]
            if (not isinstance(start, int) or not isinstance(end, int)
                    or not 0 <= start < end or not annotation.get("label")):
                raise ValueError(f"Invalid annotation in {self.path}: {annotation}")


def read_file_list(path):
    """Read only explicitly listed paths, relative to the list file."""
    path = path.expanduser().resolve()
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload["motions"] if isinstance(payload, dict) else payload
    else:
        rows = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")]
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Empty or invalid motion list: {path}")
    entries = []
    for row in rows:
        record = {"path": row} if isinstance(row, str) else row
        target = Path(record["path"]).expanduser()
        if not target.is_absolute():
            target = path.parent / target
        entries.append(MotionEntry(target, label=record.get("label", ""),
                                   source=record.get("source", ""), notes=record.get("notes", ""),
                                   annotations=record.get("annotations", [])))
    return entries


class MotionViewer:
    def __init__(self, server, entries, mjcf):
        if not entries:
            raise ValueError("At least one motion is required")
        self.server, self.entries = server, list(entries)
        self.labels = [f"{i + 1:02d} · {entry.label}" for i, entry in enumerate(entries)]
        self.events = queue.SimpleQueue()
        self.index, self.position, self.rendered_frame = 0, 0., -1
        self.motion = None
        self.offset = np.zeros(3)
        server.scene.set_up_direction("+z")
        server.scene.world_axes.visible = False
        server.scene.add_grid("/floor", width=8., height=8., plane="xy", cell_size=.25,
                              section_size=1., plane_color=(245, 247, 250), plane_opacity=.8)
        server.initial_camera.position = (2.7, -2.7, 1.6)
        server.initial_camera.look_at = (0., 0., .75)
        server.initial_camera.up_direction = (0., 0., 1.)
        self.scene = G1Scene(server, mjcf)
        with server.gui.add_folder("动作选择"):
            self.file_gui = server.gui.add_dropdown("NPZ", options=self.labels)
            self.previous = server.gui.add_button("上一条")
            self.next = server.gui.add_button("下一条")
            self.path_gui = server.gui.add_text("文件路径", initial_value=str(entries[0].path))
            self.open_path = server.gui.add_button("打开指定 NPZ")
            self.mode = server.gui.add_dropdown("播放范围", options=list(entries[0].views))
            self.info = server.gui.add_markdown("")
        with server.gui.add_folder("播放控制"):
            self.play = server.gui.add_checkbox("播放", initial_value=True)
            self.loop = server.gui.add_checkbox("循环", initial_value=True)
            self.speed = server.gui.add_dropdown("速度", options=["0.25×", "0.5×", "1×", "2×"], initial_value="1×")
            self.frame_gui = server.gui.add_slider("帧", min=0, max=1, step=1, initial_value=0)
            self.back = server.gui.add_button("前一帧")
            self.forward = server.gui.add_button("后一帧")
            self.seek_seconds = server.gui.add_number("跳转到秒", initial_value=0., min=0., step=.1)
            self.seek_button = server.gui.add_button("跳转")
            self.time_gui = server.gui.add_text("时间", initial_value="", disabled=True)
            self.phase_gui = server.gui.add_text("当前标记", initial_value="", disabled=True)
        with server.gui.add_folder("区间标记"):
            self.show_annotations = server.gui.add_checkbox("显示标记", initial_value=True)
            self.annotation_gui = server.gui.add_dropdown("标记区间", options=["无标记"])
            self.jump_start = server.gui.add_button("区间起点")
            self.jump_end = server.gui.add_button("区间末帧")
            self.annotation_info = server.gui.add_markdown("")
        with server.gui.add_folder("视图"):
            self.meshes = server.gui.add_checkbox("机器人网格", initial_value=True)
            self.skeleton = server.gui.add_checkbox("骨架", initial_value=False)
            self.front = server.gui.add_button("正面")
            self.side = server.gui.add_button("侧面")
            self.reset = server.gui.add_button("斜视角")
        for handle, action in [(self.file_gui, "file"), (self.mode, "mode"), (self.frame_gui, "seek"),
                               (self.meshes, "view"), (self.skeleton, "view"),
                               (self.show_annotations, "annotations")]:
            handle.on_update(self._input_callback(action))
        for handle, action in [(self.previous, "previous"), (self.next, "next"),
                               (self.back, "back"), (self.forward, "forward"),
                               (self.front, "front"), (self.side, "side"), (self.reset, "reset"),
                               (self.open_path, "open"), (self.seek_button, "seconds"),
                               (self.jump_start, "interval_start"), (self.jump_end, "interval_end")]:
            handle.on_click(self._input_callback(action))
        self.load_selected()

    def _input_callback(self, action):
        def callback(event):
            # Ignore callbacks triggered by server-side updates.
            if event.client_id is not None:
                self.events.put((action, getattr(event.target, "value", None), event.client_id))
        return callback

    def load_selected(self):
        entry = self.entries[self.index]
        options = list(entry.views)
        selected = self.mode.value if self.mode.value in options else options[0]
        motion = entry.views[selected].load()
        for annotation in entry.annotations:
            if annotation["frames"][1] > motion.start_frame + motion.frames and len(entry.views) == 1:
                raise ValueError(f"Annotation outside motion: {annotation}")
        self.motion = motion
        self.position, self.rendered_frame = 0., -1
        self.offset = np.array([*motion.positions[0, 0, :2], 0.], dtype=np.float32)
        with self.server.atomic():
            self.mode.options = options
            self.mode.value = selected
            self.mode.disabled = len(options) == 1
            self.file_gui.value = self.labels[self.index]
            self.path_gui.value = str(entry.path)
            self.frame_gui.value = 0
            self.frame_gui.max = max(1, motion.frames - 1)
            self.frame_gui.disabled = motion.frames == 1
            self.seek_seconds.value = 0.
            self.seek_seconds.max = (motion.frames - 1) / motion.fps
            self.info.content = (f"**{self.index + 1} / {len(self.entries)}** · {motion.fps:g} FPS · "
                                 f"{motion.frames} 帧 · {motion.frames / motion.fps:.2f} 秒\n\n"
                                 f"播放：{selected}\n\n文件：`{entry.path}`"
                                 + (f"\n\n来源：`{entry.source}`" if entry.source else ""))
            self.annotation_labels = [f"{i + 1:02d} · {a['label']} · "
                                      f"{a['frames'][0] / motion.fps:.2f}–{a['frames'][1] / motion.fps:.2f} s"
                                      for i, a in enumerate(entry.annotations)]
            self.annotation_gui.options = self.annotation_labels or ["无标记"]
            self.annotation_gui.value = self.annotation_gui.options[0]
        self.update_annotations()
        self.render()

    def update_annotations(self):
        entry = self.entries[self.index]
        visible = self.show_annotations.value
        self.annotation_gui.visible = visible
        self.jump_start.visible = self.jump_end.visible = visible
        self.jump_start.disabled = self.jump_end.disabled = not bool(entry.annotations)
        self.annotation_gui.disabled = not bool(entry.annotations)
        self.annotation_info.visible = visible
        rows = [entry.notes] if entry.notes else []
        for annotation in entry.annotations:
            a, b = annotation["frames"]
            rows.append(f"- **{annotation['label']}**：{a / self.motion.fps:.2f}–{b / self.motion.fps:.2f} s，帧 [{a}, {b})")
        self.annotation_info.content = "\n\n".join(rows) or "此文件没有提供区间标记。"
        self.rendered_frame = -1

    def handle(self, action, value=None, client_id=None):
        if action == "open":
            target = Path(self.path_gui.value.strip()).expanduser().resolve()
            entry = next((e for e in self.entries if e.path == target), None)
            if entry is None:
                entry = MotionEntry(target)
                entry.views[FULL].load()
                self.entries.append(entry)
                self.labels.append(f"{len(self.entries):02d} · {entry.label}")
                self.file_gui.options = self.labels
            self.index = self.entries.index(entry)
            self.load_selected()
        elif action in ("previous", "next", "file", "mode"):
            if action in ("previous", "next"):
                self.index = (self.index + (-1 if action == "previous" else 1)) % len(self.entries)
            elif action == "file":
                self.index = self.labels.index(value)
            elif action == "mode":
                self.mode.value = value
            self.load_selected()
        elif action in ("seek", "back", "forward", "seconds", "interval_start", "interval_end"):
            self.play.value = False
            if action in ("interval_start", "interval_end"):
                if not self.annotation_labels:
                    return
                annotation = self.entries[self.index].annotations[self.annotation_labels.index(self.annotation_gui.value)]
                target = annotation["frames"][0] if action == "interval_start" else annotation["frames"][1] - 1
                target -= self.motion.start_frame
            elif action == "seconds":
                target = round(self.seek_seconds.value * self.motion.fps)
            else:
                target = value if action == "seek" else int(self.position) + (-1 if action == "back" else 1)
            self.position = float(np.clip(target, 0, self.motion.frames - 1))
            self.render()
        elif action in ("view", "annotations"):
            if action == "annotations":
                self.update_annotations()
            self.rendered_frame = -1
            self.render()
        elif action in ("front", "side", "reset"):
            positions = {"front": (3., 0., 1.1), "side": (0., -3., 1.1), "reset": (2.7, -2.7, 1.6)}
            clients = self.server.get_clients()
            targets = [clients[client_id]] if client_id in clients else list(clients.values())
            for client in targets:
                with client.atomic():
                    client.camera.position = positions[action]
                    client.camera.look_at = (0., 0., .75)
                    client.camera.up_direction = (0., 0., 1.)

    def render(self):
        frame = min(int(self.position), self.motion.frames - 1)
        if frame == self.rendered_frame:
            return
        self.scene.update(self.motion, frame, self.offset, self.meshes.value, self.skeleton.value)
        source_frame = self.motion.start_frame + frame
        with self.server.atomic():
            self.frame_gui.value = frame
            self.time_gui.value = (f"{frame / self.motion.fps:.2f} / {self.motion.frames / self.motion.fps:.2f} s"
                                   f" | 原始 {source_frame / self.motion.fps:.2f} s，帧 {source_frame}")
            active = [a["label"] for a in self.entries[self.index].annotations
                      if a["frames"][0] <= source_frame < a["frames"][1]]
            self.phase_gui.value = " / ".join(active) or "未标记"
            self.phase_gui.visible = self.show_annotations.value
        self.rendered_frame = frame

    def tick(self, elapsed):
        changed_motion = False
        while not self.events.empty():
            action, value, client_id = self.events.get_nowait()
            try:
                self.handle(action, value, client_id)
            except (OSError, ValueError, KeyError, IndexError) as error:
                self.play.value = False
                self.info.content = f"加载失败：{error}"
            changed_motion |= action in ("previous", "next", "file", "mode", "open", "seek", "back", "forward",
                                         "seconds", "interval_start", "interval_end")
        if self.play.value and not changed_motion:
            speed = float(self.speed.value.removesuffix("×"))
            self.position += elapsed * self.motion.fps * speed
            if self.position >= self.motion.frames:
                if self.loop.value:
                    self.position %= self.motion.frames
                else:
                    self.position = float(self.motion.frames - 1)
                    self.play.value = False
            self.render()


def run_viewer(entries, mjcf, host="127.0.0.1", port=8084, label="Motion Viser"):
    server = viser.ViserServer(host=host, port=port, label=label)
    try:
        viewer = MotionViewer(server, entries, mjcf)
        print(f"Loaded {len(entries)} NPZs and {viewer.scene.mesh_count} G1 visual meshes", flush=True)
        print(f"Open http://localhost:{server.get_port()}", flush=True)
        previous = time.monotonic()
        while True:
            now = time.monotonic()
            viewer.tick(now - previous)
            previous = now
            time.sleep(.005)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--file-list", type=Path, help="JSON/TXT with explicit NPZ paths; no directory scan")
    inputs.add_argument("--npz", type=Path, nargs="+", help="One or more explicit NPZ paths")
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8084)
    args = parser.parse_args()
    try:
        entries = ([MotionEntry(path) for path in args.npz] if args.npz
                   else read_file_list(args.file_list or DEFAULT_FILE_LIST))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    run_viewer(entries, args.mjcf, args.host, args.port)


if __name__ == "__main__":
    main()


