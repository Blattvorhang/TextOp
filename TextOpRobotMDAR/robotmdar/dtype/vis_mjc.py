import atexit
from dataclasses import dataclass
from typing import Callable, Literal, Optional
import os
import numpy as np
import mujoco
import mujoco.viewer
import time


def mjc_load_everything(
    dt: float,
    keycb_fn: Optional[Callable[[int], None]] = None,
    humanoid_xml:
    str = "./description/robots/g1/g1_29dof.xml"):

    mj_model = mujoco.MjModel.from_xml_path(humanoid_xml)  # type: ignore
    mj_data = mujoco.MjData(mj_model)  # type: ignore
    mj_model.opt.timestep = dt

    viewer = mujoco.viewer.launch_passive(mj_model,
                                          mj_data,
                                          key_callback=keycb_fn)

    viewer.cam.lookat[:] = np.array([0, 0, 0.7])
    viewer.cam.distance = 3.0
    viewer.cam.azimuth = 180
    viewer.cam.elevation = -30  # 负值表示从上往下看viewer

    def show_fn(qpos: np.ndarray, contact: np.ndarray):
        mj_data.qpos[:] = qpos
        mj_data.qpos[3:7] = (mj_data.qpos[3:7])[[3, 0, 1, 2]]  # xyzw -> wxyz

        mujoco.mj_forward(mj_model, mj_data)  # type: ignore
        viewer.sync()
        ...

    atexit.register(viewer.close)
    return show_fn, viewer


@dataclass
class VisState:
    midx: int = 0
    pidx: int = 0
    fidx: int = 0
    mode: Literal['pd', 'gt'] = 'pd'
    _trig: bool = True
    _autoplay: bool = True


def get_keycb_fn(vs: VisState):

    def keycb_fn(keycode: int):
        # p: toggle pd/gt
        # n/m: next/previous batch
        # j/k: next/previous primitive
        # left/right: next/previous frame
        # R: reset to fidx=0,pidx=0
        # space: toggle autoplay
        # q: quit
        if chr(keycode) == 'P':
            if vs.mode == 'pd':
                vs.mode = 'gt'
            else:
                vs.mode = 'pd'
            print(f"Mode: {vs.mode}")
        elif chr(keycode) == 'N':
            vs.midx += 1
            print(f"Next batch: {vs.midx}")
        elif chr(keycode) == 'M':
            vs.midx -= 1
            print(f"Previous batch: {vs.midx}")
        elif chr(keycode) == 'J':
            vs.pidx += 1
            print(f"Next primitive: {vs.pidx}")
        elif chr(keycode) == 'K':
            vs.pidx -= 1
            print(f"Previous primitive: {vs.pidx}")
        elif keycode == 263:  # Left arrow
            vs.fidx -= 1
            print(f"Previous frame: {vs.fidx}")
        elif keycode == 262:  # Right arrow
            vs.fidx += 1
            print(f"Next frame: {vs.fidx}")
        elif chr(keycode) == 'R':
            vs.fidx = 0
            vs.pidx = 0
            print(f"Reset to fidx=0, pidx=0")
        elif chr(keycode) == ' ':
            vs._autoplay = not vs._autoplay
            print(f"Autoplay: {vs._autoplay}")
        elif keycode == 256 or chr(keycode) == 'Q':
            print("Esc")
            os._exit(0)
        else:
            print(
                f"Unknown key: {keycode} ({chr(keycode) if keycode < 128 else 'special'})"
            )

        vs._trig = True

    return keycb_fn


def mjc_autoloop_mdar(vs: VisState, fps: float, num_primitive: int,
                      future_len: int, history_len: int, motion_buff: dict,
                      add_batch: Callable, keycb_fn: Callable):
    show_fn, viewer = mjc_load_everything(dt=1 / fps, keycb_fn=keycb_fn)

    while viewer.is_running():
        if vs._trig:
            batch_index = vs.midx // num_primitive
            batch_offset = vs.midx % num_primitive
            if batch_index >= len(motion_buff[vs.mode]):
                add_batch()
            if vs.pidx >= num_primitive:
                vs.fidx = 0
                vs.pidx = 0
                if not vs._autoplay:
                    vs.midx += 1
                continue
            elif vs.pidx < 0:
                vs.fidx = 0
                vs.pidx = num_primitive - 1
                if not vs._autoplay:
                    vs.midx -= 1
                continue

            if vs.fidx >= future_len + history_len:
                vs.fidx = history_len
                vs.pidx += 1
                continue
            elif vs.fidx < 0:
                vs.fidx = future_len - 1
                vs.pidx -= 1
                continue

            motion_prim = motion_buff[vs.mode][batch_index][vs.pidx]
            qpos, contact = motion_prim
            qpos_curr = qpos[batch_offset][vs.fidx]
            contact_curr = contact[batch_offset][vs.fidx]
            show_fn(qpos_curr, contact_curr)

            if not vs._autoplay:
                vs._trig = False
            else:
                vs.fidx += 1
            print(vs)
        time.sleep(1 / fps)
