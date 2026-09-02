import sys
from pathlib import Path

import joblib
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataset.data_process.convert_soma_csv_to_motion_lib import (
    _contact_and_sliding_masks_from_foot_positions,
)
from dataset.data_process.pack_motion_lib_to_textop import (
    motion_lib_entry_to_textop,
)
from TextOpRobotMDAR.robotmdar.dataloader.data import SkeletonPrimitiveDataset
from TextOpRobotMDAR.robotmdar.utils.goal import GoalType
from TextOpRobotMDAR.robotmdar.train.manager import GeometryLoss


def _motion_item(length: int, marker: str = "motion"):
    return {
        "length": length,
        "motion": {"motion_len": length},
        "marker": marker,
    }


def _length_only_dataset(datadir: Path, *, weighted_sample: bool = False):
    dataset = SkeletonPrimitiveDataset.__new__(SkeletonPrimitiveDataset)
    dataset.datadir = datadir
    dataset.split = "train"
    dataset.segment_len = 10
    dataset.context_len = 10
    dataset.goal_offset = 0
    dataset.goal_offset_range = (0, 0)
    dataset.min_goal_offset = 0
    dataset.max_goal_offset = 0
    dataset.required_length = 10
    dataset.goal_type = GoalType.ROOT
    dataset.goal_per_primitive = False
    dataset.history_len = 3
    dataset.future_len = 7
    dataset.num_primitive = 1
    dataset.batch_size = 1
    dataset.weighted_sample = weighted_sample
    dataset._load_statistics = lambda: None
    return dataset


def test_dataset_filters_lengths_at_training_time_and_accepts_exact_window(tmp_path):
    joblib.dump(
        [_motion_item(9, "short"), _motion_item(10, "exact")],
        tmp_path / "train.pkl",
    )
    dataset = _length_only_dataset(tmp_path)
    dataset._load_data()
    dataset._generate_motion_primitives = lambda sample, start, goal_offset: [
        (sample["marker"], start)
    ]

    sampled = dataset._sample_motion_batch(
        generator=torch.Generator().manual_seed(0)
    )

    assert dataset.valid_indices == [1]
    assert sampled == [[("exact", 0)]]


def test_dataset_lazily_loads_manifest_sample(tmp_path):
    sample_dir = tmp_path / "samples"
    sample_dir.mkdir()
    joblib.dump(_motion_item(10, "lazy"), sample_dir / "00000000.pkl")
    joblib.dump(
        [{
            "length": 10,
            "frame_ann": [],
            "_data_path": "samples/00000000.pkl",
        }],
        tmp_path / "train.pkl",
    )
    dataset = _length_only_dataset(tmp_path)
    dataset._load_data()
    dataset._generate_motion_primitives = lambda sample, start, goal_offset: [
        (sample["marker"], start)
    ]

    sampled = dataset._sample_motion_batch(
        generator=torch.Generator().manual_seed(0)
    )

    assert sampled == [[("lazy", 0)]]
    assert "motion" not in dataset.raw_data[0]


def test_dataset_reports_active_window_when_every_clip_is_too_short(tmp_path):
    joblib.dump([_motion_item(9)], tmp_path / "train.pkl")
    dataset = _length_only_dataset(tmp_path)

    with pytest.raises(
        ValueError,
        match=r"required_length=10 .*history_len=3, future_len=7.*longest=9",
    ):
        dataset._load_data()


def test_weighted_sampling_indexes_only_valid_sequences():
    dataset = _length_only_dataset(Path("."), weighted_sample=True)
    dataset.raw_data = [_motion_item(5, "short"), _motion_item(10, "valid")]
    dataset.valid_indices = [1]
    dataset.seq_weights = np.array([1.0])
    dataset.frame_weight = False
    dataset._generate_motion_primitives = (
        lambda sample, start, goal_offset: [sample["marker"]]
    )

    sampled = dataset._sample_motion_batch()

    assert sampled == [["valid"]]


@pytest.mark.parametrize(
    ("goal_per_primitive", "expected_goal_frames"),
    [
        (True, [2, 6]),
        (False, [7, 7]),
    ],
)
def test_negative_goal_offset_is_bounded_for_both_goal_modes(
    goal_per_primitive, expected_goal_frames
):
    dataset = SkeletonPrimitiveDataset.__new__(SkeletonPrimitiveDataset)
    dataset.history_len = 2
    dataset.future_len = 4
    dataset.num_primitive = 2
    dataset.segment_len = 11
    dataset.goal_offset = -3
    dataset.goal_type = GoalType.BODY_EXT
    dataset.goal_per_primitive = goal_per_primitive
    dataset._world_goal_keypoints = lambda motion, frame: torch.zeros((4, 3))
    observed = []

    def extract(sample, prim_start, prim_end, goal_frame,
                world_goal_keypoints=None):
        observed.append((prim_start, prim_end, goal_frame))
        return {"goal_frame": goal_frame}

    dataset._extract_single_primitive = extract
    sample = {
        "motion": {
            "root_trans_offset": np.zeros((12, 3), dtype=np.float32),
        }
    }

    result = dataset._generate_motion_primitives(sample, 0)

    assert [item["goal_frame"] for item in result] == expected_goal_frames
    assert [item[2] for item in observed] == expected_goal_frames


def test_joint_state_goal_extraction_uses_direct_gt_frame_without_keypoints():
    dataset = SkeletonPrimitiveDataset.__new__(SkeletonPrimitiveDataset)
    dataset.history_len = 2
    dataset.goal_type = GoalType.JOINT_STATE
    dataset.goal_timestep_mode = "relative"
    dataset.fps = 50
    dataset._select_model_dof = lambda value: value
    dataset._primitive_action_label = lambda sample, start, end: "synthetic"

    def fail_keypoints(*args, **kwargs):
        raise AssertionError("joint_state goal must not extract FK keypoints")

    dataset._world_goal_keypoints = fail_keypoints

    length = 6
    root_pos = np.zeros((length, 3), dtype=np.float32)
    root_pos[:, 0] = np.arange(length, dtype=np.float32) * 0.1
    root_rot = np.tile(
        np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (length, 1))
    dof = np.arange(length * 29, dtype=np.float32).reshape(length, 29)
    contact = np.ones((length, 2), dtype=np.float32)
    sample = {
        "motion": {
            "root_trans_offset": root_pos,
            "root_rot": root_rot,
            "dof": dof,
            "contact_mask": contact,
        },
        "scene": {},
    }

    primitive = dataset._extract_single_primitive(
        sample, prim_start=0, prim_end=5, goal_frame=3)

    torch.testing.assert_close(
        primitive["world_goal_rot"], torch.from_numpy(root_rot[3]))
    torch.testing.assert_close(
        primitive["world_goal_dof"], torch.from_numpy(dof[3]))
    torch.testing.assert_close(
        primitive["world_goal_vel"], torch.tensor([5.0, 0.0, 0.0]))
    torch.testing.assert_close(
        primitive["time_to_arrival"], torch.tensor([0.04]))
    assert "world_goal_keypoints" not in primitive


def test_shared_extended_goal_rejects_out_of_bounds_goal_frame():
    dataset = SkeletonPrimitiveDataset.__new__(SkeletonPrimitiveDataset)
    dataset.history_len = 2
    dataset.future_len = 4
    dataset.num_primitive = 2
    dataset.segment_len = 11
    dataset.goal_offset = 1
    dataset.goal_type = GoalType.BODY_EXT
    dataset.goal_per_primitive = False
    dataset._world_goal_keypoints = lambda motion, frame: torch.zeros((4, 3))
    sample = {
        "motion": {
            "root_trans_offset": np.zeros((11, 3), dtype=np.float32),
        }
    }

    with pytest.raises(IndexError, match="exceeds motion bounds"):
        dataset._generate_motion_primitives(sample, 0)


def test_goal_timestep_mode_supports_zero_ablation_and_relative_time():
    dataset = SkeletonPrimitiveDataset.__new__(SkeletonPrimitiveDataset)
    dataset.fps = 50

    dataset.goal_timestep_mode = "zero"
    torch.testing.assert_close(
        dataset._time_to_arrival_seconds(reference_frame=10, goal_frame=60),
        torch.zeros(1),
    )

    dataset.goal_timestep_mode = "relative"
    torch.testing.assert_close(
        dataset._time_to_arrival_seconds(reference_frame=10, goal_frame=60),
        torch.ones(1),
    )


def test_bonesseed_contact_mask_rejects_low_sliding_feet():
    foot_pos = np.array(
        [
            [[0.00, 0.0, 0.04], [0.00, 0.0, 0.04]],
            [[0.01, 0.0, 0.04], [0.001, 0.0, 0.04]],
            [[0.011, 0.0, 0.04], [0.002, 0.0, 0.08]],
        ],
        dtype=np.float64,
    )

    contact, sliding = _contact_and_sliding_masks_from_foot_positions(
        foot_pos, fps=50, height_thresh=0.05, vel_thresh=0.15
    )

    np.testing.assert_array_equal(
        contact,
        np.array([[1.0, 1.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        sliding,
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]], dtype=np.float32),
    )


def test_bonesseed_contact_is_frame_rate_independent():
    def constant_velocity_motion(fps, velocity):
        time = np.arange(int(fps / 2), dtype=np.float64) / fps
        foot = np.zeros((len(time), 2, 3), dtype=np.float64)
        foot[:, :, 0] = time[:, None] * velocity
        foot[:, :, 2] = 0.04
        return foot

    for fps in (50, 120):
        contact, sliding = _contact_and_sliding_masks_from_foot_positions(
            constant_velocity_motion(fps, velocity=0.10), fps=fps
        )
        assert np.all(contact[1:] == 1)
        assert np.all(sliding == 0)

        contact, sliding = _contact_and_sliding_masks_from_foot_positions(
            constant_velocity_motion(fps, velocity=0.20), fps=fps
        )
        assert np.all(contact[1:] == 0)
        assert np.all(sliding[1:] == 1)


def test_foot_contact_loss_penalizes_only_stance_velocity():
    geometry_loss = GeometryLoss()
    geometry_loss.rec_criterion = torch.nn.HuberLoss()
    foot_positions = torch.tensor(
        [[
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.1, 0.0, 0.0], [0.1, 0.0, 0.0]],
        ]]
    )

    stance_loss = geometry_loss.calc_foot_sliding_loss(
        foot_positions, torch.ones((1, 2, 2)), fps=50.0
    )
    swing_loss = geometry_loss.calc_foot_sliding_loss(
        foot_positions, torch.zeros((1, 2, 2)), fps=50.0
    )
    excluded_loss = geometry_loss.calc_foot_sliding_loss(
        foot_positions,
        torch.ones((1, 2, 2)),
        fps=50.0,
        sliding_mask=torch.ones((1, 2, 2)),
    )

    assert stance_loss > 0
    torch.testing.assert_close(swing_loss, torch.tensor(0.0))
    torch.testing.assert_close(excluded_loss, torch.tensor(0.0))
    with pytest.raises(ValueError, match="fps must be positive"):
        geometry_loss.calc_foot_sliding_loss(
            foot_positions, torch.ones((1, 2, 2)), fps=None
        )

    ratio = geometry_loss.calc_sliding_ratio(
        torch.tensor([[1.0, 1.0, 1.0]]), torch.tensor([[1.0, 0.0, 0.0]])
    )
    torch.testing.assert_close(ratio, torch.tensor(0.25))


def test_crawl_prone_suppresses_sliding_mask():
    """Low + fast feet during crawl are intentional propulsion, not sliding."""
    T = 4
    foot_pos = np.zeros((T, 2, 3), dtype=np.float64)
    foot_pos[:, :, 2] = 0.04                      # feet low
    foot_pos[1, :, 0] = 0.0015                    # 0.0015m * 50fps = 0.075 m/s < 0.15 → contact
    foot_pos[2, :, 0] = 0.020                     # frame 1→2: 0.0185m * 50fps ≈ 0.925 m/s
    foot_pos[3, :, 0] = 0.045                     # frame 2→3: 0.025m  * 50fps = 1.25   m/s
    pelvis_z = np.array([0.25, 0.25, 0.25, 0.70], dtype=np.float64)

    contact, sliding = _contact_and_sliding_masks_from_foot_positions(
        foot_pos, fps=50, pelvis_z=pelvis_z,
    )

    # Frame 1: low, slow → contact, NOT sliding
    assert contact[1, 0] == 1.0
    assert sliding[1, 0] == 0.0

    # Frame 2: low, fast, pelvis=0.25 (crawl) → not contact, but also NOT sliding
    assert contact[2, 0] == 0.0
    assert sliding[2, 0] == 0.0

    # Frame 3: low, fast, pelvis=0.70 (standing) → sliding IS flagged
    assert contact[3, 0] == 0.0
    assert sliding[3, 0] == 1.0

    # Without pelvis_z: all fast frames are sliding (backward-compatible)
    contact2, sliding2 = _contact_and_sliding_masks_from_foot_positions(
        foot_pos, fps=50,
    )
    assert sliding2[2, 0] == 1.0   # was suppressed with pelvis_z


def test_motion_packer_and_reconstruction_surface_sliding_mask():
    sliding_mask = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    packed = motion_lib_entry_to_textop(
        "walk_test",
        {
            "root_trans_offset": np.zeros((2, 3), dtype=np.float32),
            "root_rot": np.tile(
                np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (2, 1)
            ),
            "dof": np.zeros((2, 29), dtype=np.float32),
            "contact_mask": np.zeros((2, 2), dtype=np.float32),
            "sliding_mask": sliding_mask,
            "fps": 50,
        },
    )
    np.testing.assert_array_equal(packed["motion"]["sliding_mask"], sliding_mask)

    class EchoSkeleton:
        @staticmethod
        def forward_kinematics(motion_dict, return_full=False, fps=30.0):
            return motion_dict

    dataset = SkeletonPrimitiveDataset.__new__(SkeletonPrimitiveDataset)
    dataset.skeleton = EchoSkeleton()
    dataset.fps = 50
    reconstructed = dataset.reconstruct_motion(
        torch.zeros((1, 2, 69)),
        need_denormalize=False,
        ret_fk=True,
        sliding_mask=torch.as_tensor(sliding_mask).unsqueeze(0),
    )
    torch.testing.assert_close(
        reconstructed["sliding_mask"], torch.as_tensor(sliding_mask).unsqueeze(0)
    )

    dataset.num_primitive = 1
    dataset.batch_size = 1
    dataset.goal_type = "root"
    dataset.normalize = lambda feature: feature
    dataset._convert_to_motion_features = lambda motion: torch.zeros((1, 3, 69))
    primitive = {
        "motion": {},
        "sliding_mask": torch.arange(8, dtype=torch.float32).reshape(4, 2),
        "scene": {},
        "world_goal_pos": torch.zeros(3),
        "world_goal_yaw": torch.tensor(0.0),
        "history_start_pos": torch.zeros(3),
        "history_start_rot": torch.zeros(4),
        "gt_ref_pos": torch.zeros(3),
        "gt_ref_rot": torch.zeros(4),
    }
    batch = dataset._organize_primitives_by_index([[primitive]])[0]
    assert batch["sliding_mask"].shape == (1, 3, 2)
    torch.testing.assert_close(
        batch["sliding_mask"][0], primitive["sliding_mask"][:3]
    )
