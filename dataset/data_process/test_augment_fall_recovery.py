import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from dataset.data_process import augment_fall_recovery as augment


class AugmentFallRecoveryTest(unittest.TestCase):
    def setUp(self):
        self.entry = {
            "dof": np.zeros((8, 29), dtype=np.float32),
            "root_trans_offset": np.zeros((8, 3), dtype=np.float32),
            "root_rot": np.tile([0, 0, 0, 1], (8, 1)).astype(np.float32),
            "contact_mask": np.zeros((8, 2), dtype=np.float32),
        }
        self.limits = np.tile([-3.0, 3.0], (29, 1)).astype(np.float32)

    def test_quintic_decay_and_prefix_only_weights(self):
        np.testing.assert_allclose(
            augment._window_weights(4),
            [1.0, 0.896484375, 0.5, 0.103515625, 0.0],
        )
        np.testing.assert_array_equal(
            augment._window_weights(4, prefix_only=True), np.ones(5)
        )

    def test_prefix_only_retains_all_candidates_and_counts_sonic_success(self):
        config = augment.AugmentConfig(
            accepted_per_original=3,
            max_candidates_per_original=3,
            n_lying_min=3,
            n_lying_max=3,
            collision_xml=Path("unused"),
            fk_backend="torch",
            torch_device="cpu",
            occ_hipc_root=Path("unused"),
            device="cpu",
            onnx_num_threads=1,
            validate=True,
            prefix_only=True,
            recompute_contact=False,
            quiet_policy_init=True,
        )

        class Validator:
            calls = 0

            def validate(self, entry):
                self.calls += 1
                return self.calls == 2, 0.78, 0.0

        with patch.object(augment, "_WORKER_CONFIG", config), \
             patch.object(augment, "_WORKER_JOINT_LIMITS", self.limits), \
             patch.object(augment, "_WORKER_VALIDATOR", Validator()):
            result = augment._augment_source_task(
                0, "stand_up_lying_test", self.entry, 123
            )

        self.assertEqual(result.candidates, 3)
        self.assertEqual(len(result.accepted_entries), 3)
        self.assertEqual(result.sonic_successes, 1)
        for _, candidate in result.accepted_entries:
            np.testing.assert_array_equal(candidate["dof"][3:], self.entry["dof"][3:])
            np.testing.assert_allclose(
                candidate["dof"][:3], np.tile(candidate["dof"][0], (3, 1))
            )


if __name__ == "__main__":
    unittest.main()
