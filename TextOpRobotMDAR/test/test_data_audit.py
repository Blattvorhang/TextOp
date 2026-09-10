import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from TextOpRobotMDAR.robotmdar.dataloader.data import SkeletonPrimitiveDataset


def _audit_dataset(records):
    dataset = SkeletonPrimitiveDataset.__new__(SkeletonPrimitiveDataset)
    dataset.required_length = 81
    dataset.fps = 50.0
    dataset.audit_primitive_windows = 0
    dataset.raw_data = records
    dataset.valid_indices = list(range(len(records)))
    dataset.text_embeddings_dict = {'': None}
    return dataset._build_audit_stats()


def test_audit_reports_offline_augmentation_before_and_after():
    stats = _audit_dataset([
        {
            'length': 101,
            '_fps': 50,
            '_source': 'stand_up_lying_A',
            '_recovery_boost': True,
        },
        {
            'length': 201,
            '_fps': 50,
            '_source': 'walk_A',
            '_recovery_boost': False,
        },
        {
            'length': 101,
            '_fps': 50,
            '_source': 'stand_up_lying_A_aug_001',
            '_recovery_boost': True,
        },
        {
            'length': 201,
            '_fps': 50,
            '_source': 'walk_A_aug_001',
            '_recovery_boost': False,
        },
    ])

    offline = stats['offline_augmentation']
    assert offline['before']['sequences'] == 2.0
    assert offline['after']['sequences'] == 4.0
    assert offline['added']['sequences'] == 2.0
    assert offline['before']['windows'] == 142.0
    assert offline['after']['windows'] == 284.0
    assert offline['added']['windows'] == 142.0
    assert offline['added_sequence_fraction_of_after'] == 0.5
    assert offline['added_hour_fraction_of_after'] == 0.5

    recovery = stats['fall_recovery']
    assert recovery['before']['sequences'] == 1.0
    assert recovery['after']['sequences'] == 2.0
    assert recovery['added']['sequences'] == 1.0
    assert recovery['before']['windows'] == 21.0
    assert recovery['after']['windows'] == 42.0
    assert recovery['added']['windows'] == 21.0
    assert recovery['after_sequence_fraction_of_data'] == 0.5
    assert abs(recovery['after_hour_fraction_of_data'] - 202.0 / 604.0) < 1.0e-9


def test_offline_augmented_source_detection_matches_packed_suffix():
    assert SkeletonPrimitiveDataset._is_offline_augmented_record({
        '_source': 'aug_fall_recovery__stand_up_lying_A_aug_003',
    })
    assert SkeletonPrimitiveDataset._is_offline_augmented_record({
        '_source': 'stand_up_lying_A_aug_003.pkl',
    })
    assert not SkeletonPrimitiveDataset._is_offline_augmented_record({
        '_source': 'stand_up_lying_A',
    })
