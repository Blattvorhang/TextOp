from robotmdar.utils.goal import (
    EXTENDED_BODY_GOAL_DIM,
    GoalType,
    JOINT_STATE_GOAL_DIM,
    JOINT_STATE_GOAL_DOF_DIM,
    build_ego_goal,
    build_ego_joint_state_goal,
    quaternion_yaw,
    validate_goal_config,
)
from robotmdar.utils.occupancy import (
    compute_scene_surface_batch,
    erode_voxel_26,
    query_local_occupancy,
)
