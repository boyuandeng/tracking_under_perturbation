from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.textop_tracker.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from isaaclab_tasks.manager_based.textop_tracker.tracking_env_cfg import (
    ProjGravObservationsCfg,
    TrackingEnvCfg,
)

G1_BODY_NAMES = [
    "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
]


@configclass
class G1FlatEnvCfg(TrackingEnvCfg):
    """ManagerBased G1 tracking task using the supplied custom USD."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.body_names = G1_BODY_NAMES


@configclass
class G1FlatProjGravObsEnvCfg(G1FlatEnvCfg):
    observations: ProjGravObservationsCfg = ProjGravObservationsCfg()
