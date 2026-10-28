from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from typing import Callable

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.envs.mdp.terminations import joint_pos_out_of_limit, joint_vel_out_of_limit, joint_vel_out_of_manual_limit

from isaaclab_tasks.manager_based.textop_tracker.mdp import MotionCommand
from isaaclab_tasks.manager_based.textop_tracker.mdp.rewards import _get_body_indexes
from isaaclab_tasks.manager_based.textop_tracker.mdp.rewards import recovery_gate

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:

    @torch.jit.script
    def quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        """Apply an inverse quaternion rotation to a vector.

        Args:
            quat: The quaternion in (w, x, y, z). Shape is (..., 4).
            vec: The vector in (x, y, z). Shape is (..., 3).

        Returns:
            The rotated vector in (x, y, z). Shape is (..., 3).
        """
        # store shape
        shape = vec.shape
        # reshape to (N, 3) for multiplication
        quat = quat.reshape(-1, 4)
        vec = vec.reshape(-1, 3)
        # extract components from quaternions
        xyz = quat[:, 1:]
        t = xyz.cross(vec, dim=-1) * 2
        return (vec - quat[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)


def termination_cond_on_pfail(term_fn: Callable) -> Callable:
    def wrapper(env: ManagerBasedRLEnv, pfail_threshold: float, *args) -> torch.Tensor:
        termination = term_fn(env, *args)
        pfail_total = env.command_manager.get_term("motion").metrics["pfail_total"]
        termination = termination.logical_and(pfail_total < pfail_threshold)

        return termination

    return wrapper


class TolerantTrackingTermination(ManagerTermBase):
    """Delay tracking termination while the continuous recovery gate is low."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.bad_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def __call__(self, env: ManagerBasedRLEnv, command_name: str = "motion",
                 recovery_seconds: float = 3.0, recovery_sigma: float = 0.25) -> torch.Tensor:
        bad = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        bad |= bad_anchor_pos_z_only(env, command_name, threshold=0.25)
        bad |= bad_anchor_ori(env, SceneEntityCfg("robot"), command_name, threshold=0.8)
        bad |= bad_motion_body_pos_z_only(
            env,
            command_name,
            threshold=0.25,
            body_names=[
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            ],
        )
        gate = recovery_gate(env, command_name, recovery_sigma)
        recovery = gate < 0.5
        limit = max(1, int(round(recovery_seconds / env.step_dt)))
        self.bad_steps = torch.where(bad, self.bad_steps + 1, torch.zeros_like(self.bad_steps))
        return torch.where(recovery, self.bad_steps >= limit, bad)

    def reset(self, env_ids=None):
        if self.bad_steps is not None:
            self.bad_steps[env_ids if env_ids is not None else slice(None)] = 0


def joint_pos_out_of_limit_cond_on_pfail(
    env: ManagerBasedRLEnv, pfail_threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    return termination_cond_on_pfail(joint_pos_out_of_limit)(env, pfail_threshold, asset_cfg)


def joint_vel_out_of_limit_cond_on_pfail(
    env: ManagerBasedRLEnv, pfail_threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    return termination_cond_on_pfail(joint_vel_out_of_limit)(env, pfail_threshold, asset_cfg)


def joint_vel_out_of_manual_limit_cond_on_pfail(
    env: ManagerBasedRLEnv,
    pfail_threshold: float,
    max_velocity: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    return termination_cond_on_pfail(joint_vel_out_of_manual_limit)(env, pfail_threshold, max_velocity, asset_cfg)


###########################


def joint_effort_out_of_limit_fixed(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when effort applied on the asset's joints are outside of the soft joint limits.

    In the actuators, the applied torque are the efforts applied on the joints. These are computed by clipping
    the computed torques to the joint limits. Hence, we check if the computed torques are equal to the applied
    torques.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # check if any joint effort is out of limit
    out_of_limits = torch.logical_not(
        torch.isclose(
            asset.data.computed_torque[:, asset_cfg.joint_ids], asset.data.applied_torque[:, asset_cfg.joint_ids]
        )
    )
    return torch.any(out_of_limits, dim=1)


def joint_effort_out_of_limit_fixed_cond_on_pfail(
    env: ManagerBasedRLEnv, pfail_threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    return termination_cond_on_pfail(joint_effort_out_of_limit_fixed)(env, pfail_threshold, asset_cfg)


def bad_anchor_pos(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold


def bad_anchor_pos_z_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold


def bad_anchor_ori(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    command: MotionCommand = env.command_manager.get_term(command_name)
    motion_projected_gravity_b = quat_apply_inverse(command.anchor_quat_w, asset.data.GRAVITY_VEC_W)

    robot_projected_gravity_b = quat_apply_inverse(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)
    # GRAVITY_VEC_W: [0,0,-9.81]

    return (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold


def bad_motion_body_pos(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.norm(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes], dim=-1)
    return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.abs(command.body_pos_relative_w[:, body_indexes, -1] - command.robot_body_pos_w[:, body_indexes, -1])
    return torch.any(error > threshold, dim=-1)
