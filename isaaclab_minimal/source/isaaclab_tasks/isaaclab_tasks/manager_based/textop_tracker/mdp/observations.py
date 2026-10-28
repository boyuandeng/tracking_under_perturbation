from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

from isaaclab_tasks.manager_based.textop_tracker.mdp import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def robot_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    mat = matrix_from_quat(command.robot_anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, :3].view(env.num_envs, -1)


def robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, 3:6].view(env.num_envs, -1)


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b_future(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Future N-step motion anchor position in body frame, flattened to vector."""
    command: MotionCommand = env.command_manager.get_term(command_name)

    # Reshape future data: (num_envs, future_steps, 3) and (num_envs, future_steps, 4)
    future_anchor_pos_w = command.motion_anchor_pos.view(env.num_envs, -1, 3)
    future_anchor_quat_w = command.motion_anchor_quat.view(env.num_envs, -1, 4)

    # Expand robot anchor for broadcasting: (num_envs, future_steps, 3) and (num_envs, future_steps, 4)
    robot_anchor_pos_w_exp = command.robot_anchor_pos_w[:, None, :].expand(-1, future_anchor_pos_w.shape[1], -1)
    robot_anchor_quat_w_exp = command.robot_anchor_quat_w[:, None, :].expand(-1, future_anchor_quat_w.shape[1], -1)

    # Transform all future steps at once
    pos_b, _ = subtract_frame_transforms(
        robot_anchor_pos_w_exp,
        robot_anchor_quat_w_exp,
        future_anchor_pos_w,
        future_anchor_quat_w,
    )
    return pos_b.view(env.num_envs, -1)


def motion_anchor_ori_b_future(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Future N-step motion anchor orientation in body frame, flattened to vector."""
    command: MotionCommand = env.command_manager.get_term(command_name)

    # Reshape future data: (num_envs, future_steps, 3) and (num_envs, future_steps, 4)
    future_anchor_pos_w = command.motion_anchor_pos.view(env.num_envs, -1, 3)
    future_anchor_quat_w = command.motion_anchor_quat.view(env.num_envs, -1, 4)

    # Expand robot anchor for broadcasting: (num_envs, future_steps, 3) and (num_envs, future_steps, 4)
    robot_anchor_pos_w_exp = command.robot_anchor_pos_w[:, None, :].expand(-1, future_anchor_pos_w.shape[1], -1)
    robot_anchor_quat_w_exp = command.robot_anchor_quat_w[:, None, :].expand(-1, future_anchor_quat_w.shape[1], -1)

    # Transform all future steps at once
    pos_b, ori_b = subtract_frame_transforms(
        robot_anchor_pos_w_exp,
        robot_anchor_quat_w_exp,
        future_anchor_pos_w,
        future_anchor_quat_w,
    )

    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)
