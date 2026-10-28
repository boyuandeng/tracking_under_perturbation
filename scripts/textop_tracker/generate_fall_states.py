#!/usr/bin/env python3
"""Generate a reusable bank of physically simulated fallen G1 states.

The motion archive is read only.  For every sample this program picks a random
motion frame, places the robot above the local plane, disables the implicit
actuators (stiffness and damping are both zero), and lets gravity and contact
physics run for a random 3--4 seconds.  Only the final pose is written to the
local bank.  The training reset code can therefore sample a fall without
starting another simulator or decompressing a motion file.

Run this from the TextOp container, for example::

  /isaac-sim/python.sh scripts/textop_tracker/generate_fall_states.py \
      --input /workspace/textop_npz \
      --output /workspace/stablemimic/data/textop_fall_states/fall_states_16384.npz \
      --count 16384 --num-envs 512 --device cuda:0

The source ``/workspace/textop_npz`` is expected to be mounted read-only.  The
output path must be on a writable local volume.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import mmap
from pathlib import Path
import struct
from typing import Any
import zipfile

import numpy as np


JOINT_NAMES_29 = np.asarray([
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint", "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint", "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint", "left_ankle_roll_joint",
    "right_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint", "left_wrist_roll_joint",
    "right_wrist_roll_joint", "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
])


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input", required=True, help="Read-only root containing motion.npz files")
    p.add_argument("--output", required=True, help="Writable output .npz bank")
    p.add_argument("--count", type=int, default=16384, help="Number of final states")
    p.add_argument("--num-envs", type=int, default=512, help="Physics environments per batch")
    p.add_argument("--duration-min", type=float, default=3.0, help="Minimum passive fall duration (seconds)")
    p.add_argument("--duration-max", type=float, default=4.0, help="Maximum passive fall duration (seconds)")
    p.add_argument("--physics-dt", type=float, default=0.005, help="PhysX timestep")
    p.add_argument("--seed", type=int, default=20261026)
    p.add_argument("--max-source-files", type=int, default=None)
    p.add_argument("--manifest", default=None, help="Optional JSON list of source files, avoiding a recursive remote directory scan")
    p.add_argument("--max-attempt-factor", type=int, default=5, help="Reject invalid/upright states and generate replacements")
    return p


def _source_files(root: Path, limit: int | None) -> list[Path]:
    if root.is_file():
        paths = [root]
    else:
        paths = sorted(root.rglob("*.npz"))
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"no .npz motion files under {root}")
    return paths


@contextmanager
def _motion_arrays(path: Path):
    """Read a stored NPZ through a read-only mmap instead of loading the clip."""
    keys = ("joint_pos", "body_pos_w", "body_quat_w")
    with path.open("rb") as handle, zipfile.ZipFile(handle) as archive:
        infos = {key: archive.getinfo(key + ".npy") for key in keys}
        if any(info.compress_type != zipfile.ZIP_STORED for info in infos.values()):
            with np.load(path, allow_pickle=False) as z:
                yield {key: np.asarray(z[key]) for key in keys}, (
                    np.asarray(z["joint_names"]).astype(str).tolist() if "joint_names" in z else JOINT_NAMES_29.tolist()
                )
            return
        mapping = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        arrays = {}
        try:
            for key, info in infos.items():
                handle.seek(info.header_offset)
                local = struct.unpack("<4s5H3L2H", handle.read(30))
                if local[0] != b"PK\x03\x04":
                    raise ValueError(f"{path}: invalid ZIP local header")
                member_start = info.header_offset + 30 + local[-2] + local[-1]
                handle.seek(member_start)
                version = np.lib.format.read_magic(handle)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
                elif version == (2, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
                else:
                    raise ValueError(f"{path}: unsupported NPY format {version}")
                if dtype.hasobject:
                    raise ValueError(f"{path}: object array {key} is not supported")
                arrays[key] = np.ndarray(shape, dtype=dtype, buffer=mapping, offset=handle.tell(), order="F" if fortran else "C")
            with np.load(path, allow_pickle=False) as z:
                source_names = np.asarray(z["joint_names"]).astype(str).tolist() if "joint_names" in z else JOINT_NAMES_29.tolist()
            yield arrays, source_names
        finally:
            arrays.clear()
            mapping.close()


def _load_frames(path: Path, count: int, rng: np.random.Generator,
                 joint_names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Copy only selected frames, preserving each source's documented order."""
    with _motion_arrays(path) as (arrays, source_names):
        qpos = arrays["joint_pos"]
        body_pos = arrays["body_pos_w"]
        body_quat = arrays["body_quat_w"]
        if qpos.ndim != 2 or qpos.shape[1] != 29:
            raise ValueError(f"{path}: joint_pos has shape {qpos.shape}, expected [T,29]")
        if body_pos.ndim != 3 or body_pos.shape[1] < 1 or body_pos.shape[2] != 3:
            raise ValueError(f"{path}: body_pos_w has shape {body_pos.shape}, expected [T,B,3]")
        if body_quat.ndim != 3 or body_quat.shape[1] < 1 or body_quat.shape[2] != 4:
            raise ValueError(f"{path}: body_quat_w has shape {body_quat.shape}, expected [T,B,4]")
        n = min(qpos.shape[0], body_pos.shape[0], body_quat.shape[0])
        if n < 1:
            raise ValueError(f"{path}: empty motion")
        frames = rng.integers(0, n, size=count, dtype=np.int64)
        if len(set(source_names)) != len(source_names) or any(name not in source_names for name in joint_names):
            raise ValueError(f"{path}: joint name metadata does not match the G1 robot")
        joint_ids = np.asarray([source_names.index(name) for name in joint_names], dtype=np.int64)
        jp = np.asarray(qpos[frames[:, None], joint_ids], dtype=np.float32).copy()
        rp = np.asarray(body_pos[frames, 0], dtype=np.float32).copy()
        rq = np.asarray(body_quat[frames, 0], dtype=np.float32).copy()
        # Drop ndarray views before the read-only mapping is closed.
        del qpos, body_pos, body_quat
    norm = np.linalg.norm(rq, axis=1, keepdims=True)
    rq /= np.maximum(norm, 1.0e-8)
    if not np.isfinite(jp).all() or not np.isfinite(rp).all() or not np.isfinite(rq).all():
        raise ValueError(f"{path}: selected frame contains non-finite values")
    return jp, rp, rq, frames


def _sample_source(paths: list[Path], count: int, rng: np.random.Generator,
                   joint_names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample frames while grouping reads by file (one file is resident at a time)."""
    path_ids = rng.integers(0, len(paths), size=count, dtype=np.int64)
    frame_ids = np.zeros(count, dtype=np.int64)
    joint = np.empty((count, 29), dtype=np.float32)
    root_pos = np.empty((count, 3), dtype=np.float32)
    root_quat = np.empty((count, 4), dtype=np.float32)
    valid = np.zeros(count, dtype=bool)
    for pid in np.unique(path_ids):
        ids = np.flatnonzero(path_ids == pid)
        try:
            jp, rp, rq, frames = _load_frames(paths[int(pid)], len(ids), rng, joint_names)
        except (OSError, KeyError, ValueError) as exc:
            print(f"[WARN] skipping source {paths[int(pid)]}: {exc}")
            continue
        joint[ids], root_pos[ids], root_quat[ids] = jp, rp, rq
        frame_ids[ids] = frames
        valid[ids] = True
    if not valid.all():
        # Keep retries deterministic and avoid passing uninitialized rows to
        # PhysX.  Invalid source files are uncommon; refill those rows.
        missing = np.flatnonzero(~valid)
        if len(missing):
            j2, p2, q2, f2, pids2 = _sample_source(paths, len(missing), rng, joint_names)
            joint[missing], root_pos[missing], root_quat[missing] = j2, p2, q2
            frame_ids[missing], path_ids[missing] = f2, pids2
    return joint, root_pos, root_quat, frame_ids, path_ids


def _make_scene(args: argparse.Namespace):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim import SimulationContext
    from isaaclab.terrains import TerrainImporterCfg
    from isaaclab.utils import configclass
    from isaaclab_tasks.manager_based.textop_tracker.robots.g1 import G1_CYLINDER_CFG

    # Config classes expose ``copy`` through their dataclass implementation;
    # retain the production config untouched while making this scene passive.
    passive_robot = G1_CYLINDER_CFG.copy()
    # Configure actuator models themselves as passive as well as PhysX. This
    # keeps Isaac Lab's torque-estimation buffers truthful during validation.
    for actuator in passive_robot.actuators.values():
        actuator.stiffness = 0.0
        actuator.damping = 0.0

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        terrain = TerrainImporterCfg(
            prim_path="/World/ground", terrain_type="plane", collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply", restitution_combine_mode="multiply",
                static_friction=1.0, dynamic_friction=1.0,
            ),
        )
        robot: ArticulationCfg = passive_robot.replace(prim_path="{ENV_REGEX_NS}/Robot")

    sim_cfg = sim_utils.SimulationCfg(device=args.device)
    sim_cfg.dt = float(args.physics_dt)
    sim_cfg.render_interval = 1
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(SceneCfg(num_envs=int(args.num_envs), env_spacing=2.0))
    sim.reset()
    scene.update(sim.get_physics_dt())
    return sim, scene, scene["robot"]


def _write_bank(args: argparse.Namespace, bank: dict[str, np.ndarray], paths: list[Path], joint_names: list[str],
                generated_count: int, max_motor_torque: float) -> None:
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 1,
        "count": int(len(bank["joint_pos"])),
        "physics_dt": float(args.physics_dt),
        "duration_min": float(args.duration_min),
        "duration_max": float(args.duration_max),
        "source_files": len(paths),
        "simulated_candidates": generated_count,
        "joint_velocity_reset": True,
        "root_velocity_reset": True,
        "zero_pd": True,
        "max_motor_torque": max_motor_torque,
    }
    np.savez_compressed(
        output,
        joint_pos=bank["joint_pos"].astype(np.float32),
        joint_vel=bank["joint_vel"].astype(np.float32),
        root_pos=bank["root_pos"].astype(np.float32),
        root_quat=bank["root_quat"].astype(np.float32),
        root_lin_vel=bank["root_lin_vel"].astype(np.float32),
        root_ang_vel=bank["root_ang_vel"].astype(np.float32),
        root_state=np.concatenate((bank["root_pos"], bank["root_quat"], bank["root_lin_vel"], bank["root_ang_vel"]), axis=1).astype(np.float32),
        duration=bank["duration"].astype(np.float32),
        source_file=bank["source_file"].astype(np.int32),
        source_frame=bank["source_frame"].astype(np.int64),
        joint_names=np.asarray(joint_names),
        source_paths=np.asarray([str(path) for path in paths]),
        shoulder_height=bank["shoulder_height"].astype(np.float32),
        metadata=np.asarray([json.dumps(metadata, sort_keys=True)]),
    )
    print(f"[generator] wrote {len(bank['joint_pos'])} states to {output}")


def _generate(args: argparse.Namespace) -> int:
    if args.count < 1 or args.num_envs < 1:
        raise ValueError("--count and --num-envs must be positive")
    if not (0.0 < args.duration_min <= args.duration_max):
        raise ValueError("duration range must satisfy 0 < min <= max")
    if args.manifest:
        paths = [Path(path) for path in json.loads(Path(args.manifest).read_text())]
        if args.max_source_files is not None:
            paths = paths[:args.max_source_files]
    else:
        paths = _source_files(Path(args.input).expanduser(), args.max_source_files)
    if not paths:
        raise ValueError("empty source manifest")
    rng = np.random.default_rng(args.seed)
    sim, scene, robot = _make_scene(args)
    import torch

    nenv = int(args.num_envs)
    device = sim.device
    # Override implicit actuator parameters after initialization.  This is
    # the essential zero-torque condition: zero effort targets alone would
    # still leave implicit PD springs active.
    robot.write_joint_stiffness_to_sim(0.0)
    robot.write_joint_damping_to_sim(0.0)
    zero_effort = torch.zeros((nenv, robot.num_joints), device=device)
    zero_velocity = torch.zeros_like(zero_effort)
    dt = float(args.physics_dt)
    joint_names = list(robot.joint_names)
    shoulder_ids, _ = robot.find_bodies(["left_shoulder_roll_link", "right_shoulder_roll_link"], preserve_order=True)
    if len(shoulder_ids) != 2:
        raise RuntimeError("both G1 shoulder bodies are required to validate fallen states")
    bank = {k: [] for k in ("joint_pos", "joint_vel", "root_pos", "root_quat", "root_lin_vel", "root_ang_vel", "duration", "source_file", "source_frame", "shoulder_height")}
    produced = 0
    candidates = 0
    max_motor_torque = 0.0
    while produced < args.count:
        if candidates >= args.count * args.max_attempt_factor:
            raise RuntimeError(f"only {produced} valid falls after {candidates} simulations; inspect source data/physics")
        batch = min(nenv, args.count - produced)
        jp, rp, rq, frame_ids, source_ids = _sample_source(paths, batch, rng, joint_names)
        # Keep each environment in its own local frame and preserve the
        # sampled height/orientation.  The reference trajectory remains a
        # training concern; this bank contains only physical reset states.
        origins = scene.env_origins[:batch].detach().cpu().numpy()
        rp[:, :2] = origins[:, :2]
        qpos = torch.as_tensor(jp, device=device)
        limits = robot.data.joint_pos_limits[:batch]
        qpos = torch.minimum(torch.maximum(qpos, limits[:, :, 0]), limits[:, :, 1])
        root = robot.data.default_root_state[:batch].clone()
        root[:, :3] = torch.as_tensor(rp, device=device)
        root[:, 3:7] = torch.as_tensor(rq, device=device)
        root[:, 7:] = 0.0
        # Write only the first `batch` rows; extra environments are left idle.
        env_ids = torch.arange(batch, device=device)
        robot.reset(env_ids)
        robot.write_joint_state_to_sim(qpos, zero_velocity[:batch], env_ids=env_ids)
        robot.write_root_state_to_sim(root, env_ids=env_ids)
        scene.update(dt)
        min_steps = max(1, int(np.ceil(args.duration_min / dt)))
        max_steps = max(min_steps, int(np.floor(args.duration_max / dt)))
        finish_steps = rng.integers(min_steps, max_steps + 1, size=batch, dtype=np.int64)
        durations = (finish_steps * dt).astype(np.float32)
        pending = np.ones(batch, dtype=bool)
        final_jp = np.zeros((batch, robot.num_joints), dtype=np.float32)
        final_root = np.zeros((batch, 13), dtype=np.float32)
        final_shoulder = np.zeros(batch, dtype=np.float32)
        final_min_body_height = np.zeros(batch, dtype=np.float32)
        for step in range(int(finish_steps.max())):
            # Effort targets and implicit PD targets are refreshed every step.
            robot.set_joint_effort_target(zero_effort)
            robot.set_joint_velocity_target(zero_velocity)
            robot.set_joint_position_target(robot.data.joint_pos.clone())
            scene.write_data_to_sim()
            torque = float(robot.data.applied_torque.abs().max().item())
            max_motor_torque = max(max_motor_torque, torque)
            if not np.isfinite(torque) or torque > 1e-7:
                raise RuntimeError(f"passive-fall motor torque must be zero; observed {torque}")
            sim.step(render=False)
            scene.update(dt)
            done = pending & (finish_steps <= step + 1)
            if np.any(done):
                ids = torch.as_tensor(np.flatnonzero(done), device=device, dtype=torch.long)
                final_jp[done] = robot.data.joint_pos[ids].detach().cpu().numpy()
                final_root[done] = robot.data.root_state_w[ids].detach().cpu().numpy()
                final_shoulder[done] = robot.data.body_pos_w[ids][:, shoulder_ids, 2].mean(dim=1).detach().cpu().numpy()
                final_min_body_height[done] = robot.data.body_pos_w[ids, :, 2].min(dim=1).values.detach().cpu().numpy()
                pending[done] = False
        if pending.any():
            raise RuntimeError("internal duration capture error")
        # States are stored in their environment-local frame.  Reject upright,
        # non-finite, or grossly penetrating samples and simulate replacements.
        final_root[:, :3] -= origins
        final_shoulder -= origins[:, 2]
        final_min_body_height -= origins[:, 2]
        valid = np.isfinite(final_jp).all(axis=1) & np.isfinite(final_root).all(axis=1)
        valid &= (final_root[:, 2] > 0.02) & (final_root[:, 2] < 0.50)
        valid &= (final_shoulder > 0.02) & (final_shoulder < 0.60)
        valid &= final_min_body_height > -0.08
        valid &= np.linalg.norm(final_root[:, 7:10], axis=1) < 5.0
        valid &= np.linalg.norm(final_root[:, 10:13], axis=1) < 20.0
        candidates += batch
        accepted = int(valid.sum())
        if accepted == 0:
            print(f"[generator] no valid falls in {batch} candidates; generating replacements")
            continue
        final_jp, final_root = final_jp[valid], final_root[valid]
        # A reset starts at rest by design.  Preserve the physically settled
        # pose, but zero velocities so the first policy step is reproducible.
        bank["joint_pos"].append(final_jp)
        bank["joint_vel"].append(np.zeros_like(final_jp))
        bank["root_pos"].append(final_root[:, :3])
        bank["root_quat"].append(final_root[:, 3:7] / np.maximum(np.linalg.norm(final_root[:, 3:7], axis=1, keepdims=True), 1e-8))
        bank["root_lin_vel"].append(np.zeros((accepted, 3), dtype=np.float32))
        bank["root_ang_vel"].append(np.zeros((accepted, 3), dtype=np.float32))
        bank["duration"].append(durations[valid])
        bank["source_file"].append(source_ids[valid].astype(np.int32))
        bank["source_frame"].append(frame_ids[valid])
        bank["shoulder_height"].append(final_shoulder[valid])
        produced += accepted
        print(f"[generator] {produced}/{args.count} accepted fallen states ({candidates} simulated; max motor torque {max_motor_torque:.1g})")
    bank = {key: np.concatenate(value, axis=0) for key, value in bank.items()}
    # Validate before writing so an interrupted/bad physics run never becomes
    # a training input.
    finite = all(np.isfinite(value).all() for value in bank.values() if value.dtype.kind in "fc")
    qnorm = np.linalg.norm(bank["root_quat"], axis=1)
    if not finite or not np.allclose(qnorm, 1.0, atol=2e-3):
        raise RuntimeError("fall bank validation failed: non-finite state or invalid quaternion")
    _write_bank(args, bank, paths, joint_names, candidates, max_motor_torque)
    return 0


def main() -> int:
    parser = _parser()
    # AppLauncher adds --device and headless rendering options.  It must be
    # initialized before any Isaac Lab simulation import.
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    launcher = AppLauncher(args)
    globals()["_SIM_APP"] = launcher.app
    return _generate(args)


if __name__ == "__main__":
    raise SystemExit(main())
