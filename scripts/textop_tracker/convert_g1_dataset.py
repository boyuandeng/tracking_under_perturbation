#!/usr/bin/env python3
"""Convert Unitree G1 motion files into TextOp/Isaac Lab ``motion.npz`` files.

The source data used by TextOp is normally a joblib/pickle dictionary.  A file
can contain one motion (``root_trans_offset``, ``root_rot``, ``dof``) or a
mapping of motion names to those dictionaries.  This utility recursively scans
an input directory, resamples motions to 50 Hz and, when Isaac Lab is
available, uses an ``Articulation`` in a batched ``InteractiveScene`` to record
body forward kinematics.  ``--inventory`` intentionally does not start Isaac
Sim, which makes it safe to inspect a dataset on a machine without Isaac Lab.

No network asset is used by this script.  The robot config is imported from
``isaaclab_tasks.manager_based.textop_tracker.robots.g1`` first and from the
upstream ``textop_tracker.robots.g1`` package as a compatibility fallback.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

JOINT_NAMES_29 = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input", required=True, help="A .pkl/.joblib/.npz file or directory to scan recursively")
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument("--input-fps", type=float, default=None, help="Source FPS (otherwise use fps in each motion, then 30)")
    p.add_argument("--output-fps", type=float, default=50.0)
    p.add_argument("--valid-files", default=None, help="YAML/JSON list of allowed stems/relative paths")
    p.add_argument("--clean-output", action="store_true", help="Remove output before writing")
    p.add_argument("--inventory", action="store_true", help="List source files/keys/shapes; does not start Isaac Sim")
    p.add_argument("--dry-run", action="store_true", help="Parse and resample metadata but do not write NPZ")
    p.add_argument("--quat-order", choices=("auto", "xyzw", "wxyz"), default="auto")
    p.add_argument("--batch-size", type=int, default=128, help="Number of frames evaluated by one Isaac Lab Articulation batch")
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--max-motions", type=int, default=None)
    p.add_argument("--motion-filter", default=None, help="Regex applied to motion names")
    return p


def _load(path: Path) -> Any:
    if path.suffix.lower() == ".npz":
        z = np.load(path, allow_pickle=True)
        return {k: z[k] for k in z.files}
    try:
        import joblib
        return joblib.load(path)
    except ImportError:
        import pickle
        with path.open("rb") as f:
            return pickle.load(f)


def _scalar(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        a = np.asarray(value).reshape(-1)
        return float(a[0]) if a.size else default
    except Exception:
        return default


def _as_array(value: Any, name: str) -> np.ndarray:
    a = np.asarray(value)
    if a.dtype == object:
        a = np.asarray(a.tolist())
    if a.ndim == 1:
        a = a[:, None]
    if a.ndim != 2:
        raise ValueError(f"{name} must be [T,D], got {a.shape}")
    return np.asarray(a, dtype=np.float32)


def _is_motion(d: Any) -> bool:
    return isinstance(d, dict) and any(k in d for k in ("dof", "dof_pos", "joint_pos", "qpos")) and any(
        k in d for k in ("root_rot", "root_quat", "base_rot", "root_orientation")
    )


def _find_key(d: dict[str, Any], aliases: Iterable[str]) -> str | None:
    for key in aliases:
        if key in d:
            return key
    low = {str(k).lower(): k for k in d}
    for key in aliases:
        if key.lower() in low:
            return low[key.lower()]
    return None


def _motions(obj: Any, source: Path) -> list[tuple[str, dict[str, Any]]]:
    """Return motion dictionaries from either a single motion or motion pack."""
    if _is_motion(obj):
        return [(source.stem, obj)]
    if not isinstance(obj, dict):
        return []
    out: list[tuple[str, dict[str, Any]]] = []
    for key, value in obj.items():
        if _is_motion(value):
            out.append((str(key), value))
    return out


def _source_files(root: Path) -> list[Path]:
    exts = {".pkl", ".pickle", ".joblib", ".npz"}
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def _allowed(path: Path, source_root: Path, allowed: set[str] | None) -> bool:
    if not allowed:
        return True
    variants = {path.name, path.stem, path.as_posix(), path.relative_to(source_root).as_posix() if source_root.is_dir() else path.name}
    return bool(variants & allowed)


def _read_valid(path: str | None) -> set[str] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() in (".json",):
        raw = json.loads(p.read_text())
    else:
        try:
            import yaml
            raw = yaml.safe_load(p.read_text())
        except ImportError:
            raw = [line.strip() for line in p.read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if isinstance(raw, dict):
        for key in ("valid_files", "files", "motions", "valid"):
            if key in raw:
                raw = raw[key]
                break
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple, set)):
        raise ValueError(f"{path} does not contain a list of files")
    return {str(x).replace("\\", "/") for x in raw}


def _quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return q[:, [3, 0, 1, 2]]


def _normalize_quat(q: np.ndarray, order: str) -> np.ndarray:
    if q.shape[1] != 4:
        raise ValueError(f"root quaternion must have 4 columns, got {q.shape}")
    if order == "xyzw":
        q = _quat_xyzw_to_wxyz(q)
    elif order == "auto":
        # GMR/TextOp's published motion dictionaries use XYZW.  There is no
        # representation-only test (both conventions are unit quaternions),
        # so make that convention the safe default and let callers select
        # WXYZ explicitly when their files document it.
        q = _quat_xyzw_to_wxyz(q)
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    return q / np.maximum(norm, 1.0e-8)


def _extract(motion: dict[str, Any], name: str, quat_order: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float | None]:
    pk = _find_key(motion, ("root_trans_offset", "root_pos", "root_trans", "root_translation", "base_pos"))
    qk = _find_key(motion, ("root_rot", "root_quat", "base_rot", "root_orientation"))
    dk = _find_key(motion, ("dof", "dof_pos", "joint_pos", "qpos"))
    if not pk or not qk or not dk:
        raise ValueError(f"{name}: required root position/quaternion/dof keys not found (keys={list(motion)[:20]})")
    pos, quat, dof = _as_array(motion[pk], pk), _normalize_quat(_as_array(motion[qk], qk), quat_order), _as_array(motion[dk], dk)
    n = min(len(pos), len(quat), len(dof))
    if n < 2:
        raise ValueError(f"{name}: motion has only {n} frame(s)")
    pos, quat, dof = pos[:n], quat[:n], dof[:n]
    if dof.shape[1] == 23:
        # G1 23-DOF format has no wrist joints. Insert 3 zeros after the waist
        # and 3 zeros after the left wrist block, as in the upstream converter.
        z = np.zeros((n, 3), dtype=np.float32)
        dof = np.concatenate((dof[:, :19], z, dof[:, 19:23], z), axis=1)
    if dof.shape[1] != 29:
        raise ValueError(f"{name}: expected 23 or 29 DOF, got {dof.shape[1]}")
    fps = _scalar(_find_key(motion, ("fps", "frame_rate", "framerate")) and motion[_find_key(motion, ("fps", "frame_rate", "framerate"))], None)
    return pos, quat, dof, fps


def _slerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    # Vectorized quaternion SLERP (wxyz), with linear fallback for close pairs.
    dot = np.sum(a * b, axis=1)
    b = np.where((dot < 0)[:, None], -b, b)
    dot = np.clip(np.abs(dot), -1.0, 1.0)
    theta = np.arccos(dot)
    s = np.sin(theta)
    t = t.reshape(-1)
    safe_s = np.where(s > 1e-6, s, 1.0)
    w0 = np.where(s > 1e-6, np.sin((1.0 - t) * theta) / safe_s, 1.0 - t)
    w1 = np.where(s > 1e-6, np.sin(t * theta) / safe_s, t)
    return (w0[:, None] * a + w1[:, None] * b) / np.maximum(np.linalg.norm(w0[:, None] * a + w1[:, None] * b, axis=1, keepdims=True), 1e-8)


def _resample(pos: np.ndarray, quat: np.ndarray, dof: np.ndarray, in_fps: float, out_fps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if abs(in_fps - out_fps) < 1e-6:
        return pos, quat, dof
    times = np.arange(0.0, len(pos) / in_fps, 1.0 / out_fps, dtype=np.float64)
    src = np.clip(times * in_fps, 0, len(pos) - 1)
    i0 = np.floor(src).astype(np.int64)
    i1 = np.minimum(i0 + 1, len(pos) - 1)
    t = (src - i0).astype(np.float32)
    return ((pos[i0] * (1 - t[:, None]) + pos[i1] * t[:, None]).astype(np.float32), _slerp(quat[i0], quat[i1], t).astype(np.float32), (dof[i0] * (1 - t[:, None]) + dof[i1] * t[:, None]).astype(np.float32))


def _safe_name(name: str, used: set[str]) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("._")[:120] or "motion"
    candidate = clean
    idx = 2
    while candidate in used:
        candidate = f"{clean}_{idx}"
        idx += 1
    used.add(candidate)
    return candidate


def _inventory(args: argparse.Namespace, files: list[Path], source_root: Path, allowed: set[str] | None) -> int:
    records = []
    for path in files:
        if not _allowed(path, source_root, allowed):
            continue
        try:
            obj = _load(path)
            motions = _motions(obj, path)
            rec = {"file": str(path), "motions": []}
            for name, m in motions:
                one = {"name": name, "keys": sorted(map(str, m.keys()))}
                for k in ("root_trans_offset", "root_pos", "root_rot", "root_quat", "dof", "dof_pos", "fps"):
                    if k in m:
                        try: one[k] = list(np.asarray(m[k]).shape)
                        except Exception: one[k] = "<unreadable>"
                rec["motions"].append(one)
            records.append(rec)
        except Exception as exc:
            records.append({"file": str(path), "error": f"{type(exc).__name__}: {exc}"})
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"files": records}, indent=2, ensure_ascii=False))
    print(f"Wrote inventory for {len(records)} file(s) to {out}")
    return 0


def _fk_with_isaac(motions: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]], args: argparse.Namespace) -> dict[str, dict[str, np.ndarray]]:
    """Run batched FK. Imports Isaac only after all source data has been validated."""
    import torch
    from isaaclab.app import AppLauncher
    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim import SimulationContext
    from isaaclab.utils import configclass
    try:
        from isaaclab_tasks.manager_based.textop_tracker.robots.g1 import G1_CYLINDER_CFG
    except ImportError:
        from textop_tracker.robots.g1 import G1_CYLINDER_CFG

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # A separate scene is created for each batch size, keeping GPU memory bounded.
    nbody = None
    output: dict[str, dict[str, np.ndarray]] = {}
    app = None
    try:
        sim_cfg = sim_utils.SimulationCfg(device=args.device)
        sim_cfg.dt = 1.0 / args.output_fps
        sim = SimulationContext(sim_cfg)
        n = min(max(1, args.batch_size), max(len(x[1]) for x in motions))
        scene = InteractiveScene(SceneCfg(num_envs=n, env_spacing=2.0))
        sim.reset()
        scene.update(sim.get_physics_dt())
        robot = scene["robot"]
        joint_names = list(getattr(robot, "joint_names", JOINT_NAMES_29))
        dof_idx, _ = robot.find_joints(JOINT_NAMES_29, preserve_order=True)
        body_names = list(getattr(robot, "body_names", []))
        nbody = len(body_names)
        for name, pos, quat, dof in motions:
            t = len(pos)
            rec = {"joint_pos": [], "joint_vel": [], "body_pos_w": [], "body_quat_w": [], "body_lin_vel_w": [], "body_ang_vel_w": []}
            prev_pos, prev_body = None, None
            for start in range(0, t, n):
                end = min(t, start + n)
                k = end - start
                root = robot.data.default_root_state[:n].clone()
                root[:k, :3] = torch.as_tensor(pos[start:end], device=sim.device)
                root[:k, :2] += scene.env_origins[:k, :2]
                root[:k, 3:7] = torch.as_tensor(quat[start:end], device=sim.device)
                j = robot.data.default_joint_pos[:n].clone()
                j[:k, dof_idx] = torch.as_tensor(dof[start:end], device=sim.device)
                robot.write_root_state_to_sim(root)
                robot.write_joint_state_to_sim(j, robot.data.default_joint_vel[:n])
                sim.render()
                scene.update(sim.get_physics_dt())
                jp = robot.data.joint_pos[:k].detach().cpu().numpy().copy()
                bp = robot.data.body_pos_w[:k].detach().cpu().numpy().copy()
                bq = robot.data.body_quat_w[:k].detach().cpu().numpy().copy()
                rec["joint_pos"].append(jp); rec["body_pos_w"].append(bp); rec["body_quat_w"].append(bq)
                rec["joint_vel"].append(np.zeros_like(jp))
                rec["body_lin_vel_w"].append(np.zeros_like(bp)); rec["body_ang_vel_w"].append(np.zeros((k, nbody, 3), dtype=np.float32))
            output[name] = {k: np.concatenate(v, axis=0) for k, v in rec.items()}
        output["__names__"] = {"joint_names": np.asarray(joint_names), "body_names": np.asarray(body_names)}
    finally:
        # The caller owns AppLauncher when this function is used from main;
        # close only if one was created by a future adapter.
        pass
    return output


def _main(args: argparse.Namespace) -> int:
    root = Path(args.input).expanduser()
    files = _source_files(root)
    if args.max_files is not None:
        files = files[:args.max_files]
    allowed = _read_valid(args.valid_files)
    if args.inventory:
        return _inventory(args, files, root, allowed)
    motions: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    used_names: set[str] = set()
    for path in files:
        if not _allowed(path, root, allowed):
            continue
        try:
            entries = _motions(_load(path), path)
        except Exception as exc:
            print(f"[WARN] cannot load {path}: {exc}", file=sys.stderr); continue
        for raw_name, m in entries:
            name = str(raw_name)
            if args.motion_filter and not re.search(args.motion_filter, name):
                continue
            try:
                pos, quat, dof, fps = _extract(m, name, args.quat_order)
                pos, quat, dof = _resample(pos, quat, dof, args.input_fps or fps or 30.0, args.output_fps)
            except Exception as exc:
                print(f"[WARN] skip {path}:{name}: {exc}", file=sys.stderr); continue
            motions.append((_safe_name(name, used_names), pos, quat, dof))
            if args.max_motions is not None and len(motions) >= args.max_motions:
                break
        if args.max_motions is not None and len(motions) >= args.max_motions:
            break
    if not motions:
        raise SystemExit("No valid motions found")
    out = Path(args.output).expanduser()
    if args.clean_output and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print(json.dumps({"motions": [{"name": n, "frames": len(p), "dof": list(d.shape)} for n, p, q, d in motions]}, indent=2))
        return 0

    # Isaac Lab's AppLauncher must be initialized before importing simulation APIs.
    # The normal command path appends AppLauncher arguments, then calls _main.
    fk = _fk_with_isaac(motions, args)
    names = fk.pop("__names__", {"joint_names": np.asarray(JOINT_NAMES_29), "body_names": np.asarray([], dtype=str)})
    for name, pos, quat, dof in motions:
        rec = fk[name]
        # Velocities are derived from the input trajectories (and body velocities
        # are left zero where the Articulation API does not expose a pure-FK dt).
        rec["joint_vel"] = np.gradient(rec["joint_pos"], 1.0 / args.output_fps, axis=0).astype(np.float32)
        rec["fps"] = np.asarray([args.output_fps], dtype=np.float32)
        rec["joint_names"] = names["joint_names"]
        rec["body_names"] = names["body_names"]
        rec["root_pos"] = pos.astype(np.float32)
        rec["root_quat"] = quat.astype(np.float32)
        path = out / name / "motion.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **rec)
        print(f"Saved {path} ({len(pos)} frames)")
    return 0


def main() -> int:
    parser = _parser()
    # AppLauncher arguments are accepted only for conversion, so inventory can
    # run with plain Python.  Unknown args are passed to Isaac Lab when needed.
    args, unknown = parser.parse_known_args()
    if args.inventory or args.dry_run:
        if unknown:
            parser.error("unknown arguments in inventory/dry-run mode: " + " ".join(unknown))
        return _main(args)
    try:
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(parser)
        args = parser.parse_args()
        launcher = AppLauncher(args)
        # Keep a reference until conversion finishes; Isaac closes on process exit.
        globals()["_SIM_APP"] = launcher.app
    except ImportError as exc:
        raise SystemExit("Isaac Lab is required for conversion; use --inventory or --dry-run without it") from exc
    return _main(args)


if __name__ == "__main__":
    raise SystemExit(main())
