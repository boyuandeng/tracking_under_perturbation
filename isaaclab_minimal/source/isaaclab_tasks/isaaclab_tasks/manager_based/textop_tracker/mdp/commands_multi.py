from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import math
import mmap
import numpy as np
import os
import struct
import zipfile
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING, Optional

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MultiMotionLoader:

    # The read-only textop_npz corpus uses the Isaac G1 articulation order,
    # verified against body forward kinematics, but omits name metadata.
    # Keep this order explicit so those files
    # can be consumed without rewriting the source archive.  Files generated
    # by convert_g1_dataset.py still carry names and are reordered by name.
    _DEFAULT_JOINT_NAMES = [
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
    ]
    _DEFAULT_BODY_NAMES = [
        "pelvis", "left_hip_pitch_link", "right_hip_pitch_link", "waist_yaw_link",
        "left_hip_roll_link", "right_hip_roll_link", "waist_roll_link",
        "left_hip_yaw_link", "right_hip_yaw_link", "torso_link",
        "left_knee_link", "right_knee_link", "left_shoulder_pitch_link",
        "right_shoulder_pitch_link", "left_ankle_pitch_link", "right_ankle_pitch_link",
        "left_shoulder_roll_link", "right_shoulder_roll_link", "left_ankle_roll_link",
        "right_ankle_roll_link", "left_shoulder_yaw_link", "right_shoulder_yaw_link",
        "left_elbow_link", "right_elbow_link", "left_wrist_roll_link",
        "right_wrist_roll_link", "left_wrist_pitch_link", "right_wrist_pitch_link",
        "left_wrist_yaw_link", "right_wrist_yaw_link",
    ]

    def __init__(self, motion_files: list[str],
                 robot_joint_names: Sequence[str], body_names: Sequence[str],
                 device: str = "cpu", cache_size: int = 128,
                 source_joint_names: Sequence[str] | None = None,
                 source_body_names: Sequence[str] | None = None):
        """Index NPZ headers once and load selected arrays without conversion.

        The supplied ``textop_npz`` corpus is read-only and its anonymous
        arrays have no name metadata.  ``source_*_names`` therefore describe
        that archive's axis order explicitly.  Named archives continue to be
        reordered by their embedded metadata.  Uncompressed NPZ members are
        exposed through a read-only mmap; compressed legacy files use NumPy's
        normal reader as a compatibility fallback.
        """
        if not motion_files:
            raise ValueError("motion_files 不能为空")
        if cache_size < 1:
            raise ValueError(f"cache_size must be positive, got {cache_size}")
        self.num_files = len(motion_files)
        self.motion_files = [os.fspath(path) for path in motion_files]
        self.robot_joint_names = [str(name) for name in robot_joint_names]
        self.body_names = [str(name) for name in body_names]
        self.source_joint_names = ([str(name) for name in source_joint_names]
                                   if source_joint_names is not None else self._DEFAULT_JOINT_NAMES)
        self.source_body_names = ([str(name) for name in source_body_names]
                                  if source_body_names is not None else self._DEFAULT_BODY_NAMES)
        self.device = device
        self.cache_size = int(cache_size)
        self._cpu_cache: OrderedDict[int, dict[str, torch.Tensor]] = OrderedDict()
        self._mmap_handles: OrderedDict[int, tuple[object, mmap.mmap]] = OrderedDict()
        self.joint_dim = len(self.robot_joint_names)
        self.body_dim = len(self.body_names)

        # Header inspection is I/O bound.  Concurrently inspecting headers
        # keeps startup practical for the large read-only corpus.
        workers = min(16, max(1, len(self.motion_files)))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="textop_npz") as pool:
            self._metadata = list(pool.map(self._inspect_motion_header, self.motion_files))
        self.file_lengths = torch.tensor([item["length"] for item in self._metadata],
                                         dtype=torch.long, device=self.device)
        fps = [item["fps"] for item in self._metadata]
        if any(not np.isclose(value, fps[0], rtol=1e-5, atol=1e-5) for value in fps[1:]):
            raise ValueError(f"motion files contain inconsistent FPS values: {fps[:8]}")
        self.fps = fps[0]
        self.fps_list = fps

    @staticmethod
    def _names(value: np.ndarray, key: str, motion_file: str) -> list[str]:
        """Decode a names array stored in an NPZ and reject ambiguous metadata."""
        values = np.asarray(value).reshape(-1)
        names = [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in values]
        if not names or any(not name for name in names):
            raise ValueError(f"{motion_file}: {key} is empty")
        if len(set(names)) != len(names):
            raise ValueError(f"{motion_file}: {key} contains duplicate names")
        return names

    @staticmethod
    def _npy_header(zf: zipfile.ZipFile, key: str, motion_file: str):
        """Return shape, Fortran flag, dtype, and data offset for one member."""
        info = zf.getinfo(key + ".npy")
        with zf.open(info) as fh:
            version = np.lib.format.read_magic(fh)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(fh)
            elif version in ((2, 0), (3, 0)):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(fh)
            else:
                raise ValueError(f"{motion_file}: unsupported NPY version {version} for {key}")
        # The open handle above hides the local header offset.  Uncompressed
        # members are mmap-able at the local header's payload offset.
        data_offset = None
        if info.compress_type == zipfile.ZIP_STORED:
            with open(motion_file, "rb") as raw:
                raw.seek(info.header_offset)
                fixed = raw.read(30)
                signature, _, _, _, _, _, _, _, _, name_len, extra_len = struct.unpack(
                    "<4s5H3L2H", fixed)
                if signature != b"PK\x03\x04":
                    raise ValueError(f"{motion_file}: invalid local ZIP header for {key}")
                data_offset = info.header_offset + 30 + name_len + extra_len
                raw.seek(data_offset)
                version = np.lib.format.read_magic(raw)
                if version == (1, 0):
                    np.lib.format.read_array_header_1_0(raw)
                elif version in ((2, 0), (3, 0)):
                    np.lib.format.read_array_header_2_0(raw)
                else:
                    raise ValueError(f"{motion_file}: unsupported NPY version {version} for {key}")
                data_offset = raw.tell()
        return tuple(shape), bool(fortran), np.dtype(dtype), data_offset, info.compress_type

    def _inspect_motion_header(self, motion_file: str) -> dict:
        """Validate schema and retain only metadata, never the motion payload."""
        if not os.path.isfile(motion_file):
            raise FileNotFoundError(f"Invalid file path: {motion_file}")
        required = {
            "fps", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
            "body_lin_vel_w", "body_ang_vel_w",
        }
        with zipfile.ZipFile(motion_file, "r") as zf:
            keys = {name[:-4] for name in zf.namelist() if name.endswith(".npy")}
            missing = sorted(required.difference(keys))
            if missing:
                raise ValueError(f"{motion_file}: missing required NPZ keys: {missing}")

            has_joint_names = "joint_names" in keys
            has_body_names = "body_names" in keys
            if has_joint_names != has_body_names:
                raise ValueError(f"{motion_file}: joint_names and body_names must be present together")
            if has_joint_names:
                with np.load(motion_file, allow_pickle=False) as data:
                    joint_names = self._names(data["joint_names"], "joint_names", motion_file)
                    motion_body_names = self._names(data["body_names"], "body_names", motion_file)
            else:
                joint_names = self.source_joint_names
                motion_body_names = self.source_body_names
            missing_joints = [name for name in self.robot_joint_names if name not in joint_names]
            missing_bodies = [name for name in self.body_names if name not in motion_body_names]
            if missing_joints:
                raise ValueError(f"{motion_file}: joint_names missing robot joints: {missing_joints}")
            if missing_bodies:
                raise ValueError(f"{motion_file}: body_names missing tracked bodies: {missing_bodies}")

            headers = {key: self._npy_header(zf, key, motion_file) for key in required}
            jp_shape = headers["joint_pos"][0]
            jv_shape = headers["joint_vel"][0]
            bp_shape = headers["body_pos_w"][0]
            bq_shape = headers["body_quat_w"][0]
            blv_shape = headers["body_lin_vel_w"][0]
            bav_shape = headers["body_ang_vel_w"][0]
            if len(jp_shape) != 2 or jp_shape[1] != len(joint_names):
                raise ValueError(f"{motion_file}: joint_pos shape {jp_shape} disagrees with joint_names ({len(joint_names)})")
            if jv_shape != jp_shape:
                raise ValueError(f"{motion_file}: joint_vel shape {jv_shape} != joint_pos shape {jp_shape}")
            expected_body_shapes = {
                "body_pos_w": (jp_shape[0], len(motion_body_names), 3),
                "body_quat_w": (jp_shape[0], len(motion_body_names), 4),
                "body_lin_vel_w": (jp_shape[0], len(motion_body_names), 3),
                "body_ang_vel_w": (jp_shape[0], len(motion_body_names), 3),
            }
            shape_headers = {
                "body_pos_w": bp_shape,
                "body_quat_w": bq_shape,
                "body_lin_vel_w": blv_shape,
                "body_ang_vel_w": bav_shape,
            }
            for key, expected in expected_body_shapes.items():
                actual = tuple(shape_headers[key])
                if actual != expected:
                    raise ValueError(f"{motion_file}: {key} shape {actual} != expected {expected}")
            if jp_shape[0] < 1:
                raise ValueError(f"{motion_file}: motion contains no frames")

            fps_shape, _, _, _, _ = self._npy_header(zf, "fps", motion_file)
            if np.prod(fps_shape, dtype=np.int64) != 1:
                raise ValueError(f"{motion_file}: fps must be one scalar, got shape {fps_shape}")
            with np.load(motion_file, allow_pickle=False) as data:
                fps_values = np.asarray(data["fps"]).reshape(-1)
            if not np.isfinite(float(fps_values[0])) or float(fps_values[0]) <= 0:
                raise ValueError(f"{motion_file}: fps must be positive finite, got {fps_values}")
            fps = float(fps_values[0])

        return {"length": int(jp_shape[0]), "fps": fps,
                "headers": headers, "joint_indexes": [joint_names.index(name) for name in self.robot_joint_names],
                "body_indexes": [motion_body_names.index(name) for name in self.body_names],
                "has_names": has_joint_names}

    def _load_cpu(self, motion_idx: int) -> dict[str, torch.Tensor]:
        """Load one motion into the bounded CPU cache.

        ZIP_STORED members are read from a shared read-only mmap, avoiding
        decompression and temporary allocations.  The selected robot axes are
        copied once into contiguous float32 tensors; this is the only layout
        adaptation required by the tracker.
        """
        if motion_idx in self._cpu_cache:
            self._cpu_cache.move_to_end(motion_idx)
            return self._cpu_cache[motion_idx]
        motion_file = self.motion_files[motion_idx]
        meta = self._metadata[motion_idx]
        arrays = {}
        handle = None
        mm = None
        if all(meta["headers"][key][3] is not None for key in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")):
            handle = open(motion_file, "rb")
            mm = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
            self._mmap_handles[motion_idx] = (handle, mm)
            self._mmap_handles.move_to_end(motion_idx)
            def source_array(key):
                shape, fortran, dtype, offset, _ = meta["headers"][key]
                if fortran:
                    strides = np.empty(len(shape), dtype=np.int64)
                    stride = dtype.itemsize
                    for axis in range(len(shape)):
                        strides[axis] = stride
                        stride *= shape[axis]
                    strides = tuple(int(v) for v in strides)
                else:
                    strides = None
                return np.ndarray(shape=shape, dtype=dtype, buffer=mm,
                                  offset=offset, strides=strides)
            source = {key: source_array(key) for key in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")}
        else:
            with np.load(motion_file, allow_pickle=False) as data:
                source = {key: np.asarray(data[key]) for key in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")}
        def tensor(key: str, indexes: Sequence[int]) -> torch.Tensor:
            # Selecting axes and making one C-contiguous float32 copy avoids
            # carrying archive strides into the simulator buffers.
            array = np.asarray(source[key])[:, indexes]
            return torch.from_numpy(np.array(array, dtype=np.float32, copy=True, order="C"))
        motion = {"joint_pos": tensor("joint_pos", meta["joint_indexes"]),
                  "joint_vel": tensor("joint_vel", meta["joint_indexes"]),
                  "body_pos_w": tensor("body_pos_w", meta["body_indexes"]),
                  "body_quat_w": tensor("body_quat_w", meta["body_indexes"]),
                  "body_lin_vel_w": tensor("body_lin_vel_w", meta["body_indexes"]),
                  "body_ang_vel_w": tensor("body_ang_vel_w", meta["body_indexes"])}
        self._cpu_cache[motion_idx] = motion
        self._cpu_cache.move_to_end(motion_idx)
        while len(self._cpu_cache) > self.cache_size:
            old_idx, _ = self._cpu_cache.popitem(last=False)
            old = self._mmap_handles.pop(old_idx, None)
            if old is not None:
                old[1].close()
                old[0].close()
        return motion

    def _source_arrays(self, motion_idx: int) -> dict[str, np.ndarray] | None:
        """Return zero-copy arrays for a stored NPZ, or None for compressed files."""
        meta = self._metadata[motion_idx]
        keys = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
                "body_lin_vel_w", "body_ang_vel_w")
        if not all(meta["headers"][key][3] is not None for key in keys):
            return None
        if motion_idx not in self._mmap_handles:
            handle = open(self.motion_files[motion_idx], "rb")
            mm = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
            self._mmap_handles[motion_idx] = (handle, mm)
            self._mmap_handles.move_to_end(motion_idx)
        mm = self._mmap_handles[motion_idx][1]
        arrays = {}
        for key in keys:
            shape, fortran, dtype, offset, _ = meta["headers"][key]
            if fortran:
                strides = []
                stride = dtype.itemsize
                for axis in range(len(shape)):
                    strides.append(stride)
                    stride *= shape[axis]
                arrays[key] = np.ndarray(shape=shape, dtype=dtype, buffer=mm,
                                         offset=offset, strides=tuple(strides))
            else:
                arrays[key] = np.ndarray(shape=shape, dtype=dtype, buffer=mm,
                                         offset=offset)
        return arrays

    def _load_window_cpu(self, motion_idx: int, starts: list[int],
                         length: int) -> dict[str, np.ndarray] | None:
        """Read only requested frames from an uncompressed NPZ mmap."""
        source = self._source_arrays(motion_idx)
        if source is None:
            return None
        n_frames = int(self.file_lengths[motion_idx].item())
        result = {}
        times = np.asarray(starts, dtype=np.int64)[:, None] + np.arange(length, dtype=np.int64)[None, :]
        np.clip(times, 0, n_frames - 1, out=times)
        meta = self._metadata[motion_idx]
        for key, indexes in (("joint_pos", meta["joint_indexes"]), ("joint_vel", meta["joint_indexes"]),
                             ("body_pos_w", meta["body_indexes"]), ("body_quat_w", meta["body_indexes"]),
                             ("body_lin_vel_w", meta["body_indexes"]), ("body_ang_vel_w", meta["body_indexes"])):
            # Index the selected joint/body axis in the same advanced-index
            # expression as the frame window.  This avoids first reading all
            # 30 bodies from a remote clip when the tracker uses 14.
            selected = np.asarray(indexes, dtype=np.int64)
            if source[key].ndim == 2:
                result[key] = np.asarray(source[key][times[..., None], selected], dtype=np.float32)
            else:
                result[key] = np.asarray(source[key][times[..., None], selected, :], dtype=np.float32)
        return result

    def get_motion_windows(self, motion_indices: torch.Tensor,
                           starts: torch.Tensor, length: int) -> dict[str, torch.Tensor]:
        """Gather many reset windows and transfer each array once to device."""
        if length < 1:
            raise ValueError(f"window length must be positive, got {length}")
        indices = motion_indices.detach().to(device="cpu", dtype=torch.long).reshape(-1)
        starts_cpu = starts.detach().to(device="cpu", dtype=torch.long).reshape(-1)
        if indices.numel() != starts_cpu.numel():
            raise ValueError("motion_indices and starts must have the same number of environments")
        count = int(indices.numel())
        output = {"joint_pos": torch.empty((count, length, self.joint_dim), dtype=torch.float32),
                  "joint_vel": torch.empty((count, length, self.joint_dim), dtype=torch.float32),
                  "body_pos_w": torch.empty((count, length, self.body_dim, 3), dtype=torch.float32),
                  "body_quat_w": torch.empty((count, length, self.body_dim, 4), dtype=torch.float32),
                  "body_lin_vel_w": torch.empty((count, length, self.body_dim, 3), dtype=torch.float32),
                  "body_ang_vel_w": torch.empty((count, length, self.body_dim, 3), dtype=torch.float32)}
        # Group by motion so each source array is looked up once even when a
        # reset batch contains thousands of environments.
        by_motion: dict[int, list[int]] = {}
        for env_id, motion_idx in enumerate(indices.tolist()):
            if motion_idx < 0 or motion_idx >= self.num_files:
                raise IndexError(f"motion_idx {motion_idx} outside [0, {self.num_files})")
            by_motion.setdefault(motion_idx, []).append(env_id)
        for motion_idx, env_ids in by_motion.items():
            start_values = [int(starts_cpu[env_id]) for env_id in env_ids]
            window = self._load_window_cpu(motion_idx, start_values, length)
            if window is not None:
                for local_id, env_id in enumerate(env_ids):
                    for key, value in window.items():
                        output[key][env_id].copy_(torch.from_numpy(value[local_id]))
            else:
                motion = self._load_cpu(motion_idx)
                n_frames = int(self.file_lengths[motion_idx].item())
                for env_id in env_ids:
                    times = torch.arange(int(starts_cpu[env_id]), int(starts_cpu[env_id]) + length, dtype=torch.long)
                    times.clamp_(0, n_frames - 1)
                    for key, value in motion.items():
                        output[key][env_id].copy_(value.index_select(0, times))
        return {key: value.to(self.device, non_blocking=True) for key, value in output.items()}

    def get_motion_data_batch(
            self, motion_idx: int, time_steps_start: torch.Tensor,
            time_steps_end: torch.Tensor) -> dict[str, torch.Tensor]:
        motion_idx = int(motion_idx)
        if motion_idx < 0 or motion_idx >= self.num_files:
            raise IndexError(f"motion_idx {motion_idx} outside [0, {self.num_files})")
        start = int(time_steps_start.item()) if torch.is_tensor(time_steps_start) else int(time_steps_start)
        end = int(time_steps_end.item()) if torch.is_tensor(time_steps_end) else int(time_steps_end)
        return {key: value[0] for key, value in self.get_motion_windows(
            torch.tensor([motion_idx], dtype=torch.long),
            torch.tensor([start], dtype=torch.long), end - start).items()}


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(
            self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(
            self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(self.robot.find_bodies(
            self.cfg.body_names, preserve_order=True)[0],
                                         dtype=torch.long,
                                         device=self.device)

        self.motion = MultiMotionLoader(self.cfg.motion_files,
                                        robot_joint_names=self.robot.joint_names,
                                        body_names=self.cfg.body_names,
                                        device=self.device,
                                        cache_size=self.cfg.motion_cache_size)

        self.buffer_length: int = np.min(
            [env.max_episode_length,
             (self.motion.file_lengths.max().item())]  # type: ignore
        ) + self.cfg.future_steps  # type: ignore

        # 初始化状态变量
        self.time_steps = torch.zeros(self.num_envs,
                                      dtype=torch.long,
                                      device=self.device)
        self.motion_idx = torch.zeros(self.num_envs,
                                      dtype=torch.long,
                                      device=self.device)
        self.motion_length = torch.zeros(self.num_envs,
                                         dtype=torch.long,
                                         device=self.device)
        self.buffer_start_time = torch.zeros(self.num_envs,
                                             dtype=torch.long,
                                             device=self.device)
        # Incremented whenever a command reset writes a new physical state.
        # Stateful recovery terms use it to ignore reset/clip-switch jumps.
        self.reset_generation = torch.zeros(self.num_envs,
                                            dtype=torch.long,
                                            device=self.device)
        # ``_resample_command`` is also used when a motion clip naturally
        # ends.  Keep an explicit context bit so the offline fall bank and
        # its recovery hold are sampled only by a real episode reset.
        self._episode_reset_in_progress = False

        # A fraction of resets start from a pre-generated offline fall-state
        # bank.  The bank is loaded once at startup, so sampling the physical
        # fall state needs no archive read or extra simulator steps. Reference
        # buffers remain untouched and continue from the sampled frame.
        self.fall_reset_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.fall_reset_prob = float(getattr(self.cfg, "fall_reset_prob", 0.35))
        if not 0.0 <= self.fall_reset_prob <= 1.0:
            raise ValueError(f"fall_reset_prob must be in [0,1], got {self.fall_reset_prob}")
        # A fallen initialization gets a short reference hold.  This is
        # measured in command/control updates (the motion files and the
        # environment both run at the same 50 Hz rate), rather than in the
        # absolute motion-frame index: a random start frame may leave less
        # than the recovery window before the clip's last frame.
        self.fall_hold_seconds = max(
            0.0, float(getattr(self.cfg, "fall_recovery_hold_time", 6.0))
        )
        self.fall_hold_steps = 0
        self.fall_hold_remaining = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.fall_state_path = os.fspath(getattr(
            self.cfg, "fall_state_path",
            os.environ.get("TEXTOP_FALL_STATE_PATH", "data/textop_fall_states/fall_states_16384.npz"),
        ))
        if self.fall_reset_prob > 0.0:
            self.fall_state_joint_pos, self.fall_state_root = self._load_fall_state_bank(
                self.fall_state_path
            )
            print(
                f"[TextOp] Loaded {self.fall_state_joint_pos.shape[0]} offline fall states "
                f"from {self.fall_state_path}; reset probability={self.fall_reset_prob:.2f}. "
                "Fall-state joint/root velocities are zeroed at reset."
            )
        else:
            self.fall_state_joint_pos = torch.empty((0, len(self.robot.joint_names)), device=self.device)
            self.fall_state_root = torch.empty((0, 13), device=self.device)

        # ``MotionCommand._update_command`` advances one motion frame per
        # control update.  Compute the hold length after the loader has
        # validated the common FPS of all clips.
        self.fall_hold_steps = max(
            0, int(math.ceil(self.fall_hold_seconds * float(self.motion.fps)))
        )

        # 初始化buffer，存储轨迹数据
        self._init_buffers()

        self.body_pos_relative_w = torch.zeros(self.num_envs,
                                               len(cfg.body_names),
                                               3,
                                               device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs,
                                                len(cfg.body_names),
                                                4,
                                                device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.num_motion: int = self.motion.file_lengths.shape[0]  # type:ignore
        self.failed_motion_count = torch.zeros(
            self.num_motion, dtype=torch.float, device=self.device) + 1
        self.success_motion_count = torch.zeros(
            self.num_motion, dtype=torch.float, device=self.device) + 1
        self.failed_total_count: float = 0.0 + 1e-8
        self.success_total_count: float = 0.0 + 1e-8
        self.current_failed_motion_count = torch.zeros(self.num_motion,
                                                       dtype=torch.float,
                                                       device=self.device)
        self.current_success_motion_count = torch.zeros(self.num_motion,
                                                        dtype=torch.float,
                                                        device=self.device)

        if self.cfg.adaptive_length_weighted:
            assert self.cfg.ads_type == "v1", "Adaptive length weighted is only supported for v1 adaptive sampling type"

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs,
                                                       device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs,
                                                       device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs,
                                                           device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs,
                                                           device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs,
                                                     device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs,
                                                     device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs,
                                                      device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs,
                                                      device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs,
                                                       device=self.device)
        self.metrics["pfail_entropy"] = torch.zeros(self.num_envs,
                                                    device=self.device)
        self.metrics["pfail_mean"] = torch.zeros(self.num_envs,
                                                 device=self.device)
        self.metrics["failed_total_count"] = torch.zeros(
            self.num_envs, device=self.device)  # Strange But works
        self.metrics["success_total_count"] = torch.zeros(self.num_envs,
                                                          device=self.device)
        self.metrics["pfail_total"] = torch.zeros(self.num_envs,
                                                  device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs,
                                                         device=self.device)

    def _load_fall_state_bank(self, path: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Load and validate the immutable offline fall-state bank once.

        ``root_state`` follows Isaac Lab's order ``xyz, quat(wxyz),
        linear_velocity, angular_velocity``.  The training reset explicitly
        zeros both root and joint velocity below, regardless of values saved
        by the generator, as required for a stable fall initialization.
        """
        if not path:
            raise ValueError("fall_state_path must be set when fall resets are enabled")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Fall reset bank not found: {path}. Generate the offline bank before training."
            )
        try:
            with np.load(path, allow_pickle=False) as data:
                missing = [key for key in ("joint_pos", "root_state", "joint_names") if key not in data]
                if missing:
                    raise ValueError(f"missing fall-state arrays {missing}")
                joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
                root_state = np.asarray(data["root_state"], dtype=np.float32)
                values = np.asarray(data["joint_names"]).reshape(-1)
                bank_names = [
                    item.decode("utf-8") if isinstance(item, bytes) else str(item)
                    for item in values
                ]
        except (OSError, ValueError) as exc:
            raise ValueError(f"Could not read fall-state bank {path}: {exc}") from exc

        joint_count = len(self.robot.joint_names)
        if joint_pos.ndim != 2 or joint_pos.shape[1] != joint_count:
            raise ValueError(
                f"{path}: joint_pos must have shape [N,{joint_count}], got {joint_pos.shape}"
            )
        if root_state.ndim != 2 or root_state.shape[0] != joint_pos.shape[0] or root_state.shape[1] != 13:
            raise ValueError(
                f"{path}: root_state must have shape [N,13] matching joint_pos, got {root_state.shape}"
            )
        if joint_pos.shape[0] < 1:
            raise ValueError(f"{path}: fall-state bank is empty")
        if len(bank_names) != joint_count or len(bank_names) != len(set(bank_names)):
            raise ValueError(f"{path}: joint_names must contain {joint_count} unique names")
        missing = [name for name in self.robot.joint_names if name not in bank_names]
        if missing:
            raise ValueError(f"{path}: joint_names missing robot joints {missing}")
        joint_pos = joint_pos[:, [bank_names.index(name) for name in self.robot.joint_names]]
        if not np.isfinite(joint_pos).all() or not np.isfinite(root_state).all():
            raise ValueError(f"{path}: fall-state bank contains NaN or infinite values")
        quat_norm = np.linalg.norm(root_state[:, 3:7], axis=1)
        if np.any(quat_norm < 1.0e-6):
            raise ValueError(f"{path}: root_state contains zero-length quaternions")
        root_state[:, 3:7] /= quat_norm[:, None]

        # Resident device tensors make reset sampling a gather only: no disk
        # read, NPZ parsing, or decompression occurs during training.
        return (
            torch.as_tensor(joint_pos, dtype=torch.float32, device=self.device),
            torch.as_tensor(root_state, dtype=torch.float32, device=self.device),
        )

    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset commands and sample fall states for this episode reset only.

        ``_resample_command`` is also used by ``_update_command`` when a
        reference clip naturally reaches its final frame.  Keep that path
        separate so a clip switch cannot inject another fall state or restart
        the three-second hold timer.
        """
        self._episode_reset_in_progress = True
        try:
            return super().reset(env_ids)
        finally:
            self._episode_reset_in_progress = False

    def _init_buffers(self):
        """初始化buffer存储轨迹数据"""
        # 获取joint数量
        if self.motion.joint_dim is None:
            raise RuntimeError("Motion loader did not determine the joint dimension")
        joint_dim = self.motion.joint_dim
        body_dim = len(self.cfg.body_names)

        # 初始化buffer，形状为 (num_envs, buffer_length, ...)
        self.joint_pos_buffer = torch.zeros(self.num_envs,
                                            self.buffer_length,
                                            joint_dim,
                                            device=self.device)
        self.joint_vel_buffer = torch.zeros(self.num_envs,
                                            self.buffer_length,
                                            joint_dim,
                                            device=self.device)
        self.body_pos_w_buffer = torch.zeros(self.num_envs,
                                             self.buffer_length,
                                             body_dim,
                                             3,
                                             device=self.device)
        self.body_quat_w_buffer = torch.zeros(self.num_envs,
                                              self.buffer_length,
                                              body_dim,
                                              4,
                                              device=self.device)
        self.body_lin_vel_w_buffer = torch.zeros(self.num_envs,
                                                 self.buffer_length,
                                                 body_dim,
                                                 3,
                                                 device=self.device)
        self.body_ang_vel_w_buffer = torch.zeros(self.num_envs,
                                                 self.buffer_length,
                                                 body_dim,
                                                 3,
                                                 device=self.device)

        # 初始化quaternion为[1,0,0,0]
        self.body_quat_w_buffer[:, :, :, 0] = 1.0

    def _update_buffers(self, env_ids: Optional[torch.Tensor] = None):
        """更新buffer，从motion数据中填充buffer"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        if len(env_ids) == 0:
            return

        motion_data = self.motion.get_motion_windows(
            self.motion_idx[env_ids], self.buffer_start_time[env_ids], self.buffer_length)
        self.joint_pos_buffer[env_ids] = motion_data["joint_pos"]
        self.joint_vel_buffer[env_ids] = motion_data["joint_vel"]
        self.body_pos_w_buffer[env_ids] = motion_data["body_pos_w"]
        self.body_quat_w_buffer[env_ids] = motion_data["body_quat_w"]
        self.body_lin_vel_w_buffer[env_ids] = motion_data["body_lin_vel_w"]
        self.body_ang_vel_w_buffer[env_ids] = motion_data["body_ang_vel_w"]

        if self.cfg.random_static_prob > 0:
            # Usage: p = random_static_prob
            # p=0.1, then 10% of the envs will be static.
            # p<0: disable random static.
            # p>1: enable all envs to be static.
            static_mask = torch.rand(
                len(env_ids), device=self.device) < self.cfg.random_static_prob
            if static_mask.any():
                static_env_ids = env_ids[static_mask]
                # 平移所有body, 使得所有帧的anchor等于第一帧的anchor
                # [N, buffer_length, body_dim, 3]
                anchor_first = self.body_pos_w_buffer[
                    static_env_ids, 0:1,
                    self.motion_anchor_body_index, :]  # [N, 1, 3]
                anchor_current = self.body_pos_w_buffer[
                    static_env_ids, :,
                    self.motion_anchor_body_index, :]  # [N, buffer_length, 3]
                translation = anchor_first - anchor_current  # [N, buffer_length, 3]
                self.body_pos_w_buffer[
                    static_env_ids, :, :, :] += translation[:, :, None, :]

                # 修正body的速度：移除anchor translation后的anchor vel分量, 只保留相对速度
                # frame_dt应为buffer内的步长, 取自数据
                # anchor 原先vel [N, buffer_length, 3]
                anchor_vel = self.body_lin_vel_w_buffer[
                    static_env_ids, :,
                    self.motion_anchor_body_index, :]  # [N, buffer_length, 3]
                # 修正每个body vel：减去anchor vel
                self.body_lin_vel_w_buffer[
                    static_env_ids, :, :, :] -= anchor_vel[:, :, None, :]

    @property
    def command(self) -> torch.Tensor:
        cmd = torch.cat([self.motion_joint_pos, self.motion_joint_vel], dim=1)
        return cmd

    @property
    def joint_pos(self) -> torch.Tensor:
        buffer_indices = torch.clamp(self.time_steps - self.buffer_start_time,
                                     0, self.buffer_length - 1)
        return self.joint_pos_buffer[
            torch.arange(self.num_envs, device=self.device), buffer_indices]

    @property
    def joint_vel(self) -> torch.Tensor:
        buffer_indices = torch.clamp(self.time_steps - self.buffer_start_time,
                                     0, self.buffer_length - 1)
        return self.joint_vel_buffer[
            torch.arange(self.num_envs, device=self.device), buffer_indices]

    @property
    def body_pos_w(self) -> torch.Tensor:
        buffer_indices = torch.clamp(self.time_steps - self.buffer_start_time,
                                     0, self.buffer_length - 1)
        return self.body_pos_w_buffer[
            torch.arange(self.num_envs, device=self.device),
            buffer_indices] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        buffer_indices = torch.clamp(self.time_steps - self.buffer_start_time,
                                     0, self.buffer_length - 1)
        return self.body_quat_w_buffer[
            torch.arange(self.num_envs, device=self.device), buffer_indices]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        buffer_indices = torch.clamp(self.time_steps - self.buffer_start_time,
                                     0, self.buffer_length - 1)
        return self.body_lin_vel_w_buffer[
            torch.arange(self.num_envs, device=self.device), buffer_indices]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        buffer_indices = torch.clamp(self.time_steps - self.buffer_start_time,
                                     0, self.buffer_length - 1)
        return self.body_ang_vel_w_buffer[
            torch.arange(self.num_envs, device=self.device), buffer_indices]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        buffer_indices = torch.clamp(self.time_steps - self.buffer_start_time,
                                     0, self.buffer_length - 1)
        return (self.body_pos_w_buffer[
            torch.arange(self.num_envs, device=self.device), buffer_indices,
            self.motion_anchor_body_index] + self._env.scene.env_origins)

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        buffer_indices = torch.clamp(self.time_steps - self.buffer_start_time,
                                     0, self.buffer_length - 1)
        return self.body_quat_w_buffer[
            torch.arange(self.num_envs, device=self.device), buffer_indices,
            self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        buffer_indices = torch.clamp(self.time_steps - self.buffer_start_time,
                                     0, self.buffer_length - 1)
        return self.body_lin_vel_w_buffer[
            torch.arange(self.num_envs, device=self.device), buffer_indices,
            self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        buffer_indices = torch.clamp(self.time_steps - self.buffer_start_time,
                                     0, self.buffer_length - 1)
        return self.body_ang_vel_w_buffer[
            torch.arange(self.num_envs, device=self.device), buffer_indices,
            self.motion_anchor_body_index]

    # Future reference properties for N-step lookahead
    @property
    def motion_joint_pos(self) -> torch.Tensor:
        """Future N-step joint positions reference."""
        if self.cfg.future_steps <= 1:
            return self.joint_pos
        else:
            current_indices = torch.clamp(
                self.time_steps - self.buffer_start_time, 0,
                self.buffer_length - 1)
            future_indices = current_indices[:, None] + torch.arange(
                self.cfg.future_steps, device=self.device)[None, :]
            future_indices = torch.clamp(future_indices, 0,
                                         self.buffer_length - 1)
            return self.joint_pos_buffer[
                torch.arange(self.num_envs, device=self.device)[:, None],
                future_indices].view(self.num_envs, -1)

    @property
    def motion_joint_vel(self) -> torch.Tensor:
        """Future N-step joint velocities reference."""
        if self.cfg.future_steps <= 1:
            return self.joint_vel
        else:
            current_indices = torch.clamp(
                self.time_steps - self.buffer_start_time, 0,
                self.buffer_length - 1)
            future_indices = current_indices[:, None] + torch.arange(
                self.cfg.future_steps, device=self.device)[None, :]
            future_indices = torch.clamp(future_indices, 0,
                                         self.buffer_length - 1)
            return self.joint_vel_buffer[
                torch.arange(self.num_envs, device=self.device)[:, None],
                future_indices].view(self.num_envs, -1)

    @property
    def motion_anchor_pos(self) -> torch.Tensor:
        """Future N-step anchor positions reference."""
        if self.cfg.future_steps <= 1:
            return self.anchor_pos_w
        else:
            current_indices = torch.clamp(
                self.time_steps - self.buffer_start_time, 0,
                self.buffer_length - 1)
            future_indices = current_indices[:, None] + torch.arange(
                self.cfg.future_steps, device=self.device)[None, :]
            future_indices = torch.clamp(future_indices, 0,
                                         self.buffer_length - 1)
            future_pos = self.body_pos_w_buffer[
                torch.arange(self.num_envs, device=self.device)[:, None],
                future_indices, self.motion_anchor_body_index]
            return (future_pos + self._env.scene.env_origins[:, None, :]).view(
                self.num_envs, -1)

    @property
    def motion_anchor_quat(self) -> torch.Tensor:
        """Future N-step anchor quaternions reference."""
        if self.cfg.future_steps <= 1:
            return self.anchor_quat_w
        else:
            current_indices = torch.clamp(
                self.time_steps - self.buffer_start_time, 0,
                self.buffer_length - 1)
            future_indices = current_indices[:, None] + torch.arange(
                self.cfg.future_steps, device=self.device)[None, :]
            future_indices = torch.clamp(future_indices, 0,
                                         self.buffer_length - 1)
            future_quat = self.body_quat_w_buffer[
                torch.arange(self.num_envs, device=self.device)[:, None],
                future_indices, self.motion_anchor_body_index]
            return future_quat.view(self.num_envs, -1)

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w -
                                                      self.robot_anchor_pos_w,
                                                      dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(
            self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(
            self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(
            self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w -
                                                    self.robot_body_pos_w,
                                                    dim=-1).mean(dim=-1)
        self.metrics["error_body_rot"] = quat_error_magnitude(
            self.body_quat_relative_w, self.robot_body_quat_w).mean(dim=-1)

        self.metrics["error_body_lin_vel"] = torch.norm(
            self.body_lin_vel_w - self.robot_body_lin_vel_w,
            dim=-1).mean(dim=-1)
        self.metrics["error_body_ang_vel"] = torch.norm(
            self.body_ang_vel_w - self.robot_body_ang_vel_w,
            dim=-1).mean(dim=-1)

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos -
                                                     self.robot_joint_pos,
                                                     dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel -
                                                     self.robot_joint_vel,
                                                     dim=-1)

    def _adaptive_sampling(self, env_ids: torch.Tensor) -> torch.Tensor:
        episode_failed: torch.Tensor = self._env.termination_manager.terminated[
            env_ids]  # type:ignore
        episode_timeout: torch.Tensor = self._env.termination_manager.time_outs[
            env_ids]  # type:ignore
        episode_success: torch.Tensor = ~episode_failed  # 所有不是episode_failed的env
        # 既不是terminate也不是timeout:  在一个episode内, motion结束了, 也会resample. 统计为success
        if torch.any(episode_failed):
            # 统计每个motion的失败次数
            failed_motions = self.motion_idx[env_ids][episode_failed]
            self.current_failed_motion_count[:] = torch.bincount(
                failed_motions, minlength=self.num_motion)
        if torch.any(episode_success):
            # 统计每个motion的成功次数
            success_motions = self.motion_idx[env_ids][episode_success]
            self.current_success_motion_count[:] = torch.bincount(
                success_motions, minlength=self.num_motion)

        if self.cfg.ads_type == "v1":
            p_fail = self.failed_motion_count / (
                self.failed_motion_count.sum() + 1e-8)
            if self.cfg.adaptive_length_weighted:
                p_fail = p_fail * self.motion.file_lengths
                p_fail = p_fail / (p_fail.sum() + 1e-8)
            p_fail_sample = torch.pow(p_fail, self.cfg.adaptive_beta)
            p_fail_sample = p_fail_sample / (p_fail_sample.sum() + 1e-8)

            sampling_probabilities = (
                p_fail_sample * (1 - self.cfg.adaptive_uniform_ratio) +
                self.cfg.adaptive_uniform_ratio / float(self.num_motion))
        elif self.cfg.ads_type == "v2":
            p_fail = self.failed_motion_count / (
                self.failed_motion_count + self.success_motion_count + 1e-8)
            p_fail_sample = torch.pow(p_fail, self.cfg.adaptive_beta)
            p_fail_sample = p_fail_sample / (p_fail_sample.sum() + 1e-8)

            sampling_probabilities = (
                p_fail_sample * (1 - self.cfg.adaptive_uniform_ratio) +
                self.cfg.adaptive_uniform_ratio / float(self.num_motion))
        elif self.cfg.ads_type == "v3":
            p_fail = self.failed_motion_count / (
                self.failed_motion_count + self.success_motion_count + 1e-8)
            p_fail_sample = 1 - torch.pow(1 - p_fail, self.cfg.adaptive_beta)
            p_fail_sample = p_fail_sample / (p_fail_sample.sum() + 1e-8)

            sampling_probabilities = (
                p_fail_sample * (1 - self.cfg.adaptive_uniform_ratio) +
                self.cfg.adaptive_uniform_ratio / float(self.num_motion))
        else:
            raise ValueError(
                f"Unsupported adaptive sampling type: {self.cfg.ads_type}")

        # 根据概率进行采样
        sampled_motion_ids = torch.multinomial(sampling_probabilities,
                                               len(env_ids),
                                               replacement=True)

        # Metrics
        H_samp = -(sampling_probabilities *
                   (sampling_probabilities + 1e-12).log()).sum()
        H_samp_norm = H_samp / math.log(self.num_motion)

        p_fail_norm = p_fail / (p_fail.sum() + 1e-8)
        H_pfail = -(p_fail_norm * (p_fail_norm + 1e-12).log()).sum()
        H_pfail_norm = H_pfail / math.log(self.num_motion)

        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_samp_norm
        self.metrics["pfail_entropy"][:] = H_pfail_norm
        self.metrics["pfail_mean"][:] = p_fail.mean()

        self.metrics["sampling_top1_prob"][:] = pmax
        return sampled_motion_ids

    def _uniform_sampling(self, env_ids: torch.Tensor) -> torch.Tensor:
        return (sample_uniform(0.0, 1.0,
                               (len(env_ids), ), device=self.device) *
                self.motion.num_files).long()

    def _resample_command(self, env_ids: torch.Tensor, reset_physical: bool = True):
        if len(env_ids) == 0:
            return

        # A natural clip transition changes only the reference trajectory.
        # Physical state writes belong to a real episode reset; otherwise a
        # fallen/recovering robot would be teleported to the next clip's first
        # frame as soon as its three-second hold expires.
        if reset_physical:
            self.reset_generation[env_ids] += 1

        if self.cfg.enable_adaptive_sampling:
            self.motion_idx[env_ids] = self._adaptive_sampling(env_ids)
        else:
            self.motion_idx[env_ids] = self._uniform_sampling(env_ids)
        self.motion_length[env_ids] = self.motion.file_lengths[
            self.motion_idx[env_ids]]

        if self.cfg.start_from_zero_step:
            self.time_steps[env_ids] = torch.zeros(len(env_ids),
                                                   dtype=torch.long,
                                                   device=self.device)
        else:
            self.time_steps[env_ids] = (
                sample_uniform(0.0, 1.0,
                               (len(env_ids), ), device=self.device) *
                self.motion_length[env_ids]).long()

        self.buffer_start_time[env_ids] = self.time_steps[env_ids].clone()

        # 取出 所有环境, 整个episode的reference motion data
        self._update_buffers(env_ids)

        if not reset_physical:
            return

        # Preserve the unperturbed reference pelvis position for fall resets.
        # ``root_pos`` below is intentionally randomized for normal starts,
        # but using that perturbed value for a fallen state leaves the bank
        # base up to 5 cm away from the first reference frame.  The recovery
        # episode should start from the same sampled reference location.
        reference_root_xy = self.body_pos_w[env_ids, 0, :2].clone()

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [
            self.cfg.pose_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0],
                                      ranges[:, 1], (len(env_ids), 6),
                                      device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3],
                                                 rand_samples[:, 4],
                                                 rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [
            self.cfg.velocity_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0],
                                      ranges[:, 1], (len(env_ids), 6),
                                      device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range,
                                    joint_pos.shape,
                                    device=self.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(joint_pos[env_ids],
                                        soft_joint_pos_limits[:, :, 0],
                                        soft_joint_pos_limits[:, :, 1])
        self.robot.write_joint_state_to_sim(joint_pos[env_ids],
                                            joint_vel[env_ids],
                                            env_ids=env_ids)  # type: ignore
        self.robot.write_root_state_to_sim(
            torch.cat([
                root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids],
                root_ang_vel[env_ids]
            ],
                      dim=-1),
            env_ids=env_ids,  # type: ignore
        )

        # Offline fall-state initialization. Reference time remains the
        # sampled frame and continues normally after reset; only the physical
        # robot state is replaced by a resident bank sample.  The hold timer
        # is attached to this sampled fall reset (and is cleared for ordinary
        # resets), so a normal episode does not pin its reference at the end
        # of a clip.
        # Only a real episode reset may inject an offline fall.  Natural clip
        # switches pass through this method as well, but preserve the current
        # episode's fall context by using the ordinary reference
        # initialization path and never injecting another fall state.
        if self._episode_reset_in_progress:
            fall_mask = torch.rand(len(env_ids), device=self.device) < self.fall_reset_prob
        else:
            fall_mask = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
        self.fall_reset_mask[env_ids] = fall_mask
        self.fall_hold_remaining[env_ids] = 0
        fall_ids = env_ids[fall_mask]
        if fall_ids.numel() > 0:
            self.fall_hold_remaining[fall_ids] = self.fall_hold_steps
            bank_ids = torch.randint(self.fall_state_joint_pos.shape[0],
                                     (fall_ids.numel(),), device=self.device)
            fall_joint_pos = self.fall_state_joint_pos.index_select(0, bank_ids)
            # Keep the settled bank height and tilt, while translating the
            # base onto the sampled reference's x/y location in this env.
            fall_root = self.fall_state_root.index_select(0, bank_ids).clone()
            fall_root[:, 0:2] = reference_root_xy[fall_mask]
            fall_root[:, 2] += self._env.scene.env_origins[fall_ids, 2]
            # A fall starts at rest even if a generator retained terminal
            # velocities for diagnostics: both joint and root velocities are
            # explicitly zeroed at every bank reset.
            fall_root[:, 7:13] = 0.0
            self.robot.write_joint_state_to_sim(fall_joint_pos,
                                                torch.zeros_like(fall_joint_pos),
                                                env_ids=fall_ids)
            self.robot.write_root_state_to_sim(fall_root, env_ids=fall_ids)

    def _update_command(self):
        # time_steps正常情况是按顺序增加，如果完成了一个序列，再重新按失败率采样
        self.time_steps += 1
        # A clip can end during a fallen episode because reset starts from a
        # random frame.  Keep its last frame (all reference properties clamp
        # their indices, including future_steps) until the fall recovery hold
        # expires.  Crucially, filter only the natural clip-switch IDs here:
        # episode reset still enters _resample_command directly through the
        # command manager and always receives a fresh state.
        hold_active = self.fall_hold_remaining > 0
        clip_done = self.time_steps >= self.motion_length
        env_ids = torch.where(clip_done & ~hold_active)[0]
        self._resample_command(env_ids, reset_physical=False)
        if hold_active.any():
            self.fall_hold_remaining[hold_active] -= 1

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(
            1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(
            1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(
            1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:,
                                                              None, :].repeat(
                                                                  1,
                                                                  len(self.cfg.
                                                                      body_names
                                                                      ), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(
            quat_mul(robot_anchor_quat_w_repeat,
                     quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(
            delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        alpha = self.cfg.adaptive_alpha
        if not self.cfg.ads_stable_count:
            self.failed_motion_count = self.cfg.adaptive_alpha * self.current_failed_motion_count + (
                1 - self.cfg.adaptive_alpha) * self.failed_motion_count
            self.success_motion_count = self.cfg.adaptive_alpha * self.current_success_motion_count + (
                1 - self.cfg.adaptive_alpha) * self.success_motion_count
        else:
            tried = (self.current_failed_motion_count +
                     self.current_success_motion_count > 0.5)
            # Only decay the count when Tried.
            self.failed_motion_count = torch.where(
                tried, alpha * self.current_failed_motion_count +
                (1 - alpha) * self.failed_motion_count,
                self.failed_motion_count)
            self.success_motion_count = torch.where(
                tried, alpha * self.current_success_motion_count +
                (1 - alpha) * self.success_motion_count,
                self.success_motion_count)

        total_current_failed_count = self.current_failed_motion_count.sum(
        ).item()
        total_current_success_count = self.current_success_motion_count.sum(
        ).item()
        tried = (total_current_failed_count + total_current_success_count
                 > 0.5)
        if tried:
            self.failed_total_count = alpha * total_current_failed_count + (
                1 - alpha) * self.failed_total_count
            self.success_total_count = alpha * total_current_success_count + (
                1 - alpha) * self.success_total_count
        self.metrics["failed_total_count"][:] = self.failed_total_count
        self.metrics["success_total_count"][:] = self.success_total_count
        self.metrics["pfail_total"][:] = self.failed_total_count / (
            self.failed_total_count + self.success_total_count + 1e-8)

        self.current_failed_motion_count.zero_()
        self.current_success_motion_count.zero_()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                red_color = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8,
                                                                       0.2,
                                                                       0.2))
                green_color = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2,
                                                                         0.8,
                                                                         0.2))
                current_anchor_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/Command/current/anchor",
                    markers={
                        "hit":
                        sim_utils.SphereCfg(
                            radius=0.05,
                            visual_material=red_color,
                        ),
                    },
                )
                goal_anchor_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/Command/goal/anchor",
                    markers={
                        "hit":
                        sim_utils.SphereCfg(
                            radius=0.05,
                            visual_material=green_color,
                        ),
                    },
                )

                self.current_anchor_visualizer = VisualizationMarkers(
                    current_anchor_visualizer_cfg)
                self.goal_anchor_visualizer = VisualizationMarkers(
                    goal_anchor_visualizer_cfg)

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    current_body_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
                        prim_path="/Visuals/Command/current/" + name,
                        markers={
                            "target":
                            sim_utils.SphereCfg(
                                radius=0.02,
                                visual_material=red_color,
                            ),
                        },
                    )
                    goal_body_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
                        prim_path="/Visuals/Command/goal/" + name,
                        markers={
                            "target":
                            sim_utils.SphereCfg(
                                radius=0.02,
                                visual_material=green_color,
                            ),
                        },
                    )
                    self.current_body_visualizers.append(
                        VisualizationMarkers(current_body_visualizer_cfg))
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(goal_body_visualizer_cfg))

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w,
                                                 self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w,
                                              self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(
                self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(
                self.body_pos_relative_w[:, i], self.body_quat_relative_w[:,
                                                                          i])


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING  # type: ignore

    motion_files: list[str] = MISSING  # type: ignore

    anchor_body_name: str = MISSING  # type: ignore
    body_names: list[str] = MISSING  # type: ignore

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    # Future steps configuration for N-step lookahead.
    future_steps: int = 1

    # Start each reference motion at frame zero when enabled.
    start_from_zero_step: bool = False

    # Motion-level adaptive sampling configuration.
    enable_adaptive_sampling: bool = False
    ads_type: str = "v1"
    ads_stable_count: bool = True
    adaptive_length_weighted: bool = False
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001
    adaptive_beta: float = 0.5

    # Probability of turning a sampled reference into a static pose.
    random_static_prob: float = -1.0

    # Number of complete motions retained in the bounded CPU LRU cache.  The
    # read-only textop_npz corpus is uncompressed, so a larger cache avoids
    # repeatedly parsing the same clips while keeping all data off the GPU.
    motion_cache_size: int = 64

    # Probability that a reset starts from a real offline simulated fall state.
    fall_reset_prob: float = 0.35

    # After a fall reset, hold a terminal reference frame for this many
    # seconds if the randomly sampled clip reaches its end first.
    fall_recovery_hold_time: float = 6.0

    # Read-only offline fall-state bank.  The environment variable is useful
    # for the GPU3 container, where the repository is mounted at a different
    # absolute path; the bank itself is never written during training.
    fall_state_path: str = os.environ.get(
        "TEXTOP_FALL_STATE_PATH", "data/textop_fall_states/fall_states_16384.npz"
    )
