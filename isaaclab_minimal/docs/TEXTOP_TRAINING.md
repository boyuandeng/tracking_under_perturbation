# TextOp training

The only registered task is `TextOp-Tracker-G1-ProjGravObs-v0`, which uses the bundled custom G1 29-DOF USD and includes the 3-D projected-gravity observation for both policy and critic. The dual-GPU launchers set `TEXTOP_MOTION_ROOT=/workspace/textop_npz` to read the source corpus directly.

The current configuration uses 4096 environments, a 0.005 s simulation step with decimation 4 (50 Hz control), 10 s episodes, 24 rollout steps, `[512, 256, 128]` ELU actor and critic networks, observation normalization, and 100000 PPO iterations. Checkpoints are saved every 2000 iterations. Use TensorBoard at `logs/rsl_rl/textop_tracker_g1/`.

## GPU rule

The user authorized two separate synchronized dual-GPU TextOp runs. Manage only the named TextOp containers below and leave unrelated GPU jobs untouched. All new container, image, task, and log names use `textop`.

| Container | Physical GPUs | Launcher | Recovery rewards |
| --- | --- | --- | --- |
| `textop-zzy-gpu01` | 0, 1 | `scripts/textop_tracker/train_gpu01.sh` | `shoulder_only` + base velocity threshold |
| `textop-zzy-gpu23` | 2, 3 | `scripts/textop_tracker/train_gpu23.sh` | `shoulder_only` without base velocity threshold |

Both runs use only `TextOp-Tracker-G1-ProjGravObs-v0`, image `textop-remote:latest`, workdir `/workspace/textop`, and this checkout. GPU0/1 sets `TEXTOP_RECOVERY_VARIANT=shoulder_only`, enables `TEXTOP_BASE_VELOCITY_PENALTY=1`, and runs `textop_gpu01_base_velocity_threshold`; GPU2/3 sets the same recovery variant with `TEXTOP_BASE_VELOCITY_PENALTY=0` and runs `textop_gpu23_no_base_velocity_threshold`. Each run starts two synchronized PPO workers with 4096 environments each, for 8192 total. `distributed_worker.sh` maps each worker to one container GPU and preserves global rank for policy synchronization and independent seeds; each worker sees logical `cuda:0`.

Both previous TextOp containers were stopped and removed before this two-arm ablation was launched. Keep the GPU pairs isolated.

Both current runs use `shoulder_only`, so root orientation and XY motion are disabled. The only ablation is the base velocity threshold term.

| Recovery term | Weight | Quantity before weight |
| --- | --- | --- |
| `recovery_shoulder_height` | -2.0 | `(1-g) * sum(shoulder_height_error**2)` |
| `recovery_second_action_rate` | -0.1 | `(1-g) * sum((a_t - 2*a_prev + a_prev_prev)**2)` |
| `recovery_joint_velocity` | -0.1 | `(1-g) * mean(relu(abs(joint_velocity)-6)^2 / 6^2)` |
| `recovery_base_velocity_threshold` | -0.1 (GPU0/1 only) | `(1-g) * [relu(v_xy-0.5)^2/0.5^2 + relu(|w|-1.5)^2/1.5^2]` |

Here `g = exp(-mean(shoulder_height_error**2) / 0.25**2)`. Recovery tolerance and the clip-end hold are both 6 seconds; the reference continues rolling during recovery. The six positive tracking rewards use the same gate formula with sigma 0.1. The ordinary first action-rate penalty (-0.1) remains active alongside these recovery terms.

```bash
scripts/textop_tracker/train_gpu01.sh
scripts/textop_tracker/train_gpu23.sh
```

The launchers refuse to create a duplicate container. Do not run either launcher to replace an active run without user authorization. Inside the GPU2/3 container, the equivalent training command is:

```bash
/isaac-sim/python.sh -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=2 \
  --no_python scripts/textop_tracker/distributed_worker.sh \
  --task TextOp-Tracker-G1-ProjGravObs-v0 --distributed --distributed_backend gloo \
  --headless --num_envs 4096 --device cuda:0 --max_iterations 100000 \
  --logger tensorboard --run_name textop_gpu23_no_base_velocity_threshold \
  --kit_args "--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/physics/sceneMultiGPUMode=0 --/persistent/physics/useActiveCudaContext=false --/validate/p2p/enabled=false --/validate/iommu/enabled=false --/validate/p2p/memoryCheck/enabled=false"
```

Gloo stages CUDA tensors through CPU for parameter, KL, learning-rate, and gradient synchronization. This avoids the NCCL/PhysX CUDA failures observed on this host, with extra CPU transfer overhead. Only rank0 writes TensorBoard events, configuration, and checkpoints into the shared run directory.

## Data

The fall/recovery reset states are generated once into a local, immutable bank. The source archive at `/mnt/robot-whole-body-control-data/zzy/data_g1/textop_npz` is mounted read-only; the output bank must be written under the repository or another writable local path. From the TextOp container, generate 16,384 states with 3--4 seconds of zero-PD (zero stiffness and damping) physics per sampled motion frame:

```bash
mkdir -p data/textop_fall_states
/isaac-sim/python.sh scripts/textop_tracker/generate_fall_states.py \
  --input /workspace/textop_npz \
  --output data/textop_fall_states/fall_states_16384.npz \
  --count 16384 --num-envs 512 --duration-min 3 --duration-max 4 \
  --device cuda:0
```

Training loads this bank once at startup through `TEXTOP_FALL_STATE_PATH`; both launchers set `/workspace/textop/data/textop_fall_states/fall_states_16384.npz`. At every reset, 35% of environments sample a bank state and 65% use the normal reset. For a fall reset, the bank base XY is translated to the unperturbed pelvis XY of the first frame in the newly sampled reference window; normal-start pose jitter is not applied to this translation. Bank resets explicitly set all joint velocities to zero; the reference trajectory still advances normally. Sampling a bank entry is an in-memory tensor gather, so resets do not rerun physics or decompress an NPZ file.

To inspect the active run with TensorBoard, use the helper below with `CONTAINER=textop-zzy-gpu23`. Use `CONTAINER=textop-zzy-gpu01` to inspect the GPU0/1 base velocity threshold run; `REMOTE` selects the SSH host:

```bash
CONTAINER=textop-zzy-gpu23 scripts/textop_tracker/deploy_tensorboards_textop.sh
```

Training reads the existing 50 Hz NPZ corpus directly from the read-only source mount; it does not convert or regenerate the dataset during training. The fall bank is a separate reset-state artifact and is never used as a replacement reference trajectory.

The maintained converter is `scripts/textop_tracker/convert_g1_dataset.py`. Inspect source metadata without starting Isaac Sim:

```bash
/isaac-sim/python.sh scripts/textop_tracker/convert_g1_dataset.py \
  --input data/source/zzy_g1_lafan1 --output /tmp/textop_inventory --inventory
```
