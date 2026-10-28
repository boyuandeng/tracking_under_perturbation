# TextOp training

The only registered task is `TextOp-Tracker-G1-ProjGravObs-v0`, which uses the bundled custom G1 29-DOF USD and includes the 3-D projected-gravity observation for both policy and critic. The dual-GPU launchers set `TEXTOP_MOTION_ROOT=/workspace/textop_npz` to read the source corpus directly.

The current configuration uses 4096 environments, a 0.005 s simulation step with decimation 4 (50 Hz control), 10 s episodes, 24 rollout steps, `[512, 256, 128]` ELU actor and critic networks, observation normalization, and 100000 PPO iterations. Checkpoints are saved every 2000 iterations. Use TensorBoard at `logs/rsl_rl/textop_tracker_g1/`.

## GPU rule

The user authorized two separate synchronized dual-GPU TextOp runs. Manage only the named TextOp containers below and leave unrelated GPU jobs untouched.

| Container | Physical GPUs | Launcher | Recovery rewards |
| --- | --- | --- | --- |
| `stablemimic-zzy-gpu01-textop` | 0, 1 | `scripts/textop_tracker/train_gpu01.sh` | `shoulder_only`: shoulder height, weight -2.0 |
| `stablemimic-zzy-gpu23-textop` | 2, 3 | `scripts/textop_tracker/train_gpu23.sh` | `full_recovery`: shoulder height -2.0, root orientation -1.0, XY motion -1.0 |

Both runs use only `TextOp-Tracker-G1-ProjGravObs-v0`. The launchers select the recovery terms through `TEXTOP_RECOVERY_VARIANT`, bind each container to its two physical GPU UUIDs, and copy this checkout into the container before training. Each run starts two synchronized PPO workers with 4096 environments each, for 8192 total. `distributed_worker.sh` maps each worker to one container GPU and preserves global rank for policy synchronization and independent seeds; each worker sees logical `cuda:0`.

```bash
scripts/textop_tracker/train_gpu01.sh
scripts/textop_tracker/train_gpu23.sh
```

Inside the container, the equivalent training command is:

```bash
/isaac-sim/python.sh -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=2 \
  --no_python scripts/textop_tracker/distributed_worker.sh \
  --task TextOp-Tracker-G1-ProjGravObs-v0 --distributed --distributed_backend gloo \
  --headless --num_envs 4096 --device cuda:0 --max_iterations 100000 \
  --logger tensorboard --run_name textop_gpu23_full_recovery \
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

Training loads this bank once at startup through `TEXTOP_FALL_STATE_PATH` (the dual-GPU launcher defaults to `/workspace/stablemimic/data/textop_fall_states/fall_states_16384.npz`). At every reset, 35% of environments sample a bank state and 65% use the normal reset. For a fall reset, the bank base XY is translated to the unperturbed pelvis XY of the first frame in the newly sampled reference window; normal-start pose jitter is not applied to this translation. Bank resets explicitly set all joint velocities to zero; the reference trajectory still advances normally. Sampling a bank entry is an in-memory tensor gather, so resets do not rerun physics or decompress an NPZ file.

To inspect the active run with TensorBoard, use the helper below. It defaults to the GPU2/3 TextOp container. Use `CONTAINER=stablemimic-zzy-gpu01-textop` to inspect the shoulder-only run; `REMOTE` selects the SSH host:

```bash
scripts/textop_tracker/deploy_tensorboards_textop.sh
```

Training reads the existing 50 Hz NPZ corpus directly from the read-only source mount; it does not convert or regenerate the dataset during training. The fall bank is a separate reset-state artifact and is never used as a replacement reference trajectory.

The maintained converter is `scripts/textop_tracker/convert_g1_dataset.py`. Inspect source metadata without starting Isaac Sim:

```bash
/isaac-sim/python.sh scripts/textop_tracker/convert_g1_dataset.py \
  --input data/source/zzy_g1_lafan1 --output /tmp/textop_inventory --inventory
```
