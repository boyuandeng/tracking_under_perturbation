# TextOp training

The default task is `TextOp-Tracker-G1-v0`; `TextOp-Tracker-G1-ProjGravObs-v0` is its projected-gravity observation variant. Both use the bundled custom G1 29-DOF USD. The dual-GPU launcher sets `TEXTOP_MOTION_ROOT=/workspace/textop_npz` to read the source corpus directly.

The current configuration uses 4096 environments, a 0.005 s simulation step with decimation 4 (50 Hz control), 10 s episodes, 24 rollout steps, `[512, 256, 128]` ELU actor and critic networks, observation normalization, and 100000 PPO iterations. Checkpoints are saved every 500 iterations. Use TensorBoard at `logs/rsl_rl/textop_tracker_g1/`.

## GPU rule

Never stop, restart, reconfigure, or otherwise operate on GPU0 or GPU1 jobs. The current TextOp run uses physical GPU2 and GPU3 in one synchronized container; no process is launched on GPU0 or GPU1.

The `stablemimic-zzy-gpu23-textop` container is restricted by Docker to GPU UUIDs `GPU-7fd956bf-4105-d4c6-f3fa-4b4d0fb0c327` (physical GPU2) and `GPU-82145392-6b7e-9e5f-ec15-f63c44044fa7` (physical GPU3). Launch it from the host checkout:

```bash
scripts/textop_tracker/train_gpu23.sh
```

The launcher copies this checkout into the container before training, because the image contains an older code snapshot. It starts two synchronized PPO workers with 4096 environments each, for 8192 total. `distributed_worker.sh` maps each worker to one of the two container GPUs and preserves global rank for policy synchronization and independent seeds. Each worker sees logical `cuda:0`; physical GPU0/GPU1 remain excluded by Docker.

Inside the container, the equivalent training command is:

```bash
/isaac-sim/python.sh -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=2 \
  --no_python scripts/textop_tracker/distributed_worker.sh \
  --task TextOp-Tracker-G1-v0 --distributed --distributed_backend gloo \
  --headless --num_envs 4096 --device cuda:0 --max_iterations 100000 \
  --logger tensorboard --run_name textop_gpu23_hold3 \
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

To inspect the active run with TensorBoard, use the helper below. It defaults to the dedicated TextOp container; override `REMOTE` or `CONTAINER` only when intentionally viewing another run:

```bash
scripts/textop_tracker/deploy_tensorboards_textop.sh
```

Training reads the existing 50 Hz NPZ corpus directly from the read-only source mount; it does not convert or regenerate the dataset during training. The fall bank is a separate reset-state artifact and is never used as a replacement reference trajectory.

The maintained converter is `scripts/textop_tracker/convert_g1_dataset.py`. Inspect source metadata without starting Isaac Sim:

```bash
/isaac-sim/python.sh scripts/textop_tracker/convert_g1_dataset.py \
  --input data/source/zzy_g1_lafan1 --output /tmp/textop_inventory --inventory
```
