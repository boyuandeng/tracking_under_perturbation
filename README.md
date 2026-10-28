# TextOp G1 Tracker

This workspace contains one motion tracker, `textop_tracker`, plus the Isaac Lab 2.3.2 runtime and RSL-RL train/play entry points it needs.

Registered environment:

- `TextOp-Tracker-G1-ProjGravObs-v0`: TextOp actor/privileged critic observations with 3-D projected gravity.

The active training corpus is `/mnt/robot-whole-body-control-data/zzy/data_g1/textop_npz`, mounted read-only at `/workspace/textop_npz` in the training container. Motion files are read directly without online conversion. The local 16,384-state fall bank is loaded once, with a 35% fall-reset probability. The supplied G1 USD is shipped with the task.

Training defaults are 4096 environments, 50 Hz policy control, 24 rollout steps, actor/critic layers `[512, 256, 128]`, ELU, 5 PPO epochs, 4 minibatches, adaptive learning rate 0.001, and 100000 iterations. Checkpoints are saved every 2000 iterations under `logs/rsl_rl/textop_tracker_g1/`.

See [the runtime guide](isaaclab_minimal/docs/TEXTOP_TRAINING.md) for launch commands. Two synchronized dual-GPU runs use the same ProjGrav task: GPU0/1 use `shoulder_only` recovery, and GPU2/3 use `full_recovery`. Each GPU runs 4096 environments (8192 per run). Only these two TextOp containers are authorized for training management; leave other GPU jobs untouched.
