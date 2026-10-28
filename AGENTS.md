# GPU operating rule

GPU0 and GPU1 jobs are reserved and must never be stopped, restarted, reconfigured, or otherwise modified. Do not launch new work on them. The current TextOp run is explicitly authorized to use physical GPUs 2 and 3 together in one synchronized container.

Run TextOp in the dual-GPU container restricted to physical GPU2 UUID `GPU-7fd956bf-4105-d4c6-f3fa-4b4d0fb0c327` and GPU3 UUID `GPU-82145392-6b7e-9e5f-ec15-f63c44044fa7`. The container exposes only these two GPUs. The supported launcher uses two torchrun workers with `--distributed --distributed_backend gloo`; `distributed_worker.sh` isolates each worker to one container GPU and resets `LOCAL_RANK=0` while preserving global `RANK`. Each worker uses logical `cuda:0`, a distinct global-rank seed, and 4096 environments. Gloo stages PPO collectives through CPU and synchronizes one shared policy. Keep Kit renderer/physics scene multi-GPU and IOMMU/P2P validation disabled as specified by the launcher.

Do not use broad process-kill commands, GPU resets, container restarts, or shared dependency changes to prepare this training.
