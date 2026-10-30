# GPU operating rule

The user authorized two TextOp training containers using all four physical GPUs. Keep each run isolated to its assigned pair, and do not stop, restart, reconfigure, or otherwise modify unrelated GPU jobs.

- `textop-zzy-gpu01`: GPU0 UUID `GPU-a6ff8437-cdb1-fdc0-55ab-d55479e5d102` and GPU1 UUID `GPU-40cfb5b5-6193-52ae-2c43-d15d568a30ef`; image `textop-remote:latest`, workdir `/workspace/textop`, `shoulder_only`, base velocity threshold enabled, run name `textop_gpu01_base_velocity_threshold`.
- `textop-zzy-gpu23`: GPU2 UUID `GPU-7fd956bf-4105-d4c6-f3fa-4b4d0fb0c327` and GPU3 UUID `GPU-82145392-6b7e-9e5f-ec15-f63c44044fa7`; image `textop-remote:latest`, workdir `/workspace/textop`, `shoulder_only` recovery, base velocity threshold disabled, run name `textop_gpu23_no_base_velocity_threshold`.

Both current runs use shoulder height (-2.0), second action rate (-0.1), and joint velocity above 6 rad/s (-0.1). Joint acceleration is removed. GPU0/1 additionally enables the base horizontal/ angular velocity threshold term (-0.1); GPU2/3 disables it. Recovery tolerance and clip-end hold are 6 seconds, and reference keeps rolling. All new container, image, task, and log names must use `textop`.

Both previous TextOp containers were stopped and removed before this two-arm ablation. Keep GPU pairs isolated and manage only the named TextOp containers.

Both launchers use only `TextOp-Tracker-G1-ProjGravObs-v0`, with two synchronized torchrun workers per container and 4096 environments per worker. They use `--distributed --distributed_backend gloo`; `distributed_worker.sh` isolates each worker to one container GPU and resets `LOCAL_RANK=0` while preserving global `RANK`. Each worker uses logical `cuda:0`, a distinct global-rank seed, and one shared policy. Keep Kit renderer/physics scene multi-GPU and IOMMU/P2P validation disabled as specified by the launchers.

Do not use broad process-kill commands, GPU resets, unrelated container restarts, or shared dependency changes to prepare training. The source corpus under `/mnt` is read-only. Keep only current training logs and remove temporary test scripts/logs after validation.
