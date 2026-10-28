# GPU operating rule

The user authorized two TextOp training containers using all four physical GPUs. Operate only on these named TextOp containers; do not stop, restart, reconfigure, or otherwise modify unrelated GPU jobs.

- `stablemimic-zzy-gpu01-textop`: GPU0 UUID `GPU-a6ff8437-cdb1-fdc0-55ab-d55479e5d102` and GPU1 UUID `GPU-40cfb5b5-6193-52ae-2c43-d15d568a30ef`; `shoulder_only` recovery.
- `stablemimic-zzy-gpu23-textop`: GPU2 UUID `GPU-7fd956bf-4105-d4c6-f3fa-4b4d0fb0c327` and GPU3 UUID `GPU-82145392-6b7e-9e5f-ec15-f63c44044fa7`; `full_recovery`.

Both launchers use only `TextOp-Tracker-G1-ProjGravObs-v0`, with two synchronized torchrun workers per container and 4096 environments per worker. They use `--distributed --distributed_backend gloo`; `distributed_worker.sh` isolates each worker to one container GPU and resets `LOCAL_RANK=0` while preserving global `RANK`. Each worker uses logical `cuda:0`, a distinct global-rank seed, and one shared policy. Keep Kit renderer/physics scene multi-GPU and IOMMU/P2P validation disabled as specified by the launchers.

Do not use broad process-kill commands, GPU resets, unrelated container restarts, or shared dependency changes to prepare training. The source corpus under `/mnt` is read-only. Keep only current training logs and remove temporary test scripts/logs after validation.
