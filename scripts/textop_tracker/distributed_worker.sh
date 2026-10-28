#!/usr/bin/env bash
# torchrun worker wrapper: isolate each Isaac Sim process to its assigned GPU.
# The parent container exposes only physical GPUs 2 and 3.  RANK/WORLD_SIZE
# remain untouched for RSL-RL synchronization; LOCAL_RANK is reset to zero
# because each worker now sees exactly one CUDA device.
set -Eeuo pipefail
rank=${LOCAL_RANK:?LOCAL_RANK must be set by torchrun}
case "$rank" in
    0|1) ;;
    *) echo "Unexpected local rank: $rank" >&2; exit 2 ;;
esac
export CUDA_VISIBLE_DEVICES="$rank"
export LOCAL_RANK=0
exec /isaac-sim/python.sh isaaclab_minimal/scripts/reinforcement_learning/rsl_rl/train.py "$@"
