#!/usr/bin/env bash
# One container, two synchronized PPO workers on physical GPU0 and GPU1.
# GPU2/GPU3 are owned by the separate textop-zzy-gpu23 run and are never used here.
set -Eeuo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
container=${CONTAINER:-textop-zzy-gpu01}
image=${IMAGE:-textop-remote:latest}
gpu0_uuid=$(nvidia-smi --id=0 --query-gpu=uuid --format=csv,noheader | tr -d '\r')
gpu1_uuid=$(nvidia-smi --id=1 --query-gpu=uuid --format=csv,noheader | tr -d '\r')
[[ "$gpu0_uuid" == GPU-a6ff8437-cdb1-fdc0-55ab-d55479e5d102 ]]
[[ "$gpu1_uuid" == GPU-40cfb5b5-6193-52ae-2c43-d15d568a30ef ]]
run_id=$(TZ=Asia/Shanghai date +%Y-%m-%d_%H-%M-%S)

if docker ps -a --format '{{.Names}}' | grep -Fxq "$container"; then
    echo "Container $container already exists; refusing to start a duplicate." >&2
    exit 1
fi

docker run -d --name "$container" \
    --gpus "\"device=$gpu0_uuid,$gpu1_uuid\"" \
    --shm-size=16g --workdir /workspace/textop \
    --mount type=bind,source=/mnt/robot-whole-body-control-data/zzy/data_g1/textop_npz,target=/workspace/textop_npz,readonly \
    -e NVIDIA_VISIBLE_DEVICES="$gpu0_uuid,$gpu1_uuid" \
    -e PYTHONUNBUFFERED=1 -e OMP_NUM_THREADS=8 \
    -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
    -e TEXTOP_RUN_ID="$run_id" -e TEXTOP_LOG_STAMP="$run_id" \
    -e TEXTOP_RECOVERY_VARIANT=shoulder_only \
    -e TEXTOP_BASE_VELOCITY_PENALTY=1 \
    -e TEXTOP_MOTION_ROOT=/workspace/textop_npz \
    -e TEXTOP_MOTION_MANIFEST=/workspace/textop/data/source/textop_npz_files.json \
    -e TEXTOP_FALL_STATE_PATH=/workspace/textop/data/textop_fall_states/fall_states_16384.npz \
    -e ISAACLAB_PATH=/workspace/textop/isaaclab_minimal \
    -e PYTHONPATH=/workspace/textop/isaaclab_minimal/source/isaaclab:/workspace/textop/isaaclab_minimal/source/isaaclab_assets:/workspace/textop/isaaclab_minimal/source/isaaclab_rl:/workspace/textop/isaaclab_minimal/source/isaaclab_tasks \
    --entrypoint sleep "$image" infinity

container_devices=$(docker inspect "$container" --format '{{json .HostConfig.DeviceRequests}}')
python3 -c 'import json,sys; req=json.loads(sys.argv[1]); ids=[i for r in req for i in (r.get("DeviceIDs") or [])]; assert len(req)==1 and ids==sys.argv[2:], f"Expected only GPU0/GPU1, got {req}"' "$container_devices" "$gpu0_uuid" "$gpu1_uuid"

# Install this checkout into the container before starting the workers.  The
# external motion archive remains mounted read-only.
docker cp "$repo_root/isaaclab_minimal/." "$container:/workspace/textop/isaaclab_minimal/"
docker cp "$repo_root/scripts/." "$container:/workspace/textop/scripts/"
docker cp "$repo_root/assets/." "$container:/workspace/textop/assets/"
docker exec "$container" bash -lc 'mkdir -p /workspace/textop/data/source /workspace/textop/data/textop_fall_states'
docker cp "$repo_root/data/source/textop_npz_files.json" "$container:/workspace/textop/data/source/textop_npz_files.json"
docker cp "$repo_root/data/textop_fall_states/." "$container:/workspace/textop/data/textop_fall_states/"

docker exec -d "$container" bash -lc '
    set -Eeuo pipefail
    unset CUDA_VISIBLE_DEVICES
    test -s "$TEXTOP_FALL_STATE_PATH"
    rm -rf -- /workspace/textop/logs
    mkdir -p /workspace/textop/logs
    log_path="logs/train_textop_gpu01_${TEXTOP_RUN_ID}_base_velocity_threshold.log"
    printf "%s\n" "$log_path" > logs/textop_gpu01_latest.logpath
    echo "$$" > logs/textop_gpu01.pid
    exec /isaac-sim/python.sh -m torch.distributed.run \
        --standalone --nnodes=1 --nproc_per_node=2 \
        --no_python scripts/textop_tracker/distributed_worker.sh \
        --task TextOp-Tracker-G1-ProjGravObs-v0 --headless --num_envs 4096 \
        --distributed --distributed_backend gloo --device cuda:0 \
        --max_iterations 100000 --logger tensorboard --run_name textop_gpu01_base_velocity_threshold \
        --kit_args "--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/physics/sceneMultiGPUMode=0 --/persistent/physics/useActiveCudaContext=false --/validate/p2p/enabled=false --/validate/iommu/enabled=false --/validate/p2p/memoryCheck/enabled=false" \
        > "$log_path" 2>&1
'
printf 'Container: %s\nPhysical GPUs: 0 and 1\nPer-GPU environments: 4096\nTotal environments: 8192\nRecovery variant: shoulder_only\nBase velocity threshold penalty: enabled\nLog: /workspace/textop/logs/train_textop_gpu01_%s_base_velocity_threshold.log\n' "$container" "$run_id"
