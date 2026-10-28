#!/usr/bin/env bash
# One container, two synchronized PPO workers; physical GPU0/GPU1 are excluded.
set -Eeuo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
container=stablemimic-zzy-gpu23-textop
image=${IMAGE:-stablemimic-textop-remote:latest}
gpu2_uuid=$(nvidia-smi --id=2 --query-gpu=uuid --format=csv,noheader)
gpu3_uuid=$(nvidia-smi --id=3 --query-gpu=uuid --format=csv,noheader)
[[ "$gpu2_uuid" == GPU-7fd956bf-4105-d4c6-f3fa-4b4d0fb0c327 ]]
[[ "$gpu3_uuid" == GPU-82145392-6b7e-9e5f-ec15-f63c44044fa7 ]]
run_id=$(TZ=Asia/Shanghai date +%Y-%m-%d_%H-%M-%S)

if docker ps -a --format '{{.Names}}' | grep -Fxq "$container"; then
    echo "Container $container already exists; refusing to start a duplicate." >&2
    exit 1
fi

docker run -d --name "$container" \
    --gpus "\"device=$gpu2_uuid,$gpu3_uuid\"" \
    --shm-size=16g --workdir /workspace/stablemimic \
    --mount type=bind,source=/mnt/robot-whole-body-control-data/zzy/data_g1/textop_npz,target=/workspace/textop_npz,readonly \
    -e NVIDIA_VISIBLE_DEVICES="$gpu2_uuid,$gpu3_uuid" \
    -e PYTHONUNBUFFERED=1 -e OMP_NUM_THREADS=8 \
    -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
    -e TEXTOP_RUN_ID="$run_id" -e TEXTOP_LOG_STAMP="$run_id" \
    -e TEXTOP_MOTION_ROOT=/workspace/textop_npz \
    -e TEXTOP_MOTION_MANIFEST=/workspace/stablemimic/data/source/textop_npz_files.json \
    -e TEXTOP_FALL_STATE_PATH=/workspace/stablemimic/data/textop_fall_states/fall_states_16384.npz \
    --entrypoint sleep "$image" infinity

container_devices=$(docker inspect "$container" --format '{{json .HostConfig.DeviceRequests}}')
python3 -c 'import json,sys; req=json.loads(sys.argv[1]); ids=[i for r in req for i in (r.get("DeviceIDs") or [])]; assert len(req)==1 and ids==sys.argv[2:], f"Expected only GPU2/GPU3, got {req}"' "$container_devices" "$gpu2_uuid" "$gpu3_uuid"
# The image contains an older snapshot: always install this checkout before training.
docker cp "$repo_root/isaaclab_minimal/." "$container:/workspace/stablemimic/isaaclab_minimal/"
docker cp "$repo_root/scripts/." "$container:/workspace/stablemimic/scripts/"
docker cp "$repo_root/assets/." "$container:/workspace/stablemimic/assets/"
docker cp "$repo_root/data/source/textop_npz_files.json" "$container:/workspace/stablemimic/data/source/textop_npz_files.json"
docker cp "$repo_root/data/textop_fall_states/." "$container:/workspace/stablemimic/data/textop_fall_states/"

docker exec -d "$container" bash -lc '
    set -Eeuo pipefail
    unset CUDA_VISIBLE_DEVICES
    test -s "$TEXTOP_FALL_STATE_PATH"
    # This is a newly created, idle container; discard logs baked into the image.
    rm -rf -- /workspace/stablemimic/logs
    mkdir -p /workspace/stablemimic/logs
    log_path="logs/train_textop_gpu23_${TEXTOP_RUN_ID}.log"
    printf "%s\n" "$log_path" > logs/textop_gpu23_latest.logpath
    echo "$$" > logs/textop_gpu23.pid
    exec /isaac-sim/python.sh -m torch.distributed.run \
        --standalone --nnodes=1 --nproc_per_node=2 \
        --no_python scripts/textop_tracker/distributed_worker.sh \
        --task TextOp-Tracker-G1-v0 --headless --num_envs 4096 \
        --distributed --distributed_backend gloo --device cuda:0 \
        --max_iterations 100000 --logger tensorboard --run_name textop_gpu23_hold3 \
        --kit_args "--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/physics/sceneMultiGPUMode=0 --/persistent/physics/useActiveCudaContext=false --/validate/p2p/enabled=false --/validate/iommu/enabled=false --/validate/p2p/memoryCheck/enabled=false" \
        > "$log_path" 2>&1
'
printf 'Container: %s\nPhysical GPUs: 2 and 3\nPer-GPU environments: 4096\nTotal environments: 8192\nLog: /workspace/stablemimic/logs/train_textop_gpu23_%s.log\n' "$container" "$run_id"
