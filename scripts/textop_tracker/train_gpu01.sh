#!/usr/bin/env bash
# One container, two synchronized PPO workers on physical GPU0 and GPU1.
# This launcher is intentionally separate from train_gpu23.sh so the existing
# GPU2/GPU3 TextOp run is left untouched.
set -Eeuo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
container=${CONTAINER:-stablemimic-zzy-gpu01-textop}
image=${IMAGE:-stablemimic-textop-remote:latest}
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
    --shm-size=16g --workdir /workspace/stablemimic \
    --mount type=bind,source=/mnt/robot-whole-body-control-data/zzy/data_g1/textop_npz,target=/workspace/textop_npz,readonly \
    -e NVIDIA_VISIBLE_DEVICES="$gpu0_uuid,$gpu1_uuid" \
    -e PYTHONUNBUFFERED=1 -e OMP_NUM_THREADS=8 \
    -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
    -e TEXTOP_RUN_ID="$run_id" -e TEXTOP_LOG_STAMP="$run_id" \
    -e TEXTOP_RECOVERY_VARIANT=shoulder_only \
    -e TEXTOP_MOTION_ROOT=/workspace/textop_npz \
    -e TEXTOP_MOTION_MANIFEST=/workspace/stablemimic/data/source/textop_npz_files.json \
    -e TEXTOP_FALL_STATE_PATH=/workspace/stablemimic/data/textop_fall_states/fall_states_16384.npz \
    --entrypoint sleep "$image" infinity

container_devices=$(docker inspect "$container" --format '{{json .HostConfig.DeviceRequests}}')
python3 -c 'import json,sys; req=json.loads(sys.argv[1]); ids=[i for r in req for i in (r.get("DeviceIDs") or [])]; assert len(req)==1 and ids==sys.argv[2:], f"Expected only GPU0/GPU1, got {req}"' "$container_devices" "$gpu0_uuid" "$gpu1_uuid"

# The image contains an older snapshot: install this checked-out source before
# starting the workers, while keeping the read-only motion source external.
docker cp "$repo_root/isaaclab_minimal/." "$container:/workspace/stablemimic/isaaclab_minimal/"
docker cp "$repo_root/scripts/." "$container:/workspace/stablemimic/scripts/"
docker cp "$repo_root/assets/." "$container:/workspace/stablemimic/assets/"
docker cp "$repo_root/data/source/textop_npz_files.json" "$container:/workspace/stablemimic/data/source/textop_npz_files.json"
docker cp "$repo_root/data/textop_fall_states/." "$container:/workspace/stablemimic/data/textop_fall_states/"

docker exec -d "$container" bash -lc '
    set -Eeuo pipefail
    unset CUDA_VISIBLE_DEVICES
    test -s "$TEXTOP_FALL_STATE_PATH"
    rm -rf -- /workspace/stablemimic/logs
    mkdir -p /workspace/stablemimic/logs
    log_path="logs/train_textop_gpu01_${TEXTOP_RUN_ID}.log"
    printf "%s\n" "$log_path" > logs/textop_gpu01_latest.logpath
    echo "$$" > logs/textop_gpu01.pid
    exec /isaac-sim/python.sh -m torch.distributed.run \
        --standalone --nnodes=1 --nproc_per_node=2 \
        --no_python scripts/textop_tracker/distributed_worker.sh \
        --task TextOp-Tracker-G1-ProjGravObs-v0 --headless --num_envs 4096 \
        --distributed --distributed_backend gloo --device cuda:0 \
        --max_iterations 100000 --logger tensorboard --run_name textop_gpu01_shoulder_only \
        --kit_args "--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/physics/sceneMultiGPUMode=0 --/persistent/physics/useActiveCudaContext=false --/validate/p2p/enabled=false --/validate/iommu/enabled=false --/validate/p2p/memoryCheck/enabled=false" \
        > "$log_path" 2>&1
'
printf 'Container: %s\nPhysical GPUs: 0 and 1\nPer-GPU environments: 4096\nTotal environments: 8192\nLog: /workspace/stablemimic/logs/train_textop_gpu01_%s.log\n' "$container" "$run_id"
