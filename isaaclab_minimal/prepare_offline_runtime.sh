#!/usr/bin/env bash
# Run inside the existing Isaac Lab 2.3.2 Docker image. No package downloads.
set -euo pipefail
subset_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
isaac_sim_root="${ISAACSIM_ROOT:-/isaac-sim}"
if [[ ! -e "$subset_root/_isaac_sim" ]]; then
    ln -s "$isaac_sim_root" "$subset_root/_isaac_sim"
fi
for package in isaaclab isaaclab_assets isaaclab_rl isaaclab_tasks; do
    "$isaac_sim_root/python.sh" -m pip install --no-index --no-deps --no-build-isolation -e "$subset_root/source/$package"
done
