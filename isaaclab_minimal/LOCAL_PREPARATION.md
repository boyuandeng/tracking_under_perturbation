# Runtime preparation

- Upstream Isaac Lab v2.3.2 commit: `37ddf626871758333d6ed89cf64ad702aef127d0`.
- The task package contains only the TextOp G1 tracker and two registered observation variants.
- The core runtime retains interconnected modules. The optional TacSL scene dispatch was removed so `isaaclab_contrib` is not required.
- Local G1 USD dependencies are bundled with the task. Motion data are separate under `data/textop_tracker/`.
- The existing GPU3 container provides Python 3.11, Isaac Sim, Torch 2.7.0+cu128, RSL-RL 3.1.2, Gymnasium 1.2.1, and Hydra 1.3.2.
- `prepare_offline_runtime.sh` installs local packages using `--no-index --no-deps --no-build-isolation`.

Current launch instructions and the GPU0–2 protection rule are in `docs/TEXTOP_TRAINING.md`.
