# Isaac Lab v2.3.2 runtime for TextOp

This subset retains the Isaac Lab runtime core, App experiences, G1 assets, RSL-RL bindings, task registry/Hydra utilities, and standard train/play scripts.

The only tracker implementation is `source/isaaclab_tasks/isaaclab_tasks/manager_based/textop_tracker/`. It registers `TextOp-Tracker-G1-v0` and `TextOp-Tracker-G1-ProjGravObs-v0`. Both use `G1FlatPPORunnerCfg` and the official RSL-RL entry point.

The core runtime is retained because of interconnected imports. Core camera, XR, mimic, and other generic utilities do not register additional learning tasks.

See [TextOp training](docs/TEXTOP_TRAINING.md) for the current data, configuration, and GPU2/GPU3 distributed launch instructions.
