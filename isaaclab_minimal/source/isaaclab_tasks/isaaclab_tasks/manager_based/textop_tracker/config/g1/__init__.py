"""Gym registrations for the custom-asset G1 TextOp Tracker."""
import gymnasium as gym
from . import agents, flat_env_cfg


def _register(task_id, cfg):
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": cfg,
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
        },
    )


_register("TextOp-Tracker-G1-v0", flat_env_cfg.G1FlatEnvCfg)
_register("TextOp-Tracker-G1-ProjGravObs-v0", flat_env_cfg.G1FlatProjGravObsEnvCfg)
