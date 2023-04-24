"""c51 config"""
from mindspore_rl.core.uniform_replay_buffer import UniformReplayBuffer
from mindspore_rl.environment import GymEnvironment
from mindspore_rl.environment.pyfunc_wrapper import PyFuncWrapper

from .c51 import CategoricalDQNActor, CategoricalDQNLearner, CategoricalDQNPolicy

categorical_params = {"atoms_num": 51, "v_min": -10, "v_max": 10}
learner_params = {"gamma": 0.99, "lr": 0.001}
learner_params.update(categorical_params)
trainer_params = {
    "num_evaluate_episode": 10,
    "ckpt_path": "./ckpt",
    "save_per_episode": 50,
    "eval_per_episode": 10,
    "td_step": 1,
}

collect_env_params = {"name": "CartPole-v0"}
eval_env_params = {"name": "CartPole-v0"}

policy_params = {
    "epsi_high": 0.1,
    "epsi_low": 0.1,
    "decay": 200,
    "state_space_dim": 0,
    "action_space_dim": 0,
    "hidden_size": 100,
}
policy_params.update(categorical_params)
algorithm_config = {
    "actor": {
        "number": 1,
        "type": CategoricalDQNActor,
        "policies": ["init_policy", "collect_policy", "evaluate_policy"],
    },
    "learner": {
        "number": 1,
        "type": CategoricalDQNLearner,
        "params": learner_params,
        "networks": ["policy_network", "target_network"],
    },
    "policy_and_network": {"type": CategoricalDQNPolicy, "params": policy_params},
    "collect_environment": {
        "number": 1,
        "type": GymEnvironment,
        "wrappers": [PyFuncWrapper],
        "params": collect_env_params,
    },
    "eval_environment": {
        "number": 1,
        "type": GymEnvironment,
        "wrappers": [PyFuncWrapper],
        "params": eval_env_params,
    },
    "replay_buffer": {
        "number": 1,
        "type": UniformReplayBuffer,
        "capacity": 100000,
        "sample_size": 64,
    },
}
