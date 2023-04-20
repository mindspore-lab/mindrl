# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Dreamer config.
"""
import mindspore as ms

from mindspore_rl.algorithm.dreamer.dreamer_replaybuffer import DreamerReplayBuffer
from mindspore_rl.environment import DeepMindControlEnvironment

from .dreamer import DreamerActor, DreamerLearner, DreamerPolicy

BATCH_SIZE = 50
# conv encoder and decoder
all_params = {
    "depth": 32,
    "stride_conv_encoder": 2,
    "stride_conv_decoder": 2,
    "conv_decoder_shape": (64, 64, 3),
    # reward decoder
    "reward_decoder_shape": (),
    "reward_decoder_layers": [238, 400, 400],
    # value decoder
    "value_decoder_shape": (),
    "value_decoder_layers": [238, 400, 400, 400],
    "action_decoder_layers": [238, 400, 400, 400, 400],
    "size": 6,
    "min_std": 1e-4,
    # action decoder
    "init_std": 5.0,
    "mean_scale": 5,
    # RSSM
    "stoch_size": 30,
    "hidden_size": 200,
    "deter_size": 208,
    # Actor
    "expl_amount": 0.3,
    # dynamic loss cell
    "free_nats": 3.0,
    "kl_scale": 1.0,
    # actor loss cell
    "discount": 0.99,
    "horizon": 15,
    "gamma": 0.95,
    "dtype": ms.float32,
    # Env
    "env_name": "walker_walk",
    "img_size": (64, 64),
    "action_repeat": 2,
    "normalize_action": True,
    "seed": 1,
    "episode_limits": 1000,
    "prefill_value": 5000,
    # Learner LR
    "dynamic_lr": 6e-4,
    "actor_lr": 8e-5,
    "value_lr": 8e-5,
    "train_steps": 100,
    "batch_size": BATCH_SIZE,
    "clip": 100,
    "num_save_episode": 100,
    "ckpt_path": "./ckpt",
}

collect_env_params = all_params
eval_env_params = all_params

policy_params = all_params

learner_params = all_params

trainer_part = {
    "duration": 1000,
    "batch_size": 1,
    "ckpt_path": "./ckpt",
    "num_eval_episode": 30,
    "num_save_episode": 50,
}
trainer_params = dict(trainer_part, **all_params)

algorithm_config = {
    "actor": {
        "number": 1,
        "type": DreamerActor,
        "params": all_params,
        "policies": ["init_policy", "collect_policy", "action_decoder"],
    },
    "learner": {
        "number": 1,
        "type": DreamerLearner,
        "params": all_params,
        "networks": [
            "conv_encoder",
            "conv_decoder",
            "rssm",
            "reward_decoder",
            "action_decoder",
            "value_decoder",
        ],
    },
    "policy_and_network": {"type": DreamerPolicy, "params": all_params},
    "collect_environment": {
        "number": 1,
        "type": DeepMindControlEnvironment,
        "params": collect_env_params,
    },
    "eval_environment": {
        "number": 1,
        "type": DeepMindControlEnvironment,
        "params": eval_env_params,
    },
    "replay_buffer": {
        "number": 1,
        "type": DreamerReplayBuffer,
        "capacity": 10,
        "sample_size": BATCH_SIZE,
    },
}
