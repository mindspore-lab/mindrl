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
"""COMA Trainer"""

import mindspore as ms
import numpy as np
from mindspore import Parameter, Tensor, nn
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.ops.primitive import constexpr

from mindspore_rl.agent import Trainer, trainer


@constexpr
def swap_axis(shape):
    perm = list(range(len(shape)))
    perm[0] = 1
    perm[1] = 0
    return perm


class BatchTranspose(nn.Cell):
    def __init__(self):
        super().__init__()
        self.hyper_map = C.HyperMap()

    def _transpose(self, item):
        return item.transpose(swap_axis(item.shape))

    def construct(self, experience):
        return self.hyper_map(self._transpose, experience)


class COMATrainer(Trainer):
    """
    This is the trainer class of COMA, which provides the logic of this algorithm.
    """

    def __init__(self, msrl, params):
        super().__init__(msrl)
        self.batch = params["batch_size"]

        env_config = self.msrl.collect_environment.config
        observation_space = self.msrl.collect_environment.observation_space
        action_space = self.msrl.collect_environment.action_space
        done_space = self.msrl.collect_environment.done_space
        reward_space = self.msrl.collect_environment.reward_space
        self.num_env = 8
        self.num_agent = env_config["num_agent"]
        self.agent_id = Tensor(
            np.array(
                [np.expand_dims(np.eye(self.num_agent), 0).reshape(self.num_agent, -1)]
                * self.num_env
            ),
            ms.float32,
        )

        self.episode_limit = Tensor(env_config["episode_limit"])
        self.action_dim = action_space.num_values
        self.observation_dim = (
            observation_space.shape[-1] + self.num_agent + self.action_dim
        )
        self.global_obs_dim = env_config["global_observation_dim"]

        self.reward_dim = reward_space.shape[-1]
        self.done_dim = done_space.shape[-1]
        self.episode_steps = Parameter(
            Tensor(0, ms.int32), requires_grad=False, name="episode_steps"
        )

        self.onehot = P.OneHot()
        self.transpose = BatchTranspose()

    def trainable_variables(self):
        return {}

    def train_one_episode(self):
        done = F.zeros((self.num_env, 1), ms.bool_)
        steps = Tensor([0], ms.int32)

        obs, state, avail_action = self.msrl.collect_environment.reset()

        hy = F.zeros((self.num_env * self.num_agent, 64), ms.float32)
        action = F.zeros((self.num_env, self.num_agent), ms.int32)
        last_onehot_action = F.zeros(
            (self.num_env, self.num_agent, self.action_dim),
            ms.float32,
        )

        while True:
            concat_obs = F.concat(
                (
                    obs,
                    F.reshape(last_onehot_action, (self.num_env, self.num_agent, -1)),
                    self.agent_id,
                ),
                axis=-1,
            )

            (
                next_obs,
                done,
                reward,
                action,
                hy,
                next_state,
                avail_action,
            ) = self.msrl.agent_act(
                trainer.COLLECT,
                (
                    concat_obs.reshape(-1, self.observation_dim),
                    hy,
                    avail_action,
                    self.episode_steps,
                ),
            )

            terminated = P.LogicalOr()(done, (self.episode_limit < steps))
            mark_filled = P.LogicalNot()(terminated)
            self.msrl.buffers.insert(
                (
                    obs,
                    state,
                    action,
                    avail_action,
                    reward,
                    terminated,
                    last_onehot_action,
                    mark_filled,
                )
            )

            if done.all():
                break

            obs = next_obs
            state = next_state
            last_onehot_action = F.cast(
                self.onehot(action, self.action_dim, Tensor(1.0), Tensor(0.0)),
                ms.float32,
            )
            steps += 1

        (
            obs,
            state,
            actions,
            avail_actions,
            rewards,
            terminated,
            last_actions_onehot,
            filled,
        ) = self.msrl.get_replay_buffer_elements()

        (
            obs,
            state,
            actions,
            avail_actions,
            rewards,
            terminated,
            last_actions_onehot,
            filled,
        ) = self.transpose(
            (
                obs,
                state,
                actions,
                avail_actions,
                rewards,
                terminated,
                last_actions_onehot,
                filled,
            )
        )
        last_actions_onehot = F.concat(
            (last_actions_onehot, F.zeros_like(last_actions_onehot[:, 0:1])), axis=1
        )
        actions = actions.unsqueeze(-1)
        self.episode_steps += filled.sum() + self.num_env

        experience = (
            state,
            obs,
            actions,
            avail_actions,
            rewards,
            terminated,
            last_actions_onehot,
            filled,
        )
        critic_loss, actor_loss = self.msrl.learner.learn(experience)
        return critic_loss, actor_loss

    def evaluate(self):
        pass
