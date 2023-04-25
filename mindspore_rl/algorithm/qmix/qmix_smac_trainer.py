# r Copyright 2022-2023 Huawei Technologies Co., Ltd
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
"""QMIX Trainer"""

import mindspore as ms
import numpy as np
from mindspore import Parameter, Tensor
from mindspore.common.api import ms_function
from mindspore.ops import operations as P

from mindspore_rl.agent import Trainer, trainer


class QMIXTrainer(Trainer):
    """
    This is the trainer class of QMIX, which provides the logic of this algorithm.
    """

    def __init__(self, msrl, params):
        super().__init__(msrl)
        self.msrl = msrl
        self.batch = params["batch_size"]
        self.false = Tensor([False], ms.bool_)
        self.true = Tensor(True, ms.bool_)
        self.zero_int = Tensor(0, ms.int32)
        self.zero_float = Tensor(0, ms.float32)
        self.one_int = Tensor(1, ms.int32)
        self.one_float = Tensor(1, ms.float32)
        self.zeros = P.Zeros()
        self.ones = P.Ones()
        self.expand_dims = P.ExpandDims()
        self.concat = P.Concat(axis=1)
        self.reshape = P.Reshape()
        self.cast = P.Cast()
        self.onehot = P.OneHot()
        self.zeros_like = P.ZerosLike()
        self.assign = P.Assign()
        self.equal = P.Equal()
        self.select = P.Select()

        env_config = self.msrl.collect_environment.config
        observation_space = self.msrl.collect_environment.observation_space
        action_space = self.msrl.collect_environment.action_space
        done_space = self.msrl.collect_environment.done_space
        reward_space = self.msrl.collect_environment.reward_space

        self.num_agent = env_config["num_agent"]
        self.agent_id = Tensor(
            np.expand_dims(np.eye(self.num_agent), 0).reshape(self.num_agent, -1),
            ms.float32,
        )
        self.episode_limit = env_config["episode_limit"]
        self.action_dim = action_space.num_values
        self.observation_dim = (
            observation_space.shape[-1] + self.num_agent + self.action_dim
        )
        self.global_obs_dim = env_config["global_observation_dim"]

        self.reward_dim = 1 if len(reward_space.shape) == 0 else reward_space.shape[-1]
        self.done_dim = 1 if len(done_space.shape) == 0 else done_space.shape[-1]

        self.epsilon_steps = Parameter(
            Tensor(0, ms.int32), requires_grad=False, name="epsilon_steps"
        )
        self.squeeze = P.Squeeze(axis=0)
        self.greater_equal = P.GreaterEqual()

    def trainable_variables(self):
        """trainable variables uses to save model"""
        trainable_variables = {
            "policy_net": self.msrl.learner.policy_net,
            "mixer_net": self.msrl.learner.mixer_net,
        }
        return trainable_variables

    @ms.jit
    def train_one_episode(self):
        """the train one episode implementation"""
        done = self.false
        steps = self.zero_int
        total_reward = self.zero_float
        loss = self.zero_float
        new_state = self.zeros((5, 80), ms.float32)
        hy = self.zeros((5, 64), ms.float32)

        episode_local_obs = self.zeros(
            (self.episode_limit + 1, self.num_agent, self.observation_dim), ms.float32
        )
        episode_global_obs = self.zeros(
            (self.episode_limit + 1, self.global_obs_dim), ms.float32
        )
        episode_action = self.zeros(
            (self.episode_limit + 1, self.num_agent, 1), ms.int32
        )
        episode_avail_action = self.zeros(
            (self.episode_limit + 1, self.num_agent, self.action_dim), ms.int32
        )
        episode_reward = self.zeros(
            (self.episode_limit + 1, self.reward_dim), ms.float32
        )
        episode_done = self.zeros((self.episode_limit + 1, self.done_dim), ms.bool_)
        episode_filled = self.zeros((self.episode_limit + 1, self.done_dim), ms.int32)
        episode_hy = self.zeros(
            (self.episode_limit + 1, self.num_agent, 64), ms.float32
        )

        local_obs, global_obs, avail_action = self.msrl.collect_environment.reset()
        last_onehot_action = self.zeros((self.num_agent, self.action_dim), ms.float32)
        concat_obs = self.concat(
            (
                local_obs,
                self.reshape(last_onehot_action, (self.num_agent, -1)),
                self.agent_id,
            )
        )
        episode_local_obs[steps] = concat_obs
        episode_global_obs[steps] = global_obs
        episode_avail_action[steps] = avail_action
        battle_won = Tensor(False)
        dead_allies = Tensor(0)
        dead_enemies = Tensor(0)
        steps += 1

        while (not done) and (steps < self.episode_limit):
            (
                new_state,
                done,
                reward,
                action,
                hy,
                new_global_obs,
                avail_action,
                battle_won,
                dead_allies,
                dead_enemies,
            ) = self.msrl.agent_act(
                trainer.COLLECT, (concat_obs, hy, avail_action, self.epsilon_steps)
            )
            reward = self.expand_dims(reward, 0)
            done = self.expand_dims(done, 0)
            last_onehot_action = self.onehot(
                action, self.action_dim, self.one_float, self.zero_float
            ).astype(ms.float32)
            concat_obs = self.concat(
                (
                    new_state,
                    self.reshape(last_onehot_action, (self.num_agent, -1)),
                    self.agent_id,
                )
            )

            episode_local_obs[steps] = concat_obs
            episode_global_obs[steps] = new_global_obs
            episode_hy[steps] = hy
            episode_action[steps - 1] = action
            episode_avail_action[steps] = avail_action
            episode_reward[steps - 1] = reward
            reach_episode_limit = self.expand_dims(
                self.equal(self.episode_limit, steps), 0
            )
            if reach_episode_limit:
                done = self.false
            episode_done[steps - 1] = done
            episode_filled[steps - 1] = self.expand_dims(self.one_int, 0)
            reward_squeeze = self.squeeze(reward)
            total_reward += reward_squeeze
            steps += 1

        action, hy = self.msrl.agent_get_action(
            trainer.COLLECT, (concat_obs, hy, avail_action, self.epsilon_steps)
        )
        last_onehot_action = self.onehot(
            action, self.action_dim, self.one_float, self.zero_float
        )
        concat_obs = self.concat(
            (
                new_state,
                self.reshape(last_onehot_action, (self.num_agent, -1)),
                self.agent_id,
            )
        )
        episode_local_obs[steps] = concat_obs
        episode_hy[steps] = hy
        episode_action[steps - 1] = action

        self.msrl.replay_buffer_insert(
            (
                episode_local_obs,
                episode_global_obs,
                episode_action,
                episode_avail_action,
                episode_reward,
                episode_done,
                episode_filled,
                episode_hy,
            )
        )
        self.epsilon_steps += steps
        if self.greater_equal(self.msrl.buffers.count, self.batch):
            loss = self.msrl.agent_learn(self.msrl.replay_buffer_sample())

        step_info = (battle_won, dead_allies, dead_enemies)

        return loss, total_reward, steps, step_info

    @ms.jit
    def evaluate(self):
        """Evaluation function"""
        done = self.false
        hy = self.zeros((self.num_agent, 64), ms.float32)

        local_obs, _, avail_action = self.msrl.eval_environment.reset()
        last_onehot_action = self.zeros((self.num_agent, self.action_dim), ms.float32)
        concat_obs = self.concat(
            (
                local_obs,
                self.reshape(last_onehot_action, (self.num_agent, -1)),
                self.agent_id,
            )
        )
        battle_won = Tensor(False)
        dead_allies = Tensor(0)
        dead_enemies = Tensor(0)
        while not done:
            (
                new_state,
                done,
                _,
                action,
                hy,
                _,
                avail_action,
                battle_won,
                dead_allies,
                dead_enemies,
            ) = self.msrl.agent_act(
                trainer.EVAL, (concat_obs, hy, avail_action, self.epsilon_steps)
            )
            done = self.expand_dims(done, 0)
            last_onehot_action = self.onehot(
                action, self.action_dim, self.one_float, self.zero_float
            ).astype(ms.float32)
            concat_obs = self.concat(
                (
                    new_state,
                    self.reshape(last_onehot_action, (self.num_agent, -1)),
                    self.agent_id,
                )
            )

        step_info = (battle_won, dead_allies, dead_enemies)
        return step_info
