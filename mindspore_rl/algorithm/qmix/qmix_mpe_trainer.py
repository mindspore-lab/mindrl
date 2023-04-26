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
"""QMIX Trainer"""

import mindspore as ms
import numpy as np
from mindspore import Parameter, Tensor

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
        self.stack = P.Stack()

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
        self.observation_dim = observation_space.shape[-1]
        self.global_obs_dim = env_config["global_observation_dim"]
        self.num_envs = 1

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
        total_reward = self.zero_float
        steps = 0
        loss = self.zero_float
        hy = self.zeros((self.num_agent, 64), ms.float32)

        episode_local_obs = []
        episode_global_obs = []
        episode_action = []
        episode_reward = []
        episode_done = []
        episode_done_env = []
        avail_action = self.ones((self.num_agent, self.action_dim), ms.int32)

        local_obs = self.msrl.collect_environment.reset()
        # local_obs = local_obs.squeeze(0)

        while steps < self.episode_limit:
            global_obs = local_obs.reshape((-1,))
            action, hy = self.msrl.actors.get_action(
                trainer.COLLECT, (local_obs, hy, avail_action, self.epsilon_steps)
            )
            new_local_obs, reward, done = self.msrl.collect_environment.step(
                action.astype(ms.int32)
            )
            done = self.expand_dims(done, -1)
            reward = reward[0]
            done = done[0]

            done_envs = done.all()

            episode_local_obs.append(local_obs)
            episode_global_obs.append(global_obs)
            episode_action.append(action)
            episode_reward.append(reward)
            episode_done.append(done)
            episode_done_env.append(self.expand_dims(done_envs, -1))

            local_obs = new_local_obs
            total_reward += reward
            steps += 1

        episode_local_obs.append(local_obs)
        episode_global_obs.append(local_obs.reshape((-1,)))
        episode_local_obs = self.stack(episode_local_obs)
        episode_global_obs = self.stack(episode_global_obs)
        episode_action = self.stack(episode_action)
        episode_reward = self.stack(episode_reward)
        episode_done = self.stack(episode_done)
        episode_done_env = self.stack(episode_done_env)
        self.msrl.replay_buffer_insert(
            (
                episode_local_obs,
                episode_global_obs,
                episode_action,
                episode_reward,
                episode_done,
                episode_done_env,
            )
        )

        self.epsilon_steps += steps
        if self.greater_equal(self.msrl.buffers.count, self.batch):
            loss = self.msrl.agent_learn(self.msrl.replay_buffer_sample())

        return loss, total_reward, steps

    @ms.jit
    def evaluate(self):
        """Evaluation function"""
        total_reward = self.zero_float
        hy = self.zeros((self.num_agent, 64), ms.float32)
        avail_action = self.ones((self.num_agent, self.action_dim), ms.int32)
        steps = 0

        local_obs = self.msrl.eval_environment.reset()
        while steps < self.episode_limit:
            action, hy = self.msrl.actors.get_action(
                trainer.COLLECT, (local_obs, hy, avail_action, self.epsilon_steps)
            )
            new_local_obs, reward, _ = self.msrl.collect_environment.step(
                action.astype(ms.int32)
            )
            total_reward += reward
            local_obs = new_local_obs

        return total_reward
