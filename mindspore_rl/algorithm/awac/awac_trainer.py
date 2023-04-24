# Copyright 2022 Huawei Technologies Co., Ltd
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
"""AWAC Trainer"""
import mindspore
import numpy as np
from mindspore import Parameter, Tensor
from mindspore.common.api import jit
from mindspore.ops import operations as ops

from mindspore_rl.agent.trainer import Trainer

# pylint: disable=W0702
# pylint: disable=W0611
try:
    import d4rl
except ImportError as e:
    raise ImportError(
        "d4rl is not installed.\n"
        "please refer to:\n"
        "https://github.com/Farama-Foundation/D4RL"
    ) from e


class InitBuffer:
    """Load d4rl data and set to RL replaybuffer"""

    def __init__(self, env, msbuffer, length=None):
        dataset = d4rl.qlearning_dataset(env)
        self.np_data = dataset
        self.env = env
        if length is None:
            length = self.np_data["terminals"].shape[0]
        actions = Tensor(self.np_data["actions"][:length], mindspore.float32)
        observations = Tensor(self.np_data["observations"][:length], mindspore.float32)
        rewards = Tensor(
            np.expand_dims(self.np_data["rewards"][:length], -1), mindspore.float32
        )
        terminals = Tensor(
            np.expand_dims(self.np_data["terminals"][:length], -1), mindspore.float32
        )
        next_observations = Tensor(
            self.np_data["next_observations"][:length], mindspore.float32
        )
        msbuffer.insert((observations, actions, rewards, next_observations, terminals))


class AWACTrainer(Trainer):
    """AWACTrainer"""

    def __init__(self, msrl, params):
        super().__init__(msrl)
        self.env = msrl.eval_environment
        self.zero = Tensor(0.0, mindspore.float32)
        self.done = Tensor(False, mindspore.bool_)
        self.less = ops.Less()
        self.cast = ops.Cast()
        self.zeroslike = ops.ZerosLike()
        self.num_evaluate_episode = params["num_evaluate_episode"]
        self.ms_buffer = msrl.buffers
        # pylint: disable=W0212
        InitBuffer(self.env._env._env, self.ms_buffer)
        # train offline then fit online
        self.warmup = Parameter(
            Tensor(0, mindspore.float32), requires_grad=False, name="warmup"
        )
        self.offline_train_steps = Tensor(
            params["offline_train_steps"], mindspore.float32
        )
        self.n_steps_per_episode = Tensor(
            params["n_steps_per_episode"], mindspore.float32
        )
        self.need_reset = Tensor(True, mindspore.bool_)

    @jit
    def train_one_episode(self):
        """Train one episode"""
        obs = self.msrl.collect_environment.reset().expand_dims(0)
        critic_loss = self.zero
        actor_loss = self.zero
        mean_std = self.zero
        steps = self.zero
        need_reset = self.need_reset
        while self.less(steps, self.n_steps_per_episode):
            if need_reset:
                obs = self.msrl.collect_environment.reset().expand_dims(0)
            # first learn from offline data
            # then collect experience for finetune
            if self.warmup > self.offline_train_steps:
                actions, rewards, next_obs, terminals = self.msrl.actors.act(2, obs)
                need_reset = self.cast(terminals.squeeze(), mindspore.bool_)
                if not need_reset:
                    rewards = rewards.expand_dims(0)
                    self.ms_buffer.insert((obs, actions, rewards, next_obs, terminals))
                obs = next_obs
            experience = self.ms_buffer.sample()
            critic_loss, actor_loss, mean_std = self.msrl.agent_learn(experience)
            steps += 1
            self.warmup += 1
        return (critic_loss, actor_loss, mean_std), self.zero, self.zero

    @jit
    def evaluate(self):
        """Default evaluate"""
        avg_reward = self.zero
        eval_iter = self.zero
        while self.less(eval_iter, self.num_evaluate_episode):
            obs = self.env.reset()
            done = self.done
            while not done:
                data = obs.reshape(1, -1)
                action = self.msrl.actors.get_action(data)
                obs, reward, done = self.env.step(action)
                avg_reward += reward
            eval_iter += 1
        avg_reward /= self.num_evaluate_episode
        return avg_reward

    def trainable_variables(self):
        """Default trainable variables"""
        trainable_variables = {
            "policy": self.msrl.learner.policy,
            "value_net_1": self.msrl.learner.model_1,
            "value_net_2": self.msrl.learner.model_2,
        }
        return trainable_variables
