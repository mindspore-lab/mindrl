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
"""Dreamer Trainer"""
import mindspore as ms
from mindspore import Parameter, Tensor, ops
from mindspore.ops import operations as P

from mindspore_rl.agent import trainer
from mindspore_rl.agent.trainer import Trainer


class DreamerTrainer(Trainer):
    """This is the trainer class of Dreamer algorithm. It arranges the Dreamer algorithm"""

    def __init__(self, msrl, params=None):
        super().__init__(msrl)
        self.episode_limits_tensor = Tensor(
            params["episode_limits"] / params["action_repeat"], ms.int32
        )
        self.episode_limits = int(params["episode_limits"] / params["action_repeat"])
        self.obs_shape = params["conv_decoder_shape"]
        self.action_space_dim = self.msrl.collect_environment.action_space.num_values
        self.deter_size = params["deter_size"]
        self.stoch_size = params["stoch_size"]
        self.dtype = params["dtype"]
        self.prefill_value = Tensor(params["prefill_value"], ms.int32)
        self.action_repeat = Tensor(params["action_repeat"], ms.int32)
        self.train_steps = Tensor(params["train_steps"], ms.int32)

        self.zeros = P.Zeros()
        self.ones = P.Ones()
        self.expand_dims = P.ExpandDims()

        self.zero_int = Tensor(0, ms.int32)
        self.zero_float = Tensor(0, ms.float32)
        self.true = Tensor(True, ms.bool_)
        self.init = Parameter(Tensor(False, ms.bool_), requires_grad=False, name="init")
        self.steps = Parameter(Tensor(0, ms.int32), requires_grad=False, name="steps")

    @ms.jit
    def init_training(self):
        """Use random policy to select action and interact with env"""
        i = self.zero_int
        # ============================================
        state = self.msrl.collect_environment.reset()
        state = ops.cast(state, ms.float16)
        # ============================================
        episode_obs = self.zeros(
            (self.episode_limits + 1,) + self.obs_shape, ms.float16
        )
        episode_action = self.zeros(
            (self.episode_limits + 1, self.action_space_dim), ms.float16
        )
        episode_reward = self.zeros((self.episode_limits + 1, 1), ms.float16)
        episode_discount = self.ones((self.episode_limits + 1, 1), ms.float16)

        episode_obs[i] = state
        i += 1

        prev_mean = self.zeros((1, self.stoch_size), ms.float16)
        prev_std = self.zeros((1, self.stoch_size), ms.float16)
        prev_stoch = self.zeros((1, self.stoch_size), ms.float16)
        prev_deter = self.zeros((1, self.deter_size), ms.float16)
        prev_action = self.zeros((1, self.action_space_dim), ms.float16)

        while i < self.episode_limits_tensor + 1:
            state = self.expand_dims(state, 0)
            (
                half_action,
                prev_mean,
                prev_std,
                prev_stoch,
                prev_deter,
            ) = self.msrl.agent_act(
                trainer.INIT,
                (state, prev_mean, prev_std, prev_stoch, prev_deter, prev_action),
            )
            # =========================================================================
            action = ops.cast(half_action, ms.float32)
            new_state, reward, _, discount = self.msrl.collect_environment.step(action)
            new_state = ops.cast(new_state, ms.float16)
            reward = ops.cast(reward, ms.float16)
            discount = ops.cast(discount, ms.float16)
            # =========================================================================
            episode_obs[i] = new_state
            episode_action[i] = half_action
            episode_reward[i] = reward
            episode_discount[i] = discount
            state = new_state
            prev_action = half_action
            i += 1
        self.msrl.buffers.insert(
            episode_obs, episode_action, episode_reward, episode_discount
        )
        return i

    @ms.jit
    def train_one_episode(self):
        # Init parts
        while not self.init:
            self.steps += self.init_training() * self.action_repeat
            if self.steps >= self.prefill_value:
                self.init = self.true
        i = self.zero_int
        # ============================================
        state = self.msrl.collect_environment.reset()
        state = ops.cast(state, ms.float16)
        # ============================================
        episode_obs = self.zeros(
            (self.episode_limits + 1,) + self.obs_shape, ms.float16
        )
        episode_action = self.zeros(
            (self.episode_limits + 1, self.action_space_dim), ms.float16
        )
        episode_reward = self.zeros((self.episode_limits + 1, 1), ms.float16)
        episode_discount = self.ones((self.episode_limits + 1, 1), ms.float16)

        episode_obs[i] = state
        i += 1

        # Init prev_mean, prev_std, prev_stoch, prev_deter and prev_action
        prev_mean = self.zeros((1, self.stoch_size), ms.float16)
        prev_std = self.zeros((1, self.stoch_size), ms.float16)
        prev_stoch = self.zeros((1, self.stoch_size), ms.float16)
        prev_deter = self.zeros((1, self.deter_size), ms.float16)
        prev_action = self.zeros((1, self.action_space_dim), ms.float16)
        while i < self.episode_limits_tensor + 1:
            state = self.expand_dims(state, 0)
            (
                half_action,
                prev_mean,
                prev_std,
                prev_stoch,
                prev_deter,
            ) = self.msrl.agent_act(
                trainer.COLLECT,
                (state, prev_mean, prev_std, prev_stoch, prev_deter, prev_action),
            )
            # =========================================================================
            action = ops.cast(half_action, ms.float32)
            new_state, reward, _, discount = self.msrl.collect_environment.step(action)
            new_state = ops.cast(new_state, ms.float16)
            reward = ops.cast(reward, ms.float16)
            discount = ops.cast(discount, ms.float16)
            # =========================================================================
            episode_obs[i] = new_state
            episode_action[i] = half_action
            episode_reward[i] = reward
            episode_discount[i] = discount
            state = new_state
            prev_action = half_action
            i += 1
        self.msrl.buffers.insert(
            episode_obs, episode_action, episode_reward, episode_discount
        )
        self.steps += i - 1

        j = self.zero_int
        loss = Tensor(0, ms.float16)
        while j < self.train_steps:
            data = self.msrl.buffers.sample()
            loss = self.msrl.agent_learn(data)
            j += 1
        return loss, episode_reward.sum(), i

    @ms.jit
    def evaluate(self):
        i = self.zero_int
        state = self.msrl.eval_environment.reset()
        episode_reward = self.zeros((self.episode_limits + 1, 1), self.dtype)
        i += 1

        # Init prev_mean, prev_std, prev_stoch, prev_deter and prev_action
        prev_mean = self.zeros((1, self.stoch_size), self.dtype)
        prev_std = self.zeros((1, self.stoch_size), self.dtype)
        prev_stoch = self.zeros((1, self.stoch_size), self.dtype)
        prev_deter = self.zeros((1, self.deter_size), self.dtype)
        prev_action = self.zeros((1, self.action_space_dim), self.dtype)
        while i < self.episode_limits_tensor + 1:
            state = self.expand_dims(state, 0)
            action, prev_mean, prev_std, prev_stoch, prev_deter = self.msrl.agent_act(
                trainer.EVAL,
                (state, prev_mean, prev_std, prev_stoch, prev_deter, prev_action),
            )
            new_state, reward, _, _ = self.msrl.eval_environment.step(action)
            episode_reward[i] = reward
            state = new_state
            prev_action = action
            i += 1

        return episode_reward.sum()

    def trainable_variables(self):
        return {}
