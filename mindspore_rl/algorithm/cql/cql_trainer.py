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
"""CQL Trainer"""
import numpy as np
from mindspore_rl.agent.trainer import Trainer
import mindspore
from mindspore.ops import operations as ops
from mindspore.common.api import jit
from mindspore import Tensor
#pylint: disable=W0702
#pylint: disable=W0611
try:
    import d4rl
except ImportError as e:
    raise ImportError(
        "d4rl is not installed.\n"
        "please refer to:\n"
        "https://github.com/Farama-Foundation/D4RL"
    ) from e


class InitBuffer():
    '''Load d4rl data and set to RL replaybuffer'''
    def __init__(self, env, msbuffer):
        env.seed(10)
        dataset = d4rl.qlearning_dataset(env)
        self.np_data = dataset
        self.env = env

        actions = Tensor(self.np_data['actions'], mindspore.float32)
        observations = Tensor(self.np_data['observations'], mindspore.float32)
        rewards = Tensor(np.expand_dims(self.np_data['rewards'], -1), mindspore.float32)
        terminals = Tensor(np.expand_dims(self.np_data['terminals'], -1), mindspore.float32)
        next_observations = Tensor(self.np_data['next_observations'], mindspore.float32)
        msbuffer.insert((observations, actions, rewards, next_observations, terminals))


class CQLTrainer(Trainer):
    '''CQLTrainer'''
    def __init__(self, msrl, params):
        super(CQLTrainer, self).__init__(msrl)
        self.env = msrl.eval_environment
        self.zero = Tensor(0., mindspore.float32)
        self.done = Tensor(False, mindspore.bool_)
        self.less = ops.Less()
        self.num_evaluate_episode = params['num_evaluate_episode']
        self.test_freq = params['eval_per_episode']
        self.ms_buffer = msrl.buffers
        #pylint: disable=W0212
        InitBuffer(self.env._env, self.ms_buffer)

    @jit
    def train_one_episode(self):
        '''Train one episode'''
        experience = self.ms_buffer.sample()
        critic_loss, actor_loss = self.msrl.agent_learn(experience)
        return (critic_loss, actor_loss), self.zero, self.zero

    @jit
    def evaluate(self):
        '''Default evaluate'''
        avg_reward = self.zero
        eval_iter = self.zero
        while self.less(eval_iter, self.num_evaluate_episode):
            obs = self.env.reset()
            done = self.done
            while not done:
                data = obs.reshape(1, -1)
                action = self.msrl.actors.act(data)
                obs, reward, done = self.env.step(action)
                avg_reward += reward.squeeze()
            eval_iter += 1
        avg_reward /= self.num_evaluate_episode
        return avg_reward

    def trainable_variables(self):
        '''Default trainable variables'''
        trainable_variables = {"policy": self.msrl.learner.policy,
                               "value_net_1": self.msrl.learner.model_1,
                               "value_net_2": self.msrl.learner.model_2,}
        return trainable_variables
