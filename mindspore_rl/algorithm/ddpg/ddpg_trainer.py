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
"""DDPG Trainer"""
import mindspore
from mindspore.common.api import ms_function
from mindspore import Tensor, Parameter
from mindspore.ops import operations as P
from mindspore_rl.agent.trainer import Trainer
from mindspore_rl.agent import trainer


class DDPGTrainer(Trainer):
    """This is the trainer class of DDPG algorithm. It arranges the DDPG algorithm"""

    def __init__(self, msrl, params=None):
        super(DDPGTrainer, self).__init__(msrl)
        self.zero = Tensor(0, mindspore.float32)
        self.assign = P.Assign()
        self.squeeze = P.Squeeze()
        self.mod = P.Mod()
        self.equal = P.Equal()
        self.less = P.Less()
        self.reduce_mean = P.ReduceMean()
        self.duration = params['duration']
        self.num_eval_episode = params['num_eval_episode']
        self.true = Tensor(True, mindspore.bool_)
        self.false = Tensor([False], mindspore.bool_)
        self.zero_value = Tensor(0, mindspore.float32)
        self.init_collect_size = Tensor(1000, mindspore.float32)
        self.inited = Parameter(Tensor(False, mindspore.bool_), name='init_flag')

    def trainable_variables(self):
        """Trainable variables for saving."""
        trainable_variables = {"actor_net": self.msrl.actors.actor_net}
        return trainable_variables

    @ms_function
    def init_training(self):
        """Initialize training"""
        obs = self.msrl.collect_environment.reset()
        done = self.false
        i = self.zero_value
        while self.less(i, self.init_collect_size):
            next_obs, actions, rewards, done = self.msrl.agent_act(
                trainer.INIT, obs)
            self.msrl.replay_buffer_insert(
                [obs, actions, rewards, next_obs, done])
            obs = next_obs
            i += 1
        return i

    @ms_function
    def train_one_episode(self):
        """the algorithm in one episode"""
        if not self.inited:
            self.init_training()
            self.inited = self.true

        obs = self.msrl.collect_environment.reset()
        done = self.false
        total_reward = self.zero
        steps = self.zero
        loss = self.zero
        while not done:
            next_obs, actions, rewards, done = self.msrl.agent_act(trainer.INIT, obs)
            self.msrl.replay_buffer_insert([obs, actions, rewards, next_obs, done])
            obs = next_obs
            rewards = self.squeeze(rewards)
            loss = self.msrl.agent_learn(self.msrl.replay_buffer_sample())
            total_reward += rewards
            steps += 1
        return loss, total_reward, steps


    @ms_function
    def evaluate(self):
        """evaluate function"""
        total_eval_reward = self.zero
        num_eval = self.zero
        while num_eval < self.num_eval_episode:
            eval_reward = self.zero
            done = self.false
            obs = self.msrl.collect_environment.reset()
            while not done:
                next_obs, _, rewards, done = self.msrl.agent_act(trainer.EVAL, obs)
                obs = next_obs
                rewards = self.squeeze(rewards)
                eval_reward += rewards
            num_eval += 1
            total_eval_reward += eval_reward
        avg_eval_reward = total_eval_reward / self.num_eval_episode
        return avg_eval_reward
