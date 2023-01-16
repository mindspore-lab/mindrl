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
"""TD3 Trainer"""
import mindspore
from mindspore.common.api import ms_function
from mindspore import Tensor, Parameter
from mindspore.ops import operations as P
from mindspore_rl.agent.trainer import Trainer
from mindspore_rl.agent import trainer

from .td3_summary import RecordQueue


class TD3Trainer(Trainer):
    """This is the trainer class of TD3 algorithm. It arranges the TD3 algorithm"""

    def __init__(self, msrl, params=None):
        super(TD3Trainer, self).__init__(msrl)
        self.zero = Tensor(0, mindspore.float32)
        self.assign = P.Assign()
        self.squeeze = P.Squeeze()
        self.mod = P.Mod()
        self.equal = P.Equal()
        self.less = P.Less()
        self.reduce_mean = P.ReduceMean()
        self.num_eval_episode = params['num_eval_episode']
        self.true = Tensor(True, mindspore.bool_)
        self.false = Tensor([False], mindspore.bool_)
        self.init_collect_size = Tensor(params['init_collect_size'], mindspore.float32)
        self.inited = Parameter(Tensor(False, mindspore.bool_), name='init_flag')
        if 'eval_episodes' in params:
            self.eval_episodes = params['eval_episodes']

    def trainable_variables(self):
        """Trainable variables for saving."""
        trainable_variables = {"actor_net": self.msrl.actors.actor_net}
        return trainable_variables

    @ms_function
    def init_training(self):
        """Initialize training"""
        obs = self.msrl.collect_environment.reset()
        done = self.false
        i = self.zero
        while self.less(i, self.init_collect_size):
            next_obs, action, reward, done = self.msrl.agent_act(trainer.INIT, obs)
            self.msrl.replay_buffer_insert([obs, action, reward, next_obs, done])
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
        total_rewards = self.zero
        steps = self.zero
        loss = self.zero
        while not done:
            next_obs, action, reward, done = self.msrl.agent_act(trainer.COLLECT, obs)
            self.msrl.replay_buffer_insert([obs, action, reward, next_obs, done])
            obs = next_obs
            reward = self.squeeze(reward)
            loss = self.msrl.agent_learn(self.msrl.replay_buffer_sample())
            total_rewards += reward
            steps += 1
        return loss, total_rewards, steps

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

    def load_and_eval(self, ckpt_path=None):
        """
        The interface of the eval function for offline. A checkpoint must be provided.

        Args:
            ckpt_path (string): The checkpoint file to restore net.
        """
        if ckpt_path is None:
            raise RuntimeError("Please provide a ckpt_path.")
        self._init_or_restore(ckpt_path)
        if self.eval_episodes <= 0:
            raise ValueError("In order to get average rewards,\
                evaluate episodes should be larger than 0, but got {}".format(self.eval_episodes))
        rewards = RecordQueue()
        for _ in range(self.eval_episodes):
            reward = self.evaluate()
            rewards.add(reward)
        avg_reward = rewards.mean().asnumpy()
        print("-----------------------------------------")
        print(f"Average evaluate result is {avg_reward:.3f}, checkpoint file in {ckpt_path}")
        print("-----------------------------------------")
