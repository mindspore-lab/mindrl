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
"""GAIL Trainer"""
from mindspore.common.api import jit
from mindspore import Tensor, Parameter
from mindspore_rl.agent.trainer import Trainer


#pylint: disable=W0212
class GAILTrainer(Trainer):
    """This is the trainer class of GAIL algorithm. It arranges the GAIL algorithm"""

    def __init__(self, msrl, params):
        super(GAILTrainer, self).__init__(params)
        self.collect_policy = msrl.collect_policy
        self.eval_policy = msrl.eval_policy
        self.learner = msrl.learner
        self.collect_environment = msrl.collect_environment
        self.eval_environment = msrl.eval_environment

        self.policy_replay_buffer = msrl.policy_replay_buffer
        self.min_step_before_training = Tensor(params.get('min_step_before_training'))

        self.inited = Parameter(Tensor(False), name='inited', requires_grad=False)
        self.num_eval_episode = Tensor(params.get('num_eval_episode'))

    def trainable_variables(self):
        """Trainable variables for saving."""
        trainable_variables = {"actor_net": self.collect_policy.actor_net}
        return trainable_variables

    def init_training(self):
        """Initialize training"""
        obs = self.collect_environment.reset()
        step = Tensor(0)
        while step < self.min_step_before_training:
            step += 1
            action = self.collect_policy(obs.expand_dims(0)).squeeze()
            next_obs, _, done = self.collect_environment.step(action)
            self.policy_replay_buffer.insert([obs, action, next_obs, Tensor([False])])
            obs = next_obs

            if done:
                obs = self.collect_environment.reset()
        return step

    @jit
    def train_one_episode(self):
        """the algorithm in one episode"""
        if not self.inited:
            self.init_training()
            self.inited = Tensor(True)

        obs = self.collect_environment.reset()
        total_reward = Tensor([0.])
        done = Tensor([False])
        while not done:
            action = self.collect_policy(obs.expand_dims(0)).squeeze()
            next_obs, reward, done = self.collect_environment.step(action)
            self.policy_replay_buffer.insert([obs, action, next_obs, Tensor([False])])
            obs = next_obs
            total_reward += reward

        disc_loss, policy_reward = self.learner.learn()
        return disc_loss + policy_reward, total_reward, Tensor(1000)

    @jit
    def evaluate(self):
        """evaluate function"""
        total_reward = Tensor([0.])
        episode = Tensor(0)
        while episode < self.num_eval_episode:
            episode += 1
            done = Tensor(False)
            obs = self.eval_environment.reset()
            while not done:
                action = self.eval_policy(obs.expand_dims(0)).squeeze()
                next_obs, reward, done = self.eval_environment.step(action)
                obs = next_obs
                total_reward += reward
        return (total_reward / self.num_eval_episode)[0]
