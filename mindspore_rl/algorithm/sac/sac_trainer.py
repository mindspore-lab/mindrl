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
"""SAC Trainer"""
import mindspore
from mindspore import Parameter, Tensor

from mindspore.ops import operations as P

from mindspore_rl.agent import trainer
from mindspore_rl.agent.trainer import Trainer


# pylint: disable=W0212
class SACTrainer(Trainer):
    """This is the trainer class of SAC algorithm. It arranges the SAC algorithm"""

    def __init__(self, msrl, params=None):
        super(SACTrainer, self).__init__(msrl)
        self.inited = Parameter(Tensor([False], mindspore.bool_), name="init_flag")
        self.zero = Tensor([0], mindspore.float32)
        self.fill_value = Tensor([10000], mindspore.float32)
        self.false = Tensor([False], mindspore.bool_)
        self.true = Tensor([True], mindspore.bool_)
        self.less = P.Less()
        self.reduce_mean = P.ReduceMean()
        self.duration = params["duration"]
        self.num_eval_episode = params["num_eval_episode"]

    def trainable_variables(self):
        """Trainable variables for saving."""
        trainable_variables = {"actor_net": self.msrl.actors.collect_policy.actor_net}
        return trainable_variables

    def init_training(self):
        """Initialize training"""
        state = self.msrl.collect_environment.reset()
        done = self.false
        i = self.zero
        while self.less(i, self.fill_value):
            new_state, action, reward, done = self.msrl.agent_act(trainer.INIT, state)
            self.msrl.replay_buffer_insert([state, action, reward, new_state, done])
            state = new_state
            if done:
                state = self.msrl.collect_environment.reset()
                done = self.false
            i += 1
        return done

    @mindspore.jit
    def train_one_episode(self):
        """the algorithm in one episode"""
        if not self.inited:
            self.init_training()
            self.inited = self.true

        done = self.false
        loss = self.zero
        step = self.zero
        total_reward = self.zero
        state = self.msrl.collect_environment.reset()
        while not done:
            new_state, action, reward, done = self.msrl.agent_act(
                trainer.COLLECT, state
            )
            self.msrl.replay_buffer_insert([state, action, reward, new_state, done])
            state = new_state
            total_reward += reward
            batched_transition = self.msrl.replay_buffer_sample()
            loss += self.msrl.agent_learn(batched_transition)
            step += 1
        return loss / step, total_reward, step

    @mindspore.jit
    def evaluate(self):
        """evaluate function"""
        total_eval_reward = self.zero
        num_eval = self.zero
        while num_eval < self.num_eval_episode:
            eval_reward = self.zero
            state = self.msrl.eval_environment.reset()
            j = self.zero
            while self.less(j, self.duration):
                reward, state = self.msrl.agent_act(trainer.EVAL, state)
                reward = self.reduce_mean(reward)
                eval_reward += reward
                j += 1
            num_eval += 1
            total_eval_reward += eval_reward
        avg_eval_reward = total_eval_reward / self.num_eval_episode
        return avg_eval_reward[0]
