# Copyright 2021 Huawei Technologies Co., Ltd
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
"""AC Trainer"""
import mindspore
from mindspore import Tensor

from mindspore.common.parameter import Parameter
from mindspore.ops import operations as P

from mindspore_rl.agent import trainer
from mindspore_rl.agent.trainer import Trainer


class ACTrainer(Trainer):
    """ACTrainer"""

    def __init__(self, msrl, params):
        super(ACTrainer, self).__init__(msrl)
        self.num_evaluate_episode = params["num_evaluate_episode"]
        self.zero = Parameter(Tensor(0, mindspore.float32), name="zero")
        self.done_r = Parameter(Tensor([-20.0], mindspore.float32), name="done_r")
        self.zero_value = Tensor(0, mindspore.float32)
        self.squeeze = P.Squeeze()
        self.false = Tensor(False, mindspore.bool_)
        self.less = P.Less()
        self.select = P.Select()

    def trainable_variables(self):
        """Trainable variables for saving."""
        trainable_variables = {"actor_net": self.msrl.learner.actor_net}
        return trainable_variables

    @mindspore.jit
    def train_one_episode(self):
        """Train one episode"""
        state = self.msrl.collect_environment.reset()
        done = self.false
        total_reward = self.zero
        steps = self.zero
        loss = self.zero_value
        while True:
            done, r, state_, a = self.msrl.agent_act(trainer.COLLECT, state)
            r = self.squeeze(r)
            total_reward += r
            if done:
                r = self.done_r
            loss = self.msrl.agent_learn([state, r, state_, a])
            state = state_
            steps += 1
            if done:
                break
        return loss, total_reward, steps

    @mindspore.jit
    def evaluate(self):
        """evaluate"""
        total_reward = self.zero_value
        eval_iter = self.zero_value
        while self.less(eval_iter, self.num_evaluate_episode):
            episode_reward = self.zero_value
            state = self.msrl.eval_environment.reset()
            done = self.false
            while not done:
                done, r, state = self.msrl.agent_act(trainer.EVAL, state)
                r = self.squeeze(r)
                episode_reward += r
            total_reward += episode_reward
            eval_iter += 1
        avg_reward = total_reward / self.num_evaluate_episode
        return avg_reward
