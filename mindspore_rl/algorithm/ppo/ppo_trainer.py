# Copyright 2021-2023 Huawei Technologies Co., Ltd
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
"""PPO Trainer"""
import mindspore
from mindspore import Tensor
from mindspore.common.api import jit
from mindspore.ops import operations as P

from mindspore_rl.agent import trainer
from mindspore_rl.agent.trainer import Trainer


# pylint: disable=W0212
class PPOTrainer(Trainer):
    """This is the trainer class of PPO algorithm. It arranges the PPO algorithm"""

    def __init__(self, msrl, params=None):
        # pylint: disable=R1725
        super(PPOTrainer, self).__init__(msrl)
        self.zero = Tensor(0, mindspore.float32)
        self.assign = P.Assign()
        self.squeeze = P.Squeeze()
        self.mod = P.Mod()
        self.equal = P.Equal()
        self.less = P.Less()
        self.reduce_mean = P.ReduceMean()
        self.duration = params["duration"]
        self.num_eval_episode = params["num_eval_episode"]

    def trainable_variables(self):
        """Trainable variables for saving."""
        trainable_variables = {
            "actor_net": self.msrl.learner.actor_net,
            "critic_net": self.msrl.learner.critic_net,
            "ppo_optimizer": self.msrl.learner._ppo_net_train.optimizer,
        }
        return trainable_variables

    @jit
    def train_one_episode(self):
        """the algorithm in one episode"""
        training_loss = self.zero
        training_reward = self.zero
        j = self.zero
        state = self.msrl.collect_environment.reset()
        self.msrl.replay_buffer_reset()
        while self.less(j, self.duration):
            reward, new_state, action, miu, sigma = self.msrl.agent_act(
                trainer.COLLECT, state
            )
            self.msrl.replay_buffer_insert(
                [state, action, reward, new_state, miu, sigma]
            )
            state = new_state
            reward = self.reduce_mean(reward)
            training_reward += reward
            j += 1

        replay_buffer_elements = self.msrl.get_replay_buffer_elements(
            transpose=True, shape=(1, 0, 2)
        )
        state_list = replay_buffer_elements[0]
        action_list = replay_buffer_elements[1]
        reward_list = replay_buffer_elements[2]
        next_state_list = replay_buffer_elements[3]
        miu_list = replay_buffer_elements[4]
        sigma_list = replay_buffer_elements[5]

        training_loss += self.msrl.agent_learn(
            (
                state_list,
                action_list,
                reward_list,
                next_state_list,
                miu_list,
                sigma_list,
            )
        )
        return training_loss, training_reward, j

    @jit
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
        return avg_eval_reward
