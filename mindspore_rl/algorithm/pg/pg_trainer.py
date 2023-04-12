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
"""PG Trainer"""
import mindspore
import numpy as np
from mindspore import Tensor, jit
from mindspore.ops import operations as P

from mindspore_rl.agent import trainer
from mindspore_rl.agent.trainer import Trainer
from mindspore_rl.utils import TensorArray


class PGTrainer(Trainer):
    """PGTrainer"""

    def __init__(self, msrl, params):
        # pylint: disable=R1725
        super(PGTrainer, self).__init__(msrl)
        self.num_evaluate_episode = params["num_evaluate_episode"]
        self.less = P.Less()
        self.false = Tensor(False, mindspore.bool_)
        self.zero_value = Tensor(0, mindspore.int64)
        loop_size = 200
        self.loop_size = Tensor(loop_size, mindspore.int64)
        self.obs_list = TensorArray(
            mindspore.float32, (4,), dynamic_size=False, size=loop_size
        )
        self.reward_list = TensorArray(
            mindspore.float32, (1,), dynamic_size=False, size=loop_size
        )
        self.action_list = TensorArray(
            mindspore.int32, (1,), dynamic_size=False, size=loop_size
        )
        self.masks = Tensor(np.zeros([loop_size, 1], dtype=np.bool_), mindspore.bool_)
        self.mask_done = Tensor([1], mindspore.bool_)

    def trainable_variables(self):
        """Trainable variables for saving."""
        trainable_variables = {"actor_net": self.msrl.learner.actor_net}
        return trainable_variables

    @jit
    def train_one_episode(self):
        """Train one episode"""
        obs_list, action_list, reward_list, masks, steps = self.run_one_episode()
        loss = self.msrl.agent_learn([obs_list, reward_list, action_list, masks])
        return loss, steps, self.loop_size

    @jit
    def run_one_episode(self):
        """run_one_episode(dynamic list is not supported in graph mode, so use static loop.)"""
        steps = self.zero_value
        done_status = self.zero_value
        done_num = self.zero_value
        masks = self.masks
        obs = self.msrl.collect_environment.reset()
        while steps < self.loop_size:
            self.obs_list.write(steps, obs)
            done, r, obs, a = self.msrl.agent_act(trainer.COLLECT, obs)
            self.action_list.write(steps, a)
            self.reward_list.write(steps, r)
            if done:
                if done_status == self.zero_value:
                    done_status += 1
                    done_num = steps
                masks[steps] = self.mask_done
                self.msrl.collect_environment.reset()
            steps += 1
        states = self.obs_list.stack()
        rewards = self.reward_list.stack()
        actions = self.action_list.stack()
        self.obs_list.clear()
        self.reward_list.clear()
        self.action_list.clear()
        return states, actions, rewards, masks, done_num

    @jit
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
                r = r.squeeze()
                episode_reward += r
            total_reward += episode_reward
            eval_iter += 1
        avg_reward = total_reward / self.num_evaluate_episode
        return avg_reward
