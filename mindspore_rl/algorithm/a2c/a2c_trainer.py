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
"""A2C Trainer"""
import collections
import statistics
import tqdm

from mindspore_rl.agent.trainer import Trainer
from mindspore_rl.agent import trainer
from mindspore.ops import operations as ops
from mindspore import ms_function


class A2CTrainer(Trainer):
    '''A2CTrainer'''
    def __init__(self, msrl):
        super(A2CTrainer, self).__init__(msrl)
        self.reduce_sum = ops.ReduceSum()

    def train(self, episodes, callbacks=None, ckpt_path=None):
        '''Train A2C'''
        running_reward = 0
        episode_reward: collections.deque = collections.deque(maxlen=100)
        with tqdm.trange(episodes) as t:
            for i in t:
                loss, reward = self.train_one_episode()
                episode_reward.append(reward.asnumpy().tolist())
                running_reward = statistics.mean(episode_reward)
                t.set_description(f'Episode {i}')
                t.set_postfix(episode_reward=reward.asnumpy(), loss=loss.asnumpy(), running_reward=running_reward)
                if running_reward > 195 and i >= 100:
                    print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}.')
                    break
                if i == episodes - 1:
                    print(f'\nFailed to solved this problem after running {episodes} episodes.')

    @ms_function
    def train_one_episode(self):
        '''Train one episode'''
        state = self.msrl.collect_environment.reset()
        rewards, states, actions, masks, done_num = self.msrl.agent_act(trainer.COLLECT, state)
        a2c_loss = self.msrl.agent_learn([rewards, states, actions, masks])
        return a2c_loss, done_num

    def evaluate(self):
        '''Default evaluate'''
        return

    def trainable_variables(self):
        '''Default trainable variables'''
        return
