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
"""
TD3 Custom Callbacks.
"""

import time
import mindspore.nn as nn
import mindspore.numpy as np
from mindspore import Tensor, float32, int32
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter
from mindspore.train.summary import SummaryRecord
from mindspore_rl.utils.callback import Callback, EvaluateCallback


class RecordQueue(nn.Cell):
    '''A Deque for storing scale data'''
    def __init__(self, capacity=10, dtype=float32):
        super().__init__()
        self.queue = Parameter(Tensor(np.zeros((capacity,)), dtype), requires_grad=False)
        self._capacity = capacity
        self.count = Parameter(Tensor(0, int32), name="count", requires_grad=False)
        self.head = Parameter(Tensor(0, int32), name="head", requires_grad=False)
        self.zero = Tensor(0, int32)

        self.reshape = P.Reshape()
        self.assign = P.Assign()
        self.assign_add = P.AssignAdd()
        self.reduce_sum = P.ReduceSum()
        self.reduce_mean = P.ReduceMean()

        self.greater_equal = P.GreaterEqual()
        self.capacity_tensor = Tensor([capacity,], int32)

    def add(self, record):
        self.queue[self.head] = record
        self.assign_add(self.count, 1)
        self.assign_add(self.head, 1)
        if self.head >= self._capacity:
            self.assign(self.head, self.zero)
        return True

    def reset(self):
        success = self.assign(self.count, self.zero)
        return success

    def mean(self):
        if self.count < self._capacity:
            result = self.reduce_sum(self.queue) / self.count
        else:
            result = self.reduce_mean(self.queue)
        return result


class SummaryCallback(Callback):
    '''Callback for data summary in MindInsight'''
    def __init__(self, collect_sec, eval_rate, summary_dir):
        super(SummaryCallback).__init__()
        self.summary_dir = summary_dir
        self._eval_rate = eval_rate
        self.collect_sec = collect_sec
        self.train_rewards = RecordQueue()
        self.eval_rewards = RecordQueue()
        self.last_eval_update = -1
        self.last_time = time.time()
        self.summary_record = SummaryRecord(self.summary_dir)

    def __enter__(self):
        return self

    def __exit__(self, *err):
        self.summary_record.close()

    def episode_end(self, params):
        cur_time = time.time()
        self.train_rewards.add(params.total_rewards)

        if params.eval_episode != self.last_eval_update:
            self.eval_rewards.add(params.eval_reward)

        if cur_time - self.last_time >= self.collect_sec:
            if self.last_eval_update == -1 or params.cur_episode - self.last_eval_update >= self._eval_rate:
                self.summary_record.add_value('scalar', 'AvgReward-eval', self.eval_rewards.mean())
                self.last_eval_update = params.eval_episode

            self.summary_record.add_value('scalar', 'loss', params.loss)
            self.summary_record.add_value('scalar', 'AvgReward-train', self.train_rewards.mean())
            self.summary_record.record(params.cur_episode)
            self.summary_record.flush()
            self.last_time = time.time()


class EvalRecordCallback(EvaluateCallback):
    r'''
    Evaluate callback for MindInsight record.

    Args:
        eval_rate (int): The frequency to eval.
    '''
    def episode_end(self, params):
        '''Run evaluate in the end of episode, and print the rewards.'''
        if self._eval_rate != 0 and \
            params.cur_episode % self._eval_rate == 0:
            # Call the `evaluate` function provided by user.
            rewards = params.evaluate()
            if params.cur_episode != 0:
                print("-----------------------------------------")
                print("Evaluate for episode {} total rewards is {:5.3f}" \
                    .format(params.cur_episode, rewards.asnumpy()), flush=True)
                print("-----------------------------------------")
            params['eval_reward'] = rewards
            params['eval_episode'] = params.cur_episode
