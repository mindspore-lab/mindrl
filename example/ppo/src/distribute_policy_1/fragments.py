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
"""Fragments"""
import time

import mindspore
import mindspore.nn as nn
import mindspore.nn.probability.distribution as msd
import mindspore.numpy as np
from mindspore import Tensor
from mindspore.common.api import ms_function
from mindspore.common.parameter import Parameter, ParameterTuple
from mindspore.communication.management import NCCL_WORLD_COMM_GROUP
from mindspore.ops import operations as P
from mindspore.ops.operations._inner_ops import Receive, Send


# pylint: disable=W0212
# pylint: disable=W0613
# pylint: disable=W0612
class Fragment2Kernel(nn.Cell):
    """Fragment2 kernel"""

    def __init__(self, msrl, rank):
        super(Fragment2Kernel, self).__init__()
        self.msrl = msrl
        self.learner = self.msrl.learner
        self.actor_net_param = ParameterTuple(self.learner._actor_net.get_parameters())
        self.x = ParameterTuple(self.msrl.policy_and_network.actor_net.get_parameters())
        self.zero = Tensor(0, mindspore.float32)

    @mindspore.jit
    def learn_no_comm(self):
        """learn"""
        training_loss = self.zero
        replay_buffer_elements = self.msrl.get_replay_buffer_elements(
            transpose=True, shape=(1, 2, 0, 3)
        )
        state_list = replay_buffer_elements[0]
        action_list = replay_buffer_elements[1]
        reward_list = replay_buffer_elements[2]
        next_state_list = replay_buffer_elements[3]
        miu_list = replay_buffer_elements[4]
        sigma_list = replay_buffer_elements[5]

        training_loss += self.learner.learn(
            (
                state_list,
                action_list,
                reward_list,
                next_state_list,
                miu_list,
                sigma_list,
            )
        )
        self.msrl.replay_buffer_reset()

        return training_loss


class FragmentActionkernel(nn.Cell):
    """Fragment action kernel"""

    def __init__(self, msrl, rank):
        super(FragmentActionkernel, self).__init__()
        self.msrl = msrl
        self.actor_net = self.msrl.learner._actor_net
        self.num_actor = msrl.num_actors
        self.num_collect_env = msrl.num_collect_env

        self.state = Parameter(
            Tensor(np.zeros((self.num_collect_env, 17)), mindspore.float32),
            name="F2state",
        )
        self.new_state = Parameter(
            Tensor(np.zeros((self.num_collect_env, 17)), mindspore.float32),
            name="F2new_state",
        )
        self.action = Parameter(
            Tensor(np.zeros((self.num_collect_env, 6)), mindspore.float32),
            name="F2action",
        )
        self.reward = Parameter(
            Tensor(np.zeros((self.num_collect_env, 1)), mindspore.float32),
            name="F2reward",
        )

        self.state_fused = Parameter(
            Tensor(
                np.zeros(((self.num_actor + 1) * self.num_collect_env, 17)),
                mindspore.float32,
            ),
            name="F2state_fused",
        )
        self.new_state_fused = Parameter(
            Tensor(
                np.zeros(((self.num_actor + 1) * self.num_collect_env, 17)),
                mindspore.float32,
            ),
            name="F2new_state_fused",
        )

        self.true = Tensor(1, mindspore.float32)
        self.flag = Tensor(0, mindspore.float32)

        self.depend = P.Depend()
        self.slice = P.Slice()
        self.assign = P.Assign()
        self.concat = P.Concat()
        self.expand_dims = P.ExpandDims()

        self.all_gather = P.AllGather(group=NCCL_WORLD_COMM_GROUP)

        self.norm_dist_list = [None] * self.num_actor
        for i in range(self.num_actor):
            self.norm_dist_list[i] = msd.Normal()

        self.send_list = [None] * self.num_actor
        for i in range(self.num_actor):
            self.send_list[i] = Send(
                sr_tag=(i + 1), dest_rank=(i + 1), group=NCCL_WORLD_COMM_GROUP
            )

    @mindspore.jit
    def gather_state(self):
        """all_gather"""
        state_fused = self.all_gather(self.state)
        self.assign(self.state_fused, state_fused)
        return self.true

    @mindspore.jit
    def execution(self):
        """execution"""
        state_list = []
        new_state_list = []
        action_list = []
        reward_list = []
        miu_list = []
        sigma_list = []
        i = 0
        actions = self.action
        while i < self.num_actor:
            state = self.slice(
                self.state_fused,
                ((i + 1) * self.num_collect_env, 0),
                (self.num_collect_env, 17),
            )
            state_list.append(state)

            miu, sigma = self.actor_net(state)
            action = self.norm_dist_list[i].sample((), miu, sigma)
            action_list.append(action)
            miu_list.append(miu)
            sigma_list.append(sigma)

            action = self.depend(action, self.send_list[i](action))
            actions = actions + action
            i += 1

        new_state = self.depend(self.new_state, actions)
        new_state_fused = self.all_gather(new_state)
        self.assign(self.new_state_fused, new_state_fused)

        reward = self.depend(self.reward, new_state_fused)
        reward_fused = self.all_gather(reward)
        new_state_fused = self.depend(self.new_state_fused, reward_fused)

        i = 0
        while i < self.num_actor:
            new_state = self.slice(
                new_state_fused,
                ((i + 1) * self.num_collect_env, 0),
                (self.num_collect_env, 17),
            )
            new_state_list.append(new_state)
            i += 1

        i = 0
        while i < self.num_actor:
            reward = self.slice(
                reward_fused,
                ((i + 1) * self.num_collect_env, 0),
                (self.num_collect_env, 1),
            )
            reward_list.append(reward)
            i += 1

        state = state_list[0]
        action = action_list[0]
        new_state = new_state_list[0]
        reward = reward_list[0]
        miu = miu_list[0]
        sigma = sigma_list[0]

        state = self.expand_dims(state, 0)
        action = self.expand_dims(action, 0)
        reward = self.expand_dims(reward, 0)
        new_state = self.expand_dims(new_state, 0)
        miu = self.expand_dims(miu, 0)
        sigma = self.expand_dims(sigma, 0)

        i = 1
        while i < self.num_actor:
            state_temp = state_list[i]
            state_temp = self.expand_dims(state_temp, 0)

            new_state_temp = new_state_list[i]
            new_state_temp = self.expand_dims(new_state_temp, 0)

            action_temp = action_list[i]
            action_temp = self.expand_dims(action_temp, 0)

            reward_temp = reward_list[i]
            reward_temp = self.expand_dims(reward_temp, 0)

            miu_temp = miu_list[i]
            miu_temp = self.expand_dims(miu_temp, 0)

            sigma_temp = sigma_list[i]
            sigma_temp = self.expand_dims(sigma_temp, 0)

            state = self.concat((state, state_temp))
            new_state = self.concat((new_state, new_state_temp))
            action = self.concat((action, action_temp))
            reward = self.concat((reward, reward_temp))
            miu = self.concat((miu, miu_temp))
            sigma = self.concat((sigma, sigma_temp))
            i += 1

        self.msrl.replay_buffer_insert([state, action, reward, new_state, miu, sigma])
        self.assign(self.state_fused, self.new_state_fused)
        return self.true


class Fragment1Kernel(nn.Cell):
    """Fragment 1 kernel"""

    def __init__(self, msrl, rank):
        super(Fragment1Kernel, self).__init__()
        self.msrl = msrl
        self.rank = rank
        self.num_actor = msrl.num_actors
        self.num_collect_env = msrl.num_collect_env
        self.all_gather = P.AllGather(group=NCCL_WORLD_COMM_GROUP)
        self.slice = P.Slice()
        self.assign = P.Assign()
        self.actions = Parameter(
            Tensor(
                np.ones(((self.num_actor + 1), self.num_collect_env, 6)),
                mindspore.float32,
            ),
            name="F1actions",
        )
        self.send_reward = Send(sr_tag=0, dest_rank=0, group=NCCL_WORLD_COMM_GROUP)
        self.recv_action = Receive(
            sr_tag=self.rank,
            src_rank=0,
            shape=[self.num_collect_env, 6],
            dtype=mindspore.float32,
            group=NCCL_WORLD_COMM_GROUP,
        )
        self.depend = P.Depend()
        self.true = Tensor(1, mindspore.float32)
        self.flag = Tensor(0, mindspore.float32)

    @mindspore.jit
    def gather_state(self):
        """gather"""
        state = self.msrl.collect_environment.reset()
        state_fused = self.all_gather(state)
        return self.true

    @mindspore.jit
    def execution(self):
        """execute"""
        action = self.recv_action()

        new_state, reward, _ = self.msrl.collect_environment.step(action)
        new_state_fused = self.all_gather(new_state)

        reward = self.depend(reward, new_state_fused)
        reward_fused = self.all_gather(reward)
        return reward


class Fragment1:
    """Fragment 1"""

    def __init__(self, msrl, rank, duration, episode):
        self.stepper = Fragment1Kernel(msrl, rank)
        self.duration = duration
        self.episode = episode
        self.reduce_mean = P.ReduceMean()

    def run(self):
        """run"""
        for i in range(self.episode):
            _ = self.stepper.gather_state()
            reward = Tensor(0, mindspore.float32)
            for _ in range(self.duration):
                r = self.stepper.execution()
                r = self.reduce_mean(r)
                reward += r
            print("episode: {}, reward: {}".format(i, reward))
        return True


class Fragment2:
    """Fragment 2"""

    def __init__(self, msrl, rank, duration, episode):
        self.msrl = msrl
        self.action = FragmentActionkernel(msrl, rank)
        self.learner = Fragment2Kernel(msrl, rank)
        self.duration = duration
        self.episode = episode

    def run(self):
        """run"""
        for i in range(self.episode):
            start = time.perf_counter()
            flag = self.action.gather_state()
            for j in range(self.duration):
                flag = self.action.execution()
            training_loss = self.learner.learn_no_comm()

            end = time.perf_counter()
            timespan = end - start
            print("episode training time: {}s".format(timespan))
            print("--------------------------")
            print()
        return True


def get_all_fragments(num_actors):
    """get all fragments"""
    flist = [Fragment2]
    for _ in range(num_actors):
        flist.append(Fragment1)
    return flist
