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
'''fragments'''
import time
import mindspore
import mindspore.numpy as np
import mindspore.nn as nn
import mindspore.ops.operations as P
from mindspore.common.api import ms_function
from mindspore import Tensor
from mindspore.common.parameter import Parameter, ParameterTuple
from mindspore.communication.management import NCCL_WORLD_COMM_GROUP
import mindspore.nn.probability.distribution as msd

#pylint: disable=W0212
#pylint: disable=W0612


class Fragment1Kernel(nn.Cell):
    '''Fragment 1 kernel'''
    def __init__(self, msrl, rank, duration):
        super(Fragment1Kernel, self).__init__()
        self.msrl = msrl
        self.rank = rank
        self.duration = duration
        self.actor = msrl.actors
        self.num_actor = msrl.num_actors
        self.num_collect_env = msrl.num_collect_env

        self.zero = Tensor(0, mindspore.float32)
        self.actor_net_param = ParameterTuple(self.actor._actor_net.get_parameters())
        self.actor_net = self.actor._actor_net

        self.state_list = Parameter(Tensor(np.zeros((1, self.num_collect_env, self.duration, 17)),
                                           mindspore.float32), name="F1state_list")
        self.action_list = Parameter(Tensor(np.zeros((1, self.num_collect_env, self.duration, 6)),
                                            mindspore.float32), name="F1action_list")
        self.reward_list = Parameter(Tensor(np.zeros((1, self.num_collect_env, self.duration, 1)),
                                            mindspore.float32), name="F1reward_list")
        self.next_state_list = Parameter(Tensor(np.zeros((1, self.num_collect_env, self.duration, 17)),
                                                mindspore.float32), name="F1next_state_list")
        self.miu_list = Parameter(Tensor(np.zeros((1, self.num_collect_env, self.duration, 6)),
                                         mindspore.float32), name="F1_miu_list")
        self.sigma_list = Parameter(Tensor(np.zeros((1, self.num_collect_env, self.duration, 6)),
                                           mindspore.float32), name="F1sigma_list")

        self.network_weigt_0 = Parameter(Tensor(np.zeros((6)), mindspore.float32), name="F1w0")
        self.network_weigt_1 = Parameter(Tensor(np.zeros((200, 17)), mindspore.float32), name="F1w1")
        self.network_weigt_2 = Parameter(Tensor(np.zeros((200)), mindspore.float32), name="F1w2")
        self.network_weigt_3 = Parameter(Tensor(np.zeros((100, 200)), mindspore.float32), name="F1w3")
        self.network_weigt_4 = Parameter(Tensor(np.zeros((100)), mindspore.float32), name="F1w4")
        self.network_weigt_5 = Parameter(Tensor(np.zeros((6, 100)), mindspore.float32), name="F1w5")
        self.network_weigt_6 = Parameter(Tensor(np.zeros((6)), mindspore.float32), name="F1w6")

        self.slice = P.Slice()
        self.less = P.Less()
        self.reduce_mean = P.ReduceMean()
        self.assign = P.Assign()
        self.all_gather = P.AllGather(group=NCCL_WORLD_COMM_GROUP)
        self.norm_dist = msd.Normal()
        self.depend = P.Depend()
        self.expand_dims = P.ExpandDims()

    @ms_function
    def execute(self):
        '''execute'''
        training_reward = self.zero
        state = self.msrl.collect_environment.reset()
        self.msrl.replay_buffer_reset()

        i = self.zero
        while self.less(i, self.duration):
            reward, new_state, action, miu, sigma = self.msrl.agent_act(state)
            self.msrl.replay_buffer_insert([state, action, reward, new_state, miu, sigma])
            state = new_state
            reward = self.reduce_mean(reward)
            training_reward += reward
            i += 1

        replay_buffer_elements = self.msrl.get_replay_buffer_elements(transpose=True, shape=(1, 0, 2))

        tmp_state_list = replay_buffer_elements[0]
        tmp_action_list = replay_buffer_elements[1]
        tmp_reward_list = replay_buffer_elements[2]
        tmp_next_state_list = replay_buffer_elements[3]
        tmp_miu_list = replay_buffer_elements[4]
        tmp_sigma_list = replay_buffer_elements[5]

        tmp_state_list = self.expand_dims(tmp_state_list, 0)
        tmp_action_list = self.expand_dims(tmp_action_list, 0)
        tmp_reward_list = self.expand_dims(tmp_reward_list, 0)
        tmp_next_state_list = self.expand_dims(tmp_next_state_list, 0)
        tmp_miu_list = self.expand_dims(tmp_miu_list, 0)
        tmp_sigma_list = self.expand_dims(tmp_sigma_list, 0)

        self.assign(self.state_list, tmp_state_list)
        self.assign(self.action_list, tmp_action_list)
        self.assign(self.reward_list, tmp_reward_list)
        self.assign(self.next_state_list, tmp_next_state_list)
        self.assign(self.miu_list, tmp_miu_list)
        self.assign(self.sigma_list, tmp_sigma_list)

        state_fused = self.all_gather(self.state_list)
        action_list = self.depend(self.action_list, state_fused)

        action_fused = self.all_gather(action_list)
        reward_list = self.depend(self.reward_list, action_fused)

        reward_fused = self.all_gather(reward_list)
        next_state_list = self.depend(self.next_state_list, reward_fused)

        next_state_list_fused = self.all_gather(next_state_list)
        miu_list = self.depend(self.miu_list, next_state_list_fused)

        miu_fused = self.all_gather(miu_list)
        sigma_list = self.depend(self.sigma_list, miu_fused)

        sigma_fused = self.all_gather(sigma_list)


        nw0 = self.depend(self.network_weigt_0, sigma_fused)
        w0 = self.all_gather(nw0)

        nw1 = self.depend(self.network_weigt_1, w0)
        w1 = self.all_gather(nw1)

        nw2 = self.depend(self.network_weigt_2, w1)
        w2 = self.all_gather(nw2)

        nw3 = self.depend(self.network_weigt_3, w2)
        w3 = self.all_gather(nw3)

        nw4 = self.depend(self.network_weigt_4, w3)
        w4 = self.all_gather(nw4)

        nw5 = self.depend(self.network_weigt_5, w4)
        w5 = self.all_gather(nw5)

        nw6 = self.depend(self.network_weigt_6, w5)
        w6 = self.all_gather(nw6)

        t0 = self.slice(w0, (0,), (6,))
        t1 = self.slice(w1, (0, 0), (200, 17))
        t2 = self.slice(w2, (0,), (200,))
        t3 = self.slice(w3, (0, 0), (100, 200))
        t4 = self.slice(w4, (0,), (100,))
        t5 = self.slice(w5, (0, 0), (6, 100))
        t6 = self.slice(w6, (0,), (6,))

        self.assign(self.actor_net_param[0], t0)
        self.assign(self.actor_net_param[1], t1)
        self.assign(self.actor_net_param[2], t2)
        self.assign(self.actor_net_param[3], t3)
        self.assign(self.actor_net_param[4], t4)
        self.assign(self.actor_net_param[5], t5)
        self.assign(self.actor_net_param[6], t6)

        return training_reward


class Fragment1():
    '''Fragment 1'''
    def __init__(self, msrl, rank, duration, episode):
        self.actor = Fragment1Kernel(msrl, rank, duration)
        self.episode = episode

    def run(self):
        '''run'''
        for i in range(self.episode):
            reward = self.actor.execute()
            print("episode: {}, reward: {}".format(i, reward))
        return reward.asnumpy().tolist()


class Fragment2Kernel(nn.Cell):
    '''Fragment 2 kernel'''
    def __init__(self, msrl, rank, duration):
        super(Fragment2Kernel, self).__init__()
        self.msrl = msrl
        self.rank = rank
        self.duration = duration

        self.learner = self.msrl.learner
        self.actor_net_param = ParameterTuple(self.learner._actor_net.get_parameters())
        self.num_actor = msrl.num_actors
        self.num_collect_env = msrl.num_collect_env

        self.zero = Tensor(0, mindspore.float32)
        self.state_list = Parameter(Tensor(np.zeros((1, self.num_collect_env, self.duration, 17)),
                                           mindspore.float32), name="F2state_list")
        self.action_list = Parameter(Tensor(np.zeros((1, self.num_collect_env, self.duration, 6)),
                                            mindspore.float32), name="F2action_list")
        self.reward_list = Parameter(Tensor(np.zeros((1, self.num_collect_env, self.duration, 1)),
                                            mindspore.float32), name="F2reward_list")
        self.next_state_list = Parameter(Tensor(np.zeros((1, self.num_collect_env, self.duration, 17)),
                                                mindspore.float32), name="F2next_state_list")
        self.miu_list = Parameter(Tensor(np.zeros((1, self.num_collect_env, self.duration, 6)),
                                         mindspore.float32), name="F2miu_list")
        self.sigma_list = Parameter(Tensor(np.zeros((1, self.num_collect_env, self.duration, 6)),
                                           mindspore.float32), name="F2sigma_list")
        self.network_weigt_0 = Parameter(Tensor(np.zeros((6)), mindspore.float32), name="F2w0")
        self.network_weigt_1 = Parameter(Tensor(np.zeros((200, 17)), mindspore.float32), name="F2w1")
        self.network_weigt_2 = Parameter(Tensor(np.zeros((200)), mindspore.float32), name="F2w2")
        self.network_weigt_3 = Parameter(Tensor(np.zeros((100, 200)), mindspore.float32), name="F2w3")
        self.network_weigt_4 = Parameter(Tensor(np.zeros((100)), mindspore.float32), name="F2w4")
        self.network_weigt_5 = Parameter(Tensor(np.zeros((6, 100)), mindspore.float32), name="F2w5")
        self.network_weigt_6 = Parameter(Tensor(np.zeros((6)), mindspore.float32), name="F2w6")
        self.assign = P.Assign()
        self.slice = P.Slice()
        self.all_gather = P.AllGather(group=NCCL_WORLD_COMM_GROUP)
        self.depend = P.Depend()

    @ms_function
    def execute(self):
        '''execute'''
        training_loss = self.zero

        state_fused = self.all_gather(self.state_list)
        action_list = self.depend(self.action_list, state_fused)
        action_fused = self.all_gather(action_list)
        reward_list = self.depend(self.reward_list, action_fused)
        reward_fused = self.all_gather(reward_list)
        next_state_list = self.depend(self.next_state_list, reward_fused)
        next_state_list_fused = self.all_gather(next_state_list)
        miu_list = self.depend(self.miu_list, next_state_list_fused)
        miu_fused = self.all_gather(miu_list)
        sigma_list = self.depend(self.sigma_list, miu_fused)
        sigma_fused = self.all_gather(sigma_list)

        state_list = self.slice(state_fused, (1, 0, 0, 0), (self.num_actor, self.num_collect_env, self.duration, 17))
        action_list = self.slice(action_fused, (1, 0, 0, 0), (self.num_actor, self.num_collect_env, self.duration, 6))
        reward_list = self.slice(reward_fused, (1, 0, 0, 0), (self.num_actor, self.num_collect_env, self.duration, 1))
        next_state_list = self.slice(next_state_list_fused, (1, 0, 0, 0),
                                     (self.num_actor, self.num_collect_env, self.duration, 17))
        miu_list = self.slice(miu_fused, (1, 0, 0, 0), (self.num_actor, self.num_collect_env, self.duration, 6))
        sigma_list = self.slice(sigma_fused, (1, 0, 0, 0), (self.num_actor, self.num_collect_env, self.duration, 6))

        training_loss += self.learner.learn((state_list, action_list, reward_list,
                                             next_state_list, miu_list, sigma_list))

        actor_net_param = self.depend(self.actor_net_param, training_loss)
        self.assign(self.network_weigt_0, actor_net_param[0])
        self.assign(self.network_weigt_1, actor_net_param[1])
        self.assign(self.network_weigt_2, actor_net_param[2])
        self.assign(self.network_weigt_3, actor_net_param[3])
        self.assign(self.network_weigt_4, actor_net_param[4])
        self.assign(self.network_weigt_5, actor_net_param[5])
        self.assign(self.network_weigt_6, actor_net_param[6])

        nw0 = self.depend(self.network_weigt_0,
                          self.assign(self.network_weigt_0, actor_net_param[0]))
        w0 = self.all_gather(nw0)

        nw1 = self.depend(self.network_weigt_1, w0)
        w1 = self.all_gather(nw1)

        nw2 = self.depend(self.network_weigt_2, w1)
        w2 = self.all_gather(nw2)

        nw3 = self.depend(self.network_weigt_3, w2)
        w3 = self.all_gather(nw3)

        nw4 = self.depend(self.network_weigt_4, w3)
        w4 = self.all_gather(nw4)

        nw5 = self.depend(self.network_weigt_5, w4)
        w5 = self.all_gather(nw5)

        nw6 = self.depend(self.network_weigt_6, w5)
        w6 = self.all_gather(nw6)

        return training_loss


class Fragment2():
    '''Fragment 2'''
    def __init__(self, msrl, rank, duration, episode):
        self.learner = Fragment2Kernel(msrl, rank, duration)
        self.episode = episode

    def run(self):
        '''run'''
        for i in range(self.episode):
            start = time.perf_counter()
            loss = self.learner.execute()
            end = time.perf_counter()
            timespan = end - start
            print("episode: {} training time: {}s".format(i, timespan))
            print("--------------------------")
            print()
        return loss


def get_all_fragments(num_actors):
    '''get all fragments'''
    flist = [Fragment2]
    for _ in range(num_actors):
        flist.append(Fragment1)
    return flist
