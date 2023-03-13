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
'''Advantage Actor Critic'''

from mindspore_rl.agent.learner import Learner
from mindspore_rl.agent.actor import Actor
from mindspore_rl.utils import DiscountedReturn
from mindspore_rl.utils import TensorArray
import mindspore
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import Parameter
import mindspore.ops as ops
import mindspore.nn.probability.distribution as msd
from mindspore.ops import operations as P
import numpy as np

SEED = 42
np.random.seed(SEED)


class A2CPolicyAndNetwork():
    '''A2CPolicyAndNetwork'''

    class ActorCriticNet(nn.Cell):
        '''ActorCriticNet'''

        def __init__(self, input_size, hidden_size, output_size, compute_type=mindspore.float32):
            super().__init__()
            self.common = nn.Dense(
                input_size, hidden_size, weight_init='XavierUniform').to_float(compute_type)
            self.actor = nn.Dense(hidden_size, output_size,
                                  weight_init='XavierUniform').to_float(compute_type)
            self.critic = nn.Dense(hidden_size, 1, weight_init='XavierUniform').to_float(compute_type)
            self.relu = nn.LeakyReLU()
            self.cast = ops.Cast()

        def construct(self, x):
            x = self.common(x)
            x = self.relu(x)
            return self.cast(self.actor(x), mindspore.float32), self.cast(self.critic(x), mindspore.float32)

    class Loss(nn.Cell):
        '''Actor-Critic loss'''

        def __init__(self, a2c_net):
            super().__init__(auto_prefix=False)
            self.a2c_net = a2c_net
            self.reduce_sum = ops.ReduceSum(keep_dims=False)
            self.log = ops.Log()
            self.gather = ops.GatherD()
            self.softmax = ops.Softmax()
            self.smoothl1_loss = nn.SmoothL1Loss(beta=1.0, reduction='sum')

        def construct(self, states, actions, returns):
            '''Calculate actor loss and critic loss'''
            action_logits_ts, values = self.a2c_net(states)
            action_probs_t = self.softmax(action_logits_ts)
            action_probs = self.gather(action_probs_t, 1, actions)
            advantage = returns - values
            action_log_probs = self.log(action_probs)
            adv_mul_prob = action_log_probs * advantage
            actor_loss = -self.reduce_sum(adv_mul_prob)
            critic_loss = self.smoothl1_loss(values, returns)
            return critic_loss + actor_loss

    def __init__(self, params):
        self.a2c_net = self.ActorCriticNet(params.get('state_space_dim'), params.get('hidden_size'),
                                           params.get('action_space_dim'), params.get('compute_type'))
        optimizer = nn.Adam(self.a2c_net.trainable_params(),
                            learning_rate=params['lr'])
        loss_net = self.Loss(self.a2c_net)
        self.a2c_net_train = nn.TrainOneStepCell(loss_net, optimizer)
        self.a2c_net_train.set_train(mode=True)


#pylint: disable=W0223
class A2CActor(Actor):
    '''A2C Actor'''

    def __init__(self, params=None):
        super(A2CActor, self).__init__()
        self._params_config = params
        self.a2c_net = params['a2c_net']
        self._environment = params['collect_environment']
        loop_size = 200
        self.loop_size = Tensor(loop_size, mindspore.int64)
        self.c_dist = msd.Categorical(dtype=mindspore.float32, seed=SEED)
        self.expand_dims = P.ExpandDims()
        self.reshape = P.Reshape()
        self.cast = P.Cast()
        self.softmax = ops.Softmax()
        self.zero = Tensor(0, mindspore.int64)
        self.start = Parameter(self.zero)
        self.done = Tensor(True, mindspore.bool_)
        self.states = TensorArray(mindspore.float32, (4,), dynamic_size=False, size=loop_size)
        self.actions = TensorArray(mindspore.int32, (1,), dynamic_size=False, size=loop_size)
        self.rewards = TensorArray(mindspore.float32, (1,), dynamic_size=False, size=loop_size)
        self.masks = Tensor(np.zeros([loop_size, 1], dtype=np.bool_), mindspore.bool_)
        self.mask_done = Tensor([1], mindspore.bool_)
        self.print = P.Print()

    def act(self, phase, params):
        '''Store returns into TensorArrays from env'''
        if phase == 2:
            t = self.start
            done_status = self.zero
            done_num = self.zero
            masks = self.masks
            while t < self.loop_size:
                self.states.write(t, params)
                ts0 = self.expand_dims(params, 0)
                action_logits, _ = self.a2c_net(ts0)
                action_probs_t = self.softmax(action_logits)
                action = self.reshape(self.c_dist.sample(
                    (1,), probs=action_probs_t), (1,))
                action = self.cast(action, mindspore.int32)
                self.actions.write(t, action)
                new_state, reward, done = self._environment.step(action)
                self.rewards.write(t, reward)
                params = new_state
                if done == self.done:
                    if done_status == self.zero:
                        done_status += 1
                        done_num = t
                    masks[t] = self.mask_done
                    self._environment.reset()
                t += 1
            rewards = self.rewards.stack()
            states = self.states.stack()
            actions = self.actions.stack()
            self.rewards.clear()
            self.states.clear()
            self.actions.clear()
            return rewards, states, actions, masks, done_num
        self.print("Phase is incorrect")
        return 0


class A2CLearner(Learner):
    '''A2C Learner'''

    def __init__(self, params):
        super(A2CLearner, self).__init__()
        self._params_config = params
        self._a2c_net_train = params['a2c_net_train']
        self.shape = ops.DynamicShape()
        self.moments = nn.Moments(keep_dims=False)
        self.sqrt = ops.Sqrt()
        self.zero = Tensor(0, mindspore.int64)
        self.epsilon = Tensor(1.1920929e-07, mindspore.float32)
        self.zero_float = Tensor([0.0], mindspore.float32)
        self.discount_return = DiscountedReturn(gamma=self._params_config['gamma'])

    def learn(self, experience):
        '''Calculate the loss and update'''
        rewards = experience[0]
        states = experience[1]
        actions = experience[2]
        masks = experience[3]
        returns = self.discount_return(rewards, masks, self.zero_float)
        adv_mean, adv_var = self.moments(returns)
        normalized_returns = (returns - adv_mean) / \
            (self.sqrt(adv_var) + self.epsilon)
        a2c_loss = self._a2c_net_train(states, actions, normalized_returns)
        return a2c_loss
