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
"""DQN"""

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter, ParameterTuple
from mindspore_rl.agent.actor import Actor
from mindspore_rl.agent.learner import Learner
from mindspore_rl.network import FullyConnectedNet
from mindspore_rl.policy import EpsilonGreedyPolicy
from mindspore_rl.policy import GreedyPolicy
from mindspore_rl.policy import RandomPolicy

_update_opt = C.MultitypeFuncGraph("update_opt")


@_update_opt.register("Tensor", "Tensor")
def _parameter_update(policy_param, target_param):
    assign = P.Assign()
    new_param = (1 - 0.05) * target_param + 0.05 * policy_param
    output = assign(target_param, new_param)
    return output


class DQNPolicy():
    """DQN Policy"""

    def __init__(self, params):
        self.policy_network = FullyConnectedNet(
            params['state_space_dim'],
            params['hidden_size'],
            params['action_space_dim'],
            params['compute_type'])
        self.target_network = FullyConnectedNet(
            params['state_space_dim'],
            params['hidden_size'],
            params['action_space_dim'],
            params['compute_type'])

        self.init_policy = RandomPolicy(params['action_space_dim'])
        self.collect_policy = EpsilonGreedyPolicy(self.policy_network, (1, 1), params['epsi_high'],
                                                  params['epsi_low'], params['decay'], params['action_space_dim'])
        self.evaluate_policy = GreedyPolicy(self.policy_network)


class DQNActor(Actor):
    """DQN Actor"""

    def __init__(self, params):
        super(DQNActor, self).__init__()
        self.init_policy = params['init_policy']
        self.collect_policy = params['collect_policy']
        self.evaluate_policy = params['evaluate_policy']
        self._environment = params['collect_environment']
        self._eval_env = params['eval_environment']
        self.replay_buffer = params['replay_buffer']
        self.step = Parameter(
            Tensor(
                0,
                ms.int32),
            name="step",
            requires_grad=False)
        self.expand_dims = P.ExpandDims()
        self.reshape = P.Reshape()
        self.ones = P.Ones()
        self.abs = P.Abs()
        self.assign = P.Assign()
        self.select = P.Select()
        self.reward = Tensor([1,], ms.float32)
        self.penalty = Tensor([-1,], ms.float32)
        self.print = P.Print()

    def act(self, phase, params):
        '''act func'''
        if phase == 1:
            # Fill the replay buffer
            action = self.init_policy()
            new_state, reward, done = self._environment.step(action)
            action = self.reshape(action, (1,))
            my_reward = self.select(done, self.penalty, self.reward)
            return done, reward, new_state, action, my_reward
        if phase == 2:
            # Experience collection
            self.step += 1
            ts0 = self.expand_dims(params, 0)
            step_tensor = self.ones((1, 1), ms.float32) * self.step

            action = self.collect_policy(ts0, step_tensor)
            new_state, reward, done = self._environment.step(action)
            action = self.reshape(action, (1,))
            my_reward = self.select(done, self.penalty, self.reward)
            return done, reward, new_state, action, my_reward
        if phase == 3:
            # Evaluate the trained policy
            ts0 = self.expand_dims(params, 0)
            action = self.evaluate_policy(ts0)
            new_state, reward, done = self._eval_env.step(action)
            return done, reward, new_state
        self.print("Phase is incorrect")
        return 0

    def get_action(self, phase, params):
        """Default get_action function"""
        return


class DQNLearner(Learner):
    """DQN Learner"""

    class PolicyNetWithLossCell(nn.Cell):
        """DQN policy network with loss cell"""

        def __init__(self, backbone, loss_fn):
            super(DQNLearner.PolicyNetWithLossCell,
                  self).__init__(auto_prefix=False)
            self._backbone = backbone
            self._loss_fn = loss_fn
            self.gather = P.GatherD()

        def construct(self, x, a0, label):
            """constructor for Loss Cell"""
            out = self._backbone(x)
            out = self.gather(out, 1, a0)
            loss = self._loss_fn(out, label)
            return loss

    def __init__(self, params=None):
        super(DQNLearner, self).__init__()
        self.policy_network = params['policy_network']
        self.target_network = params['target_network']
        self.policy_param = ParameterTuple(
            self.policy_network.get_parameters())
        self.target_param = ParameterTuple(
            self.target_network.get_parameters())

        optimizer = nn.Adam(
            self.policy_network.trainable_params(),
            learning_rate=params['lr'])
        loss_fn = nn.MSELoss()
        loss_q_net = self.PolicyNetWithLossCell(self.policy_network, loss_fn)
        self.policy_network_train = nn.TrainOneStepCell(loss_q_net, optimizer)
        self.policy_network_train.set_train(mode=True)

        self.gamma = Tensor(params['gamma'], ms.float32)
        self.expand_dims = P.ExpandDims()
        self.reshape = P.Reshape()
        self.hyper_map = C.HyperMap()
        self.ones_like = P.OnesLike()
        self.select = P.Select()

    def learn(self, experience):
        """Model update"""
        s0, a0, r1, s1 = experience
        next_state_values = self.target_network(s1)
        next_state_values = next_state_values.max(axis=1)
        r1 = self.reshape(r1, (-1,))

        y_true = r1 + self.gamma * next_state_values

        # Modify last step reward
        one = self.ones_like(r1)
        y_true = self.select(r1 == -one, one, y_true)
        y_true = self.expand_dims(y_true, 1)

        success = self.policy_network_train(s0, a0, y_true)
        return success

    def update(self):
        """Update the network parameters"""
        assign_result = self.hyper_map(
            _update_opt,
            self.policy_param,
            self.target_param)
        return assign_result
