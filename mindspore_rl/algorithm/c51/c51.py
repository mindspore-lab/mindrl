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
"""C51"""

import mindspore as ms
import mindspore.numpy as mnp
from mindspore import Tensor, nn, ops
from mindspore.common.parameter import Parameter, ParameterTuple
from mindspore.ops import composite as C
from mindspore.ops import operations as P

from mindspore_rl.agent.actor import Actor
from mindspore_rl.agent.learner import Learner
from mindspore_rl.policy import RandomPolicy
from mindspore_rl.utils.discounted_return import DiscountedReturn
from mindspore_rl.utils.soft_update import SoftUpdate

from .c51policy import (
    EpsilonGreedyPolicyForValueDistribution,
    GreedyPolicyForValueDistribution,
)
from .fullyconnectednet_noisy import FullyConnectedNet


class CategoricalDQNPolicy:
    """C51 Policy"""

    def __init__(self, params):
        self.policy_network = FullyConnectedNet(
            params["state_space_dim"],
            params["hidden_size"],
            params["action_space_dim"] * params["atoms_num"],
            params["action_space_dim"],
            params["atoms_num"],
            params["compute_type"],
        )
        self.target_network = FullyConnectedNet(
            params["state_space_dim"],
            params["hidden_size"],
            params["action_space_dim"] * params["atoms_num"],
            params["action_space_dim"],
            params["atoms_num"],
            params["compute_type"],
        )

        self.init_policy = RandomPolicy(params["action_space_dim"])
        self.collect_policy = EpsilonGreedyPolicyForValueDistribution(
            self.policy_network,
            (1, 1),
            params["epsi_high"],
            params["epsi_low"],
            params["decay"],
            params["atoms_num"],
            params["v_min"],
            params["v_max"],
            params["action_space_dim"],
        )
        self.evaluate_policy = GreedyPolicyForValueDistribution(
            self.policy_network,
            params["atoms_num"],
            params["v_min"],
            params["v_max"],
            params["action_space_dim"],
        )


class CategoricalDQNActor(Actor):
    """C51 Actor"""

    def __init__(self, params):
        super().__init__()
        self.init_policy = params["init_policy"]
        self.collect_policy = params["collect_policy"]
        self.evaluate_policy = params["evaluate_policy"]
        self._environment = params["collect_environment"]
        self._eval_env = params["eval_environment"]
        self.replay_buffer = params["replay_buffer"]
        self.step = Parameter(Tensor(0, ms.int32), name="step", requires_grad=False)
        self.expand_dims = P.ExpandDims()
        self.reshape = P.Reshape()
        self.ones = P.Ones()
        self.abs = P.Abs()
        self.assign = P.Assign()
        self.select = P.Select()
        self.reward = Tensor(
            [
                1,
            ],
            ms.float32,
        )
        self.penalty = Tensor(
            [
                -1,
            ],
            ms.float32,
        )
        self.print = P.Print()

    def act(self, phase, params):
        """get next information"""
        # param is state
        if phase == 1:
            # Fill the replay buffer
            action = self.init_policy()
            new_state, reward, done = self._environment.step(action)
            done = self.expand_dims(done, 0)
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
            done = self.expand_dims(done, 0)
            action = self.reshape(action, (1,))
            my_reward = self.select(done, self.penalty, self.reward)

            return done, reward, new_state, action, my_reward
        if phase == 3:
            # Evaluate the trained policy
            ts0 = self.expand_dims(params, 0)
            action = self.evaluate_policy(ts0)
            new_state, reward, done = self._eval_env.step(action)
            done = self.expand_dims(done, 0)
            return done, reward, new_state
        self.print("Phase is incorrect")
        return 0

    def get_action(self, phase, params):
        """Default get_action function"""
        return 0


class CategoricalDQNLearner(Learner):
    """C51 Learner"""

    class PolicyNetWithLossCell(nn.Cell):
        """C51 policy network with loss cell"""

        def __init__(self, backbone):
            super(CategoricalDQNLearner.PolicyNetWithLossCell, self).__init__(
                auto_prefix=False
            )
            self._backbone = backbone
            self.gather_nd = ops.GatherNd()
            self.concat = ops.Concat(-1)
            self.cast = ops.Cast()
            self.shape = ops.Shape()
            self.log = ops.Log()
            self.expand_dims = P.ExpandDims()
            self.softmax = nn.Softmax()

        def construct(self, x, action, label):
            """constructor for Loss Cell"""

            # Obtain the current Q-value logits for the selected actions.
            dist = self._backbone(x)
            dist = self.softmax(dist)
            batch_size = self.shape(action)[0]
            indices = mnp.arange(0, batch_size, 1)
            batch_indices = self.expand_dims(indices, 1).reshape(batch_size, 1)
            reshaped_actions = self.concat((batch_indices, action))
            chosen_action_dist = self.gather_nd(dist, reshaped_actions)
            chosen_action_dist = ops.clip_by_value(
                chosen_action_dist, 0.0000001, 0.99999999
            )
            loss = self.cross_entropy(label, chosen_action_dist)
            return loss

        def cross_entropy(self, proj_dist, dist):
            """cross entropy"""

            return (-proj_dist * self.log(dist)).sum(1).mean()

    def __init__(self, params=None):
        super().__init__()
        self.policy_network = params["policy_network"]
        self.target_network = params["target_network"]
        self.policy_param = ParameterTuple(self.policy_network.get_parameters())
        self.target_param = ParameterTuple(self.target_network.get_parameters())

        optimizer = nn.Adam(
            self.policy_network.trainable_params(), learning_rate=params["lr"]
        )
        loss_q_net = self.PolicyNetWithLossCell(self.policy_network)
        self.policy_network_train = nn.TrainOneStepCell(loss_q_net, optimizer)
        self.policy_network_train.set_train(mode=True)

        self.atoms_num = Tensor(params["atoms_num"], ms.int64)
        self.gamma = Tensor(params["gamma"], ms.float32)
        self.v_min = Tensor(params["v_min"], ms.float32)
        self.v_max = Tensor(params["v_max"], ms.float32)

        self.atoms_num_ = params["atoms_num"]

        self.expand_dims = P.ExpandDims()
        self.reshape = P.Reshape()
        self.hyper_map = C.HyperMap()
        self.ones_like = P.OnesLike()
        self.select = P.Select()

        self.mul = ops.Mul()
        self.zeroslike = ops.ZerosLike()
        self.shape = ops.Shape()
        self.cast = ops.Cast()
        self.get_max_index = ops.ArgMaxWithValue(1)
        self.onehot = ops.OneHot()
        self.gather_nd = ops.GatherNd()
        self.concat = ops.Concat(-1)
        self.get_range = ops.Range()
        self.tile = ops.Tile()
        self.softmax = nn.Softmax()
        self.target_support = mnp.linspace(
            params["v_min"], params["v_max"], params["atoms_num"]
        )
        self.discount_op = DiscountedReturn(gamma=params["gamma"])
        self.updater = SoftUpdate(0.95, 10, self.policy_param, self.target_param)

    def next_distribution(self, next_observation, batch_size):
        """get the distribution of next step"""

        next_target_probabilities = self.target_network(next_observation)
        next_target_probabilities = self.softmax(next_target_probabilities)
        next_target_q_values = (self.target_support * next_target_probabilities).sum(-1)
        next_action = self.get_max_index(next_target_q_values)[0]
        next_qt_argmax = self.expand_dims(next_action, 1)
        next_qt_argmax = self.cast(next_qt_argmax, ms.int32)
        batch_indices = self.get_range(
            Tensor(0, ms.int32), Tensor(batch_size, ms.int32), Tensor(1, ms.int32)
        )
        batch_indices = self.expand_dims(batch_indices, 1).reshape(batch_size, 1)
        next_qt_index = self.concat((batch_indices, next_qt_argmax))
        return self.gather_nd(next_target_probabilities, next_qt_index)

    def projection_distribution(self, next_observation, reward, done):
        """get the discretized distribution"""

        batch_size = self.shape(next_observation)[0]
        num_dims = self.shape(self.target_support)[0]
        target_support_deltas = self.target_support[1:] - self.target_support[:-1]
        delta_z = target_support_deltas[0]
        weights = self.next_distribution(next_observation, batch_size)

        support = self.tile(self.target_support, (batch_size,))
        support = support.reshape(batch_size, num_dims)
        if reward.shape[1] > 1:
            reward_td = self.discount_op(
                reward.T, done.T, Tensor([0.0], dtype=ms.float32)
            )[0]
            reward_td = reward_td.expand_dims(1)
            final_gamma = ops.prod(self.gamma * (1 - done), 1).expand_dims(1)
            supports = reward_td + support * final_gamma
        else:
            supports = reward + support * self.gamma

        clipped_support = self.expand_dims(
            ops.clip_by_value(supports, self.v_min, self.v_max), 1
        )
        tiled_support = self.tile(clipped_support, (1, 1, num_dims, 1))
        reshaped_target_support = self.tile(
            self.expand_dims(self.target_support, 1), (batch_size, 1)
        )
        reshaped_target_support = reshaped_target_support.reshape(
            batch_size, num_dims, 1
        )
        numerator = (tiled_support - reshaped_target_support).abs()
        quotient = 1 - (numerator / delta_z)
        clipped_quotient = ops.clip_by_value(quotient, 0, 1)
        weights = self.expand_dims(weights, 1)
        inner_prod = clipped_quotient * weights
        projection = inner_prod.sum(3)
        projection = projection.reshape(batch_size, num_dims)
        return projection

    def learn(self, experience):
        """Update the c51"""

        observation, action, reward, next_observation, done = experience
        proj_dist = self.projection_distribution(next_observation, reward, done)
        success = self.policy_network_train(observation, action, proj_dist)
        return success

    def update(self):
        """Update the network parameters"""

        return self.updater()
