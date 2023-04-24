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
"""Policy Gradient"""

import mindspore
import mindspore.nn.probability.distribution as msd
from mindspore import Tensor, nn
from mindspore.ops import operations as P

from mindspore_rl.agent.actor import Actor
from mindspore_rl.agent.learner import Learner
from mindspore_rl.utils import DiscountedReturn


class PGPolicyAndNetwork:
    """PGPolicyAndNetwork"""

    class ActorNet(nn.Cell):
        """ActorNet"""

        def __init__(
            self, input_size, hidden_size, output_size, compute_type=mindspore.float32
        ):
            super().__init__()
            self.dense1 = nn.Dense(
                input_size, hidden_size, weight_init="XavierUniform"
            ).to_float(compute_type)
            self.dense2 = nn.Dense(
                hidden_size, output_size, weight_init="XavierUniform"
            ).to_float(compute_type)
            self.active = P.Tanh()
            self.softmax = P.Softmax()
            self.cast = P.Cast()

        def construct(self, x):
            x = self.dense1(x)
            x = self.active(x)
            x = self.dense2(x)
            return self.cast(self.softmax(x), mindspore.float32)

    class CollectPolicy(nn.Cell):
        """Collect Policy"""

        def __init__(self, actor_net):
            super(PGPolicyAndNetwork.CollectPolicy, self).__init__()
            self.actor_net = actor_net
            self.reshape = P.Reshape()
            self.c_dist = msd.Categorical(dtype=mindspore.float32)

        def construct(self, params):
            action_probs_t = self.actor_net(params)
            action = self.reshape(self.c_dist.sample((1,), probs=action_probs_t), (1,))
            return action

    class EvalPolicy(nn.Cell):
        """Eval Policy"""

        def __init__(self, actor_net):
            super(PGPolicyAndNetwork.EvalPolicy, self).__init__()
            self.actor_net = actor_net
            self.reshape = P.Reshape()
            self.argmax = P.Argmax(output_type=mindspore.int32)

        def construct(self, params):
            action_probs_t = self.actor_net(params)
            action = self.reshape(self.argmax(action_probs_t), (1,))
            return action

    def __init__(self, params):
        self.actor_net = self.ActorNet(
            params.get("state_space_dim"),
            params.get("hidden_size"),
            params.get("action_space_dim"),
            params.get("compute_type"),
        )
        self.collect_policy = self.CollectPolicy(self.actor_net)
        self.eval_policy = self.EvalPolicy(self.actor_net)


# pylint: disable=W0223
class PGActor(Actor):
    """PG Actor"""

    def __init__(self, params=None):
        # pylint: disable=R1725
        super(PGActor, self).__init__()
        self._params_config = params
        self._environment = params.get("collect_environment")
        self._eval_env = params.get("eval_environment")
        self.collect_policy = params.get("collect_policy")
        self.eval_policy = params.get("eval_policy")
        self.expand_dims = P.ExpandDims()
        self.cast = P.Cast()
        self.print = P.Print()

    def act(self, phase, params):
        if phase == 2:
            # Sample action to act in env
            ts0 = self.expand_dims(params, 0)
            action = self.collect_policy(ts0)
            action = self.cast(action, mindspore.int32)
            new_state, reward, done = self._environment.step(action)
            reward = self.expand_dims(reward, 0)
            done = self.expand_dims(done, 0)
            return done, reward, new_state, action
        if phase == 3:
            # Evaluate the trained policy
            ts0 = self.expand_dims(params, 0)
            action = self.eval_policy(ts0)
            new_state, reward, done = self._eval_env.step(
                self.cast(action, mindspore.int32)
            )
            reward = self.expand_dims(reward, 0)
            done = self.expand_dims(done, 0)
            return done, reward, new_state

        self.print("Phase is incorrect")
        return 0


class PGLearner(Learner):
    """PG Learner"""

    class ActorNNLoss(nn.Cell):
        """Actor loss"""

        def __init__(self, actor_net, depth):
            super().__init__(auto_prefix=False)
            self.actor_net = actor_net
            self.reduce_mean = P.ReduceMean()
            self.reduce_sum = P.ReduceSum()
            self.onehot = P.OneHot()
            self.depth = depth
            self.on_value = Tensor(1.0, mindspore.float32)
            self.off_value = Tensor(0.0, mindspore.float32)
            self.log = P.Log()
            self.cast = P.Cast()

        def construct(self, state, action, reward):
            onehot_action = self.onehot(
                action, self.depth, self.on_value, self.off_value
            ).reshape((-1, self.depth))
            act_prob = self.actor_net(state)
            log_prob = self.reduce_sum(-1.0 * self.log(act_prob) * onehot_action, 1)
            loss = self.reduce_mean(log_prob * reward.reshape((-1,)))
            return loss

    def __init__(self, params):
        # pylint: disable=R1725
        super(PGLearner, self).__init__()
        self._params_config = params
        self.actor_net = params["actor_net"]
        self.action_dim = params["action_space_dim"]
        self.zero_float = Tensor([0.0], mindspore.float32)
        optimizer = nn.Adam(
            self.actor_net.trainable_params(), learning_rate=params["lr"]
        )
        actor_loss_net = self.ActorNNLoss(self.actor_net, self.action_dim)
        self.actor_net_train = nn.TrainOneStepCell(actor_loss_net, optimizer)
        self.actor_net_train.set_train(mode=True)
        self.discount_return = DiscountedReturn(gamma=params["gamma"])

    def learn(self, experience):
        """Calculate the td_error"""
        state = experience[0]
        reward = experience[1]
        action = experience[2]
        mask = experience[3]
        returns = self.discount_return(reward, mask, self.zero_float)
        loss = self.actor_net_train(state, action, returns)
        return loss
