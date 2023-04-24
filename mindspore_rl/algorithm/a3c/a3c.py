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
"""Async Advantage Actor Critic"""

# pylint: disable=W0237
import mindspore
import mindspore.nn.probability.distribution as msd
import numpy as np
from mindspore import Tensor, nn, ops
from mindspore.common.parameter import ParameterTuple
from mindspore.ops import composite as C
from mindspore.ops import operations as P

from mindspore_rl.agent.actor import Actor
from mindspore_rl.agent.learner import Learner
from mindspore_rl.utils import BatchRead, DiscountedReturn, TensorArray

SEED = 16
np.random.seed(SEED)


class A3CPolicyAndNetwork:
    """A3CPolicyAndNetwork"""

    class ActorCriticNet(nn.Cell):
        """ActorCriticNet"""

        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.common = nn.Dense(input_size, hidden_size, weight_init="XavierUniform")
            self.actor = nn.Dense(hidden_size, output_size, weight_init="XavierUniform")
            self.critic = nn.Dense(hidden_size, 1, weight_init="XavierUniform")
            self.relu = nn.LeakyReLU()

        def construct(self, x):
            x = self.common(x)
            x = self.relu(x)
            return self.actor(x), self.critic(x)

    def __init__(self, params):
        self.a3c_net = self.ActorCriticNet(
            params["state_space_dim"], params["hidden_size"], params["action_space_dim"]
        )
        self.a3c_net_learn = self.ActorCriticNet(
            params["state_space_dim"], params["hidden_size"], params["action_space_dim"]
        )
        self.a3c_net_copy = ParameterTuple(self.a3c_net_learn.trainable_params())


# pylint: disable=W0223
class A3CActor(Actor):
    """A3C Actor"""

    class Loss(nn.Cell):
        """Actor-Critic loss"""

        def __init__(self, a2c_net):
            super().__init__(auto_prefix=False)
            self.a2c_net = a2c_net
            self.reduce_sum = ops.ReduceSum(keep_dims=False)
            self.log = ops.Log()
            self.gather = ops.GatherD()
            self.softmax = ops.Softmax()
            self.smoothl1_loss = nn.SmoothL1Loss(beta=1.0, reduction="sum")

        def construct(self, states, actions, returns):
            """Calculate actor loss and critic loss"""
            action_logits_ts, values = self.a2c_net(states)
            action_probs_t = self.softmax(action_logits_ts)
            action_probs = self.gather(action_probs_t, 1, actions)
            advantage = returns - values
            action_log_probs = self.log(action_probs)
            adv_mul_prob = action_log_probs * advantage
            actor_loss = -self.reduce_sum(adv_mul_prob)
            critic_loss = self.smoothl1_loss(values, returns)
            return critic_loss + actor_loss

    # pylint: disable=W0613
    def __init__(self, params=None, actor_id=None):
        super().__init__()
        self._params_config = params
        self.a3c_net = params["a3c_net"]
        self.local_param = self.a3c_net.trainable_params()
        self._environment = params["collect_environment"]
        self.c_dist = msd.Categorical(dtype=mindspore.float32, seed=SEED)
        self.expand_dims = P.ExpandDims()
        self.reshape = P.Reshape()
        self.cast = P.Cast()
        self.softmax = ops.Softmax()
        # (dst, src) overwrite dst by src
        self.pull = BatchRead()
        self.depend = ops.Depend()
        self.sens = 1.0
        self.grad = C.GradOperation(get_by_list=True, sens_param=False)
        self.local_weight = ParameterTuple(self.local_param)
        self.zero = Tensor(0, mindspore.int64)
        loop_size = 200
        self.loop_size = Tensor(loop_size, mindspore.int64)
        self.done = Tensor(True, mindspore.bool_)
        self.states = TensorArray(
            mindspore.float32, (4,), dynamic_size=False, size=loop_size
        )
        self.actions = TensorArray(
            mindspore.int32, (1,), dynamic_size=False, size=loop_size
        )
        self.rewards = TensorArray(
            mindspore.float32, (1,), dynamic_size=False, size=loop_size
        )
        self.masks = Tensor(np.zeros([loop_size, 1], dtype=np.bool_), mindspore.bool_)
        self.mask_done = Tensor([1], mindspore.bool_)
        self.shape = ops.DynamicShape()
        self.moments = nn.Moments(keep_dims=False)
        self.sqrt = ops.Sqrt()
        self.zero = Tensor(0, mindspore.int64)
        self.epsilon = Tensor(1.1920929e-07, mindspore.float32)
        self.zero_float = Tensor([0.0], mindspore.float32)
        self.discount_return = DiscountedReturn(gamma=self._params_config["gamma"])
        self.print = P.Print()
        self.loss_net = self.Loss(self.a3c_net)

    # pylint: disable=W0221
    def act(self, phase, actor_id=0, weight_copy=None):
        """Store returns into TensorArrays from env"""
        if phase == 2:
            s = self._environment[actor_id].reset()
            update = self.pull(self.local_weight, weight_copy)
            t = self.zero
            done_status = self.zero
            done_num = self.zero
            masks = self.masks
            while t < self.loop_size:
                self.states.write(t, s)
                ts0 = self.expand_dims(s, 0)
                action_logits, _ = self.a3c_net(ts0)
                # update local net before run
                action_logits = self.depend(action_logits, update)
                action_probs_t = self.softmax(action_logits)
                action = self.reshape(
                    self.c_dist.sample((1,), probs=action_probs_t), (1,)
                )
                action = self.cast(action, mindspore.int32)
                self.actions.write(t, action)
                new_state, reward, done = self._environment[actor_id].step(action)
                reward = self.expand_dims(reward, 0)
                done = self.expand_dims(done, 0)
                self.rewards.write(t, reward)
                s = new_state
                if done == self.done:
                    if done_status == self.zero:
                        done_status += 1
                        done_num = t
                    masks[t] = self.mask_done
                    self._environment[actor_id].reset()
                t += 1
            rewards = self.rewards.stack()
            states = self.states.stack()
            actions = self.actions.stack()
            self.rewards.clear()
            self.states.clear()
            self.actions.clear()
            # compute local loss and grads
            returns = self.discount_return(rewards, masks, self.zero_float)
            adv_mean, adv_var = self.moments(returns)
            normalized_returns = (returns - adv_mean) / (
                self.sqrt(adv_var) + self.epsilon
            )
            loss = self.loss_net(states, actions, normalized_returns)
            grads = self.grad(self.loss_net, self.local_weight)(
                *(states, actions, normalized_returns)
            )
            return done_num, grads, loss
        self.print("Phase is incorrect")
        return 0


class A3CLearner(Learner):
    """A3C Learner"""

    def __init__(self, params):
        super().__init__()
        self.a3c_net_learn = params["a3c_net_learn"]
        self.weight_copy = params["a3c_net_copy"]
        self.global_weight = self.a3c_net_learn.trainable_params()
        self.global_params = ParameterTuple(self.global_weight)
        self.optimizer = nn.Adam(self.global_weight, learning_rate=params["lr"])

    # pylint: disable=W0221
    def learn(self, grads):
        """update"""
        success = self.optimizer(grads)
        return success
