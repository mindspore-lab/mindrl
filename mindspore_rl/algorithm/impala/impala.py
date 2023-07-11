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
"""Importance Weighted Actor-Learner Architectures"""

# pylint: disable=W0237
import mindspore as ms
import mindspore.nn.probability.distribution as msd
from mindspore import Tensor, nn, ops
from mindspore.common.parameter import ParameterTuple

from mindspore_rl.agent.actor import Actor
from mindspore_rl.agent.learner import Learner
from mindspore_rl.algorithm.impala.vtrace import (
    get_importance_weight,
    get_vs_and_advantages,
    log_probs_logits,
)
from mindspore_rl.utils import BatchRead, TensorArray


class IMPALANetwork:
    """IMPALA Policy and Network"""

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
        self.learner_net = self.ActorCriticNet(
            params["state_space_dim"], params["hidden_size"], params["action_space_dim"]
        )
        self.actor_net = self.ActorCriticNet(
            params["state_space_dim"], params["hidden_size"], params["action_space_dim"]
        )


# pylint: disable=W0223
class IMPALAActor(Actor):
    """Actor"""

    # pylint: disable=W0613
    def __init__(self, params=None, actor_id=None):
        super().__init__()
        # local net
        self.net = params["actor_net"]
        self.local_param = self.net.trainable_params()
        self.local_weight = ParameterTuple(self.local_param)
        self.state_space_dim = params["state_space_dim"]
        self.action_space_dim = params["action_space_dim"]

        # env
        self._environment = params["collect_environment"]
        self._eval_env = params["eval_environment"]
        self.pull = BatchRead()

        self.c_dist = msd.Categorical(dtype=ms.float32)
        self.expand_dims = ops.ExpandDims()
        self.reshape = ops.Reshape()
        self.cast = ops.Cast()
        self.concat = ops.concat
        self.softmax = ops.Softmax()
        self.depend = ops.Depend()
        self.zero = Tensor(0, ms.int64)
        self.loop_size = params["length"]
        loop_size = self.loop_size
        # self.loop_size = Tensor(loop_size, ms.int64)
        self.done = Tensor(True, ms.bool_)
        self.false = Tensor(False, ms.bool_)
        self.states = TensorArray(
            ms.float32, (self.state_space_dim,), dynamic_size=False, size=loop_size
        )
        self.actions = TensorArray(ms.int32, (1,), dynamic_size=False, size=loop_size)
        self.rewards = TensorArray(ms.float32, (1,), dynamic_size=False, size=loop_size)
        self.policys = TensorArray(
            ms.float32, (self.action_space_dim,), dynamic_size=False, size=loop_size
        )
        self.batch_size = params["batch_size"]
        self.mask_done = Tensor([0], ms.bool_)
        self.print = ops.Print
        self.squeeze = ops.Squeeze()
        self.transpose = ops.Transpose()
        self.neg_value = Tensor([-1], ms.float32)

    def act(self, phase, actor_id=0, weight_copy=None, eval_state=None):
        """interact with environment and return [(x,a,t,u(x|t))....]
        Returns:
            When phase == 2:
            A tuple (states, actions, rewards, policys)
            states : a tensor of shape [T, B, state_dim]
            actions : a tensor of shape [T, B]
            rewards : a tensor of shape [T, B]
            policys : a tensor of shape [T, B, action_dim]
            Where T is time scale, B is batch size

            When phase == 1:
            Return one step reward using current policy
        """
        if phase == 2:
            # Collect
            cur = self.zero
            trajectory_rewards = ops.zeros(
                (self.loop_size, self.batch_size), dtype=ms.float32
            )
            trajectory_states = ops.zeros(
                (self.loop_size, self.batch_size, self.state_space_dim), ms.float32
            )
            trajectory_actions = ops.zeros((self.loop_size, self.batch_size), ms.int32)
            trajectory_policys = ops.zeros(
                (self.loop_size, self.batch_size, self.action_space_dim), ms.float32
            )
            masks = ops.ones((self.loop_size, self.batch_size), ms.int32)
            update = self.pull(self.local_weight, weight_copy)
            while cur < self.batch_size:
                s = self._environment[actor_id].reset()
                t = self.zero
                while t < self.loop_size:
                    self.states.write(t, s)
                    ts0 = self.expand_dims(s, 0)
                    action_logits, _ = self.net(ts0)
                    # update local net before run
                    action_logits = self.depend(action_logits, update)
                    self.policys.write(t, self.squeeze(action_logits))
                    action_probs_t = self.softmax(action_logits)
                    action = self.reshape(
                        self.c_dist.sample((1,), probs=action_probs_t), (1,)
                    )
                    action = self.cast(action, ms.int32)
                    self.actions.write(t, action)
                    new_state, reward, done = self._environment[actor_id].step(action)
                    reward = self.expand_dims(reward, 0)
                    self.rewards.write(
                        t, reward if done != self.done else self.neg_value
                    )
                    s = new_state
                    if done == self.done:
                        masks[t][cur] = self.mask_done
                        s = self._environment[actor_id].reset()
                    t += 1
                rewards = self.rewards.stack()  # [T]
                states = self.states.stack()  # [T,num_of_states]
                actions = self.squeeze(self.actions.stack())  # [T]
                policys = self.policys.stack()  # [T,num_of_actions]

                rewards = self.squeeze(rewards)

                trajectory_rewards[:, cur] = rewards
                trajectory_states[:, cur, :] = states
                trajectory_actions[:, cur] = actions
                trajectory_policys[:, cur, :] = policys

                self.rewards.clear()
                self.states.clear()
                self.actions.clear()
                self.policys.clear()
                cur += 1

            return (
                trajectory_states,
                trajectory_actions,
                trajectory_rewards,
                trajectory_policys,
                masks,
            )

        if phase == 3:
            # Evaluate
            s = eval_state
            ts0 = self.expand_dims(s, 0)
            action_logits, _ = self.net(ts0)
            action_probs_t = self.softmax(action_logits)
            action = self.reshape(self.c_dist.sample((1,), probs=action_probs_t), (1,))
            action = self.cast(action, ms.int32)
            new_state, reward, done = self._eval_env.step(action)
            return new_state, self.squeeze(reward), done
        self.print("Phase is incorrect")
        return 0


class IMPALALearner(Learner):
    """IMPALA Learner"""

    def __init__(self, params):
        super().__init__()
        self.net = params["learner_net"]
        self.global_weight = self.net.trainable_params()
        self.global_params = ParameterTuple(self.global_weight)
        self.optimizer = nn.Adam(
            self.global_weight, learning_rate=params["lr"], weight_decay=0.99
        )
        self.loss_net = self.Loss(self.net)
        self.impala_net_train = nn.TrainOneStepCell(self.loss_net, self.optimizer)
        self.impala_net_train.set_train(mode=True)
        self.reduce_sum = ops.ReduceSum()

        self.clip_rho_thres = params["clip_rho_threshold"]
        self.clip_pg_rho_thres = params["clip_pg_rho_threshold"]
        self.clip_cs_rho_thres = params["clip_cs_threshold"]
        self.discount = params["discount"]
        self.fill = ops.Fill()
        self.squeeze = ops.Squeeze()

        self.baseline_cost = params["baseline_cost"]
        self.entropy_cost = params["entropy_cost"]

    class Loss(nn.Cell):
        """Actor-Critic loss"""

        def __init__(self, net):
            super().__init__(auto_prefix=False)
            self.net = net
            self.reduce_sum = ops.ReduceSum()
            self.square = ops.Square()
            self.stop_gradient = ops.StopGradient()
            self.squeeze = ops.Squeeze()
            self.softmax = ops.Softmax()
            self.log_softmax = ops.log_softmax

        def construct(
            self, states, actions, pg_advantages, vs, baseline_cost, entropy_cost
        ):
            """Calculate policy_gradient_loss, baseline_loss, entropy_loss"""
            logits, values = self.net(states)
            values = self.squeeze(values)

            logp = -log_probs_logits(logits, actions)
            pg_advantages = self.stop_gradient(pg_advantages)
            policy_gradient_loss = self.reduce_sum(logp * pg_advantages)

            vs = self.stop_gradient(vs)
            baseline_loss = (
                baseline_cost * 0.5 * self.reduce_sum(self.square(values - vs))
            )

            policy = self.softmax(logits)
            log_policy = self.log_softmax(logits)
            entropy = self.reduce_sum(-policy * log_policy, -1)
            entropy_loss = -self.reduce_sum(entropy_cost * entropy)

            return policy_gradient_loss + baseline_loss + entropy_loss

    def learn(self, trajectory):
        states, actions, rewards, policys, masks = trajectory
        actions = actions.astype(ms.int32)
        masks = masks.astype(ms.int32)
        target_policy, values = self.net(states)
        values = self.squeeze(values)

        log_rho = get_importance_weight(
            behavior_policy_logits=policys,
            target_policy_logits=target_policy,
            actions=actions,
        )
        discounts = self.fill(
            ms.float32, (states.shape[0], states.shape[1]), self.discount
        )
        bootstrap_value = values[-1]
        vs, pg_advantage = get_vs_and_advantages(
            log_rho,
            discounts,
            rewards,
            values,
            bootstrap_value,
            self.clip_rho_thres,
            self.clip_pg_rho_thres,
            self.clip_cs_rho_thres,
            masks,
        )
        loss = self.impala_net_train(
            states, actions, pg_advantage, vs, self.baseline_cost, self.entropy_cost
        )
        return loss
