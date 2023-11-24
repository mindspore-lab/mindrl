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
"""PPO"""
import mindspore
import mindspore.nn.probability.distribution as msd
import numpy as np
from mindspore import Tensor, nn, ops
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.ops import composite as C
from mindspore.ops import operations as P

from mindspore_rl.agent.actor import Actor
from mindspore_rl.agent.learner import Learner


class PPOPolicy:
    """This is PPOPolicy class. You should define your networks (PPOActorNet and PPOCriticNet here)
    which you prepare to use in the algorithm. Moreover, you should also define you loss function
    (PPOLossCell here) which calculates the loss between policy and your ground truth value.
    """

    class PPOActorNet(nn.Cell):
        """PPOActorNet is the actor network of PPO algorithm. It takes a set of state as input
        and outputs miu, sigma of a normal distribution"""

        def __init__(
            self,
            input_size,
            hidden_size1,
            hidden_size2,
            output_size,
            sigma_init_std,
            compute_type=mindspore.float32,
        ):
            super(PPOPolicy.PPOActorNet, self).__init__()
            self.compute_type = compute_type
            self.linear1_actor = nn.Dense(
                input_size, hidden_size1, weight_init="XavierUniform"
            ).to_float(compute_type)
            self.linear2_actor = nn.Dense(
                hidden_size1, hidden_size2, weight_init="XavierUniform"
            ).to_float(compute_type)

            self.linear_miu_actor = nn.Dense(hidden_size2, output_size).to_float(
                compute_type
            )
            sigma_init_value = np.log(np.exp(sigma_init_std) - 1)
            self.bias_sigma_actor = Parameter(
                initializer(sigma_init_value, [output_size]),
                name="bias_sigma_actor",
                requires_grad=True,
            )
            self.tanh_actor = nn.Tanh()

            self.zeros_like = P.ZerosLike()
            self.bias_add = P.BiasAdd()
            self.reshape = P.Reshape()
            self.softplus_actor = ops.Softplus()
            self.cast = ops.Cast()

        def construct(self, x):
            """calculate miu and sigma"""
            x = self.tanh_actor(self.linear1_actor(x))
            x = self.tanh_actor(self.linear2_actor(x))
            miu = self.tanh_actor(self.linear_miu_actor(x))
            miu_shape = miu.shape
            miu = self.reshape(miu, (-1, 6))

            bias_sigma_actor = self.cast(self.bias_sigma_actor, self.compute_type)
            sigma = self.softplus_actor(
                self.bias_add(self.zeros_like(miu), bias_sigma_actor)
            )
            miu = self.reshape(miu, miu_shape)
            sigma = self.reshape(sigma, miu_shape)
            # Cast experiences to FP32 since replay buffer do not support fp16.
            # Graph optimize will remove redundant cast operators.
            miu = self.cast(miu, mindspore.float32)
            sigma = self.cast(sigma, mindspore.float32)
            return miu, sigma

    class PPOCriticNet(nn.Cell):
        """PPOCriticNet is the critic network of PPO algorithm. It takes a set of states as input
        and outputs the value of input state"""

        def __init__(
            self,
            input_size,
            hidden_size1,
            hidden_size2,
            output_size,
            compute_type=mindspore.float32,
        ):
            super(PPOPolicy.PPOCriticNet, self).__init__()
            self.linear1_critic = nn.Dense(
                input_size, hidden_size1, weight_init="XavierUniform"
            ).to_float(compute_type)
            self.linear2_critic = nn.Dense(
                hidden_size1, hidden_size2, weight_init="XavierUniform"
            ).to_float(compute_type)
            self.linear3_critic = nn.Dense(hidden_size2, output_size).to_float(
                compute_type
            )
            self.tanh_critic = nn.Tanh()
            self.cast = ops.Cast()

        def construct(self, x):
            """predict value"""
            x = self.tanh_critic(self.linear1_critic(x))
            x = self.tanh_critic(self.linear2_critic(x))
            x = self.linear3_critic(x)
            # Cast experiences to FP32 since replay buffer do not support fp16.
            # Graph optimize will remove redundant cast operators.
            x = self.cast(x, mindspore.float32)
            return x

    class CollectPolicy(nn.Cell):
        """Collect Policy"""

        def __init__(self, actor_net):
            super(PPOPolicy.CollectPolicy, self).__init__()
            self.actor_net = actor_net
            self.norm_dist = msd.Normal()

        def construct(self, params):
            miu, sigma = self.actor_net(params)
            action = self.norm_dist.sample((), miu, sigma)
            return action, miu, sigma

    class EvalPolicy(nn.Cell):
        """Eval Policy"""

        def __init__(self, actor_net):
            super(PPOPolicy.EvalPolicy, self).__init__()
            self.actor_net = actor_net

        def construct(self, params):
            miu, _ = self.actor_net(params)
            return miu

    def __init__(self, params):
        self.actor_net = self.PPOActorNet(
            params["state_space_dim"],
            params["hidden_size1"],
            params["hidden_size2"],
            params["action_space_dim"],
            params["sigma_init_std"],
            params["compute_type"],
        )
        self.critic_net = self.PPOCriticNet(
            params["state_space_dim"],
            params["hidden_size1"],
            params["hidden_size2"],
            1,
            params["compute_type"],
        )

        self.collect_policy = self.CollectPolicy(self.actor_net)
        self.eval_policy = self.EvalPolicy(self.actor_net)


# pylint: disable=W0223
class PPOActor(Actor):
    """This is an actor class of PPO algorithm, which is used to interact with environment, and
    generate/insert experience (data)"""

    def __init__(self, params=None):
        super().__init__()
        self._params_config = params
        self._environment = params["collect_environment"]
        self._eval_env = params["eval_environment"]
        self._buffer = params["replay_buffer"]
        self.collect_policy = params["collect_policy"]
        self.eval_policy = params["eval_policy"]
        self.norm_dist = msd.Normal()
        self.print = P.Print()
        self.expand_dims = P.ExpandDims()

    def act(self, phase, params):
        """collect experience and insert to replay buffer (used during training)"""
        if phase == 2:
            action, miu, sigma = self.collect_policy(params)
            new_state, reward, _ = self._environment.step(action)
            reward = self.expand_dims(reward, -1)
            return reward, new_state, action, miu, sigma
        if phase == 3:
            action = self.eval_policy(params)
            new_state, reward, _ = self._eval_env.step(action)
            reward = self.expand_dims(reward, 0)
            return reward, new_state
        self.print("Phase is incorrect")
        return 0

    # pylint: disable=W0221
    def get_action(self, params):
        """Default get_action function"""
        action, miu, sigma = self.collect_policy(params)
        return action, miu, sigma


class PPOLearner(Learner):
    """This is the learner class of PPO algorithm, which is used to update the policy net"""

    class PPOLossCell(nn.Cell):
        """PPOLossCell calculates the loss of PPO algorithm"""

        def __init__(self, actor_net, critic_net, epsilon, critic_coef):
            super(PPOLearner.PPOLossCell, self).__init__(auto_prefix=False)
            self.actor_net = actor_net
            self.critic_net = critic_net
            self.epsilon = epsilon
            self.critic_coef = critic_coef

            self.reduce_mean = P.ReduceMean()
            self.reduce_sum = P.ReduceSum()
            self.div = P.Div()
            self.mul = P.Mul()
            self.minimum = P.Minimum()
            self.add = P.Add()
            self.sub = P.Sub()
            self.square = P.Square()
            self.exp = P.Exp()
            self.squeeze = P.Squeeze()
            self.norm_dist_new = msd.Normal()

        def construct(self, actions, states, advantage, log_prob_old, discounted_r):
            """calculate the total loss"""
            # Actor Loss
            miu_new, sigma_new = self.actor_net(states)
            log_prob_new = self.reduce_sum(
                self.norm_dist_new.log_prob(actions, miu_new, sigma_new), -1
            )
            importance_ratio = self.exp(log_prob_new - log_prob_old)
            surr = self.mul(importance_ratio, advantage)
            clip_surr = self.mul(
                C.clip_by_value(
                    importance_ratio, 1.0 - self.epsilon, 1.0 + self.epsilon
                ),
                advantage,
            )
            actor_loss = self.reduce_mean(-self.minimum(surr, clip_surr))

            # Critic Loss
            value_prediction = self.critic_net(states)
            value_prediction = self.squeeze(value_prediction)
            squared_advantage_critic = self.square(discounted_r - value_prediction)
            critic_loss = self.reduce_mean(squared_advantage_critic) * self.critic_coef

            # Total Loss
            total_loss = actor_loss + critic_loss
            return total_loss

    def __init__(self, params):
        super().__init__()
        self._params_config = params
        self.gamma = Tensor(self._params_config["gamma"], mindspore.float32)
        self.iter_times = params["iter_times"]
        self.actor_net = params["actor_net"]
        self.critic_net = params["critic_net"]

        trainable_parameter = (
            self.critic_net.trainable_params() + self.actor_net.trainable_params()
        )
        optimizer_ppo = nn.Adam(trainable_parameter, learning_rate=params["lr"], eps=1e-5)
        ppo_loss_net = self.PPOLossCell(
            self.actor_net,
            self.critic_net,
            Tensor(params["epsilon"], mindspore.float32),
            Tensor(params["critic_coef"], mindspore.float32),
        )
        self._ppo_net_train = nn.TrainOneStepCell(ppo_loss_net, optimizer_ppo)
        self._ppo_net_train.set_train(mode=True)

        self.sub = P.Sub()
        self.mul = P.Mul()
        self.add = P.Add()
        self.zeros_like = P.ZerosLike()
        self.reshape = P.Reshape()
        self.expand_dims = P.ExpandDims()
        self.stack = P.Stack()
        self.assign = P.Assign()
        self.squeeze = P.Squeeze()
        self.moments = nn.Moments(axis=(0, 1), keep_dims=True)
        self.sqrt = P.Sqrt()
        self.norm_dist_old = msd.Normal()
        self.reduce_sum = P.ReduceSum()
        self.zero = Tensor(0, mindspore.float32)
        self.zero_int = Tensor(0, mindspore.int32)

    def learn(self, experience):
        """prepare for the value (advantage, discounted reward), which is used to calculate
        the loss"""
        (
            state_list,
            action_list,
            reward_list,
            next_state_list,
            miu_list,
            sigma_list,
        ) = experience
        last_state = next_state_list[:, -1]
        rewards = self.squeeze(reward_list)

        last_value_prediction = self.critic_net(last_state)
        last_value_prediction = self.squeeze(last_value_prediction)
        last_value_prediction = self.zeros_like(last_value_prediction)

        def discounted_reward(rewards, v_last, gamma):
            """Compute discounter reward"""
            discounted_r = self.zeros_like(rewards)
            iter_num = self.zero_int
            iter_end = len(rewards[0])
            while iter_num < iter_end:
                i = iter_end - iter_num - 1
                v_last = self.add(rewards[:, i], self.mul(gamma, v_last))
                discounted_r[:, i] = self.reshape(v_last, (-1,))
                iter_num += 1
            return discounted_r

        discounted_r = discounted_reward(rewards, last_value_prediction, self.gamma)

        def gae(
            reward_list, next_state_list, critic_value, v_last, gamma, td_lambda=0.95
        ):
            """Compute advantage"""
            next_critic_value = self.critic_net(next_state_list)
            delta = self.squeeze(reward_list + gamma * next_critic_value - critic_value)
            weighted_discount = gamma * td_lambda
            advantage = self.zeros_like(delta)
            v_last = self.zeros_like(v_last)
            iter_num = self.zero_int
            iter_end = len(delta[0])
            while iter_num < iter_end:
                i = iter_end - iter_num - 1
                v_last = self.add(delta[:, i], self.mul(weighted_discount, v_last))
                advantage[:, i] = self.reshape(v_last, (-1,))
                iter_num += 1
            return advantage

        value_prediction = self.critic_net(state_list)
        advantage = gae(
            reward_list,
            next_state_list,
            value_prediction,
            last_value_prediction,
            self.gamma,
        )

        def _normalized_advantage(advantage, epsilon=1e-8):
            adv_mean, adv_variance = self.moments(advantage)
            normalized_advantage = (advantage - adv_mean) / (
                self.sqrt(adv_variance) + epsilon
            )
            return normalized_advantage

        normalized_advantage = _normalized_advantage(advantage)

        log_prob_old = self.reduce_sum(
            self.norm_dist_old.log_prob(action_list, miu_list, sigma_list), -1
        )

        i = self.zero
        loss = self.zero

        while i < self.iter_times:
            loss += self._ppo_net_train(
                action_list,
                state_list,
                normalized_advantage,
                log_prob_old,
                discounted_r,
            )
            i += 1

        return loss / self.iter_times
