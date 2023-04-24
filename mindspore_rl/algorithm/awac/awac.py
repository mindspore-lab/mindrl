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
"""Accelerating Online Reinforcement Learning with Offline Datasets"""
# pylint: disable=W0237
import mindspore
import mindspore.nn.probability.distribution as msd
import numpy as np
from mindspore import Tensor, nn, ops

from mindspore_rl.agent.actor import Actor
from mindspore_rl.agent.learner import Learner
from mindspore_rl.algorithm.sac import SACPolicy
from mindspore_rl.utils import SoftUpdate

SEED = 10
np.random.seed(SEED)


class AWACPolicyAndNetwork(SACPolicy):
    """AWACPolicyAndNetwork: using the implement of SAC"""

    def __init__(self, params):
        super().__init__(params)
        self.model_1 = self.critic_net1
        self.model_2 = self.critic_net2
        self.target_model_1 = self.target_critic_net1
        self.target_model_2 = self.target_critic_net2
        self.policy = self.actor_net
        self.eval_policy = self.eval_policy


# pylint: disable=W0223
class AWACActor(Actor):
    """AWAC Actor"""

    def __init__(self, params=None):
        super().__init__()
        self.policy = params["actor_net"]
        self.eval_policy = params["eval_policy"]
        self._environment = params["collect_environment"]
        self.dist = msd.Normal()
        self.cast = ops.Cast()
        self.sigmoid = ops.Sigmoid()
        self.max_ = Tensor(0.0, mindspore.float32)
        self.min_ = Tensor(-6.0, mindspore.float32)

    # pylint: disable=W0221
    def act(self, phase, state):
        """act with environment"""
        _, _, logstd, mean_tanh = self.policy(state)
        logstd = self.sigmoid(logstd)
        logstd = self.min_ + logstd * (self.max_ - self.min_)
        std = logstd.exp()
        esp = ops.standard_normal(mean_tanh.shape)
        x_t = mean_tanh + esp * std
        actions = x_t.clip(min=-1, max=1)

        next_obs, rewards, done = self._environment.step(actions)
        done = self.cast(done, mindspore.float32).expand_dims(0)
        rewards = rewards.expand_dims(0)
        done = done.expand_dims(0)
        next_obs = next_obs.expand_dims(0)
        return actions, rewards, next_obs, done

    # pylint: disable=W0221
    def get_action(self, obs):
        """return action"""
        action = self.eval_policy(obs)
        return action


class AWACLearner(Learner):
    """AWAC Learner"""

    class CriticLoss(nn.Cell):
        """Critic loss"""

        def __init__(
            self, model_1, model_2, target_model_1, target_model_2, policy, gamma
        ):
            super(AWACLearner.CriticLoss, self).__init__(auto_prefix=False)
            self.model_1 = model_1
            self.model_2 = model_2
            self.target_model_1 = target_model_1
            self.target_model_2 = target_model_2
            self.policy = policy
            self.gamma = gamma
            self.dist = msd.Normal()
            self.cast = ops.Cast()
            self.one = Tensor(1.0, mindspore.float32)
            self.mse_loss = nn.MSELoss(reduction="none")
            self.sigmoid = ops.Sigmoid()
            self.max_ = Tensor(0.0, mindspore.float32)
            self.min_ = Tensor(-6.0, mindspore.float32)
            self.alpha = Tensor([1e-20], mindspore.float32)

        def sample(self, obs):
            """sample an action"""
            _, _, logstd, act_mean = self.policy(obs)
            logstd = self.cast(logstd, mindspore.float32)
            act_mean = self.cast(act_mean, mindspore.float32)
            esp = ops.standard_normal(act_mean.shape)
            logstd = self.sigmoid(logstd)
            logstd = self.min_ + logstd * (self.max_ - self.min_)
            std = logstd.exp()
            x_t = act_mean + esp * std
            act = x_t.clip(min=-1.0, max=1.0)
            log_pi_row = self.dist.log_prob(act, act_mean, std)
            log_pi = log_pi_row.sum(-1, keepdims=True)
            return act, log_pi

        def compute_target(self, next_obs, reward, terminal):
            """comput target"""
            next_action, next_log_prob = self.sample(next_obs)
            q1_next = self.target_model_1(next_obs, next_action)
            q2_next = self.target_model_2(next_obs, next_action)
            target_q = ops.minimum(q1_next, q2_next)
            target = reward + self.gamma * (self.one - terminal) * (
                target_q - self.alpha * next_log_prob
            )
            return target

        def construct(self, obs, action, target):
            """Calculate critic loss"""
            cur_q1 = self.model_1(obs, action)
            cur_q2 = self.model_2(obs, action)
            qf1_loss = self.mse_loss(cur_q1, target).mean()
            qf2_loss = self.mse_loss(cur_q2, target).mean()
            critic_loss = qf1_loss + qf2_loss
            return critic_loss

    class ActorLoss(nn.Cell):
        """Actor loss"""

        def __init__(self, model_1, model_2, policy):
            super(AWACLearner.ActorLoss, self).__init__(auto_prefix=False)
            self.model_1 = model_1
            self.model_2 = model_2
            self.policy = policy
            self.min = ops.Minimum()
            self.dist = msd.Normal()
            self.softmax = ops.Softmax(axis=0)
            self.sigmoid = ops.Sigmoid()
            self.cast = ops.Cast()
            self.max_ = Tensor(0.0, mindspore.float32)
            self.min_ = Tensor(-6.0, mindspore.float32)

        def get_std(self, obs):
            _, _, logstd, _ = self.policy(obs)
            logstd = self.sigmoid(logstd)
            logstd = self.min_ + logstd * (self.max_ - self.min_)
            std = logstd.exp()
            return std

        def construct(self, obs, action):
            """Calculate actor loss"""
            # sample
            _, _, logstd, mean_tanh = self.policy(obs)
            logstd = self.cast(logstd, mindspore.float32)
            mean_tanh = self.cast(mean_tanh, mindspore.float32)
            esp = ops.standard_normal(mean_tanh.shape)

            logstd = self.sigmoid(logstd)
            logstd = self.min_ + logstd * (self.max_ - self.min_)
            std = logstd.exp()

            x_t = mean_tanh + esp * std
            act = x_t.clip(min=-1, max=1)

            log_pi_row = self.dist.log_prob(action, mean_tanh, std)
            log_pi = log_pi_row.sum(-1, keepdims=True)

            # compute q value
            q1_pi = self.model_1(obs, act)
            q2_pi = self.model_2(obs, act)
            min_q_pi = self.min(q1_pi, q2_pi)
            # offline q value
            q1_pi_off = self.model_1(obs, action)
            q2_pi_off = self.model_2(obs, action)
            min_q_pi_off = self.min(q1_pi_off, q2_pi_off)

            adv_value = (min_q_pi_off - min_q_pi).reshape(-1)
            weight = self.softmax(adv_value / 2.0).reshape(-1, 1)
            weight = ops.stop_gradient(weight)

            actor_loss = -(log_pi * weight * weight.size).sum()
            return actor_loss

    def __init__(self, params):
        super().__init__()
        self._params_config = params
        self.model_1 = params["critic_net1"]
        self.model_2 = params["critic_net2"]
        target_model_1 = params["target_critic_net1"]
        target_model_2 = params["target_critic_net2"]
        self.policy = params["actor_net"]
        gamma = Tensor(self._params_config["gamma"], mindspore.float32)
        tau = self._params_config["update_factor"]
        self.cell_list = nn.CellList()
        self.cell_list.append(self.model_1)
        self.cell_list.append(self.model_2)
        self.cell_list.append(target_model_1)
        self.cell_list.append(target_model_2)
        critic_trainable_params = (
            self.cell_list[0].trainable_params() + self.cell_list[1].trainable_params()
        )
        critic_target_trainable_params = (
            self.cell_list[2].trainable_params() + self.cell_list[3].trainable_params()
        )
        actor_trainable_params = self.policy.trainable_params()
        self.soft_update = SoftUpdate(
            tau, 1, critic_trainable_params, critic_target_trainable_params
        )
        # train net
        self.critic_loss_net = AWACLearner.CriticLoss(
            self.model_1,
            self.model_2,
            target_model_1,
            target_model_2,
            self.policy,
            gamma,
        )
        self.actor_loss_net = AWACLearner.ActorLoss(
            self.model_1, self.model_2, self.policy
        )

        critic_optim = nn.Adam(
            critic_trainable_params, learning_rate=params["critic_lr"]
        )
        actor_optim = nn.Adam(
            actor_trainable_params, learning_rate=params["actor_lr"], weight_decay=1e-4
        )
        self.critic_train = nn.TrainOneStepCell(self.critic_loss_net, critic_optim)
        self.actor_train = nn.TrainOneStepCell(self.actor_loss_net, actor_optim)

    def learn(self, experience):
        """Calculate the loss and update the target"""
        obs, action, reward, next_obs, terminal = experience
        target = self.critic_loss_net.compute_target(next_obs, reward, terminal)
        critic_loss = self.critic_train(obs, action, target)
        actor_loss = self.actor_train(obs, action)
        self.soft_update()
        return critic_loss, actor_loss, self.actor_loss_net.get_std(obs).mean()
