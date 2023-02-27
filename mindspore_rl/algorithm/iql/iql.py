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
'''Offline Reinforcement Learning With Implicit Q-Learning'''

from mindspore_rl.agent.learner import Learner
from mindspore_rl.agent.actor import Actor
from mindspore_rl.utils import SoftUpdate
from mindspore_rl.algorithm.sac import SACPolicy
import mindspore
import mindspore.nn as nn
from mindspore import Tensor
import mindspore.ops as ops
import mindspore.nn.probability.distribution as msd
import numpy as np

SEED = 10
np.random.seed(SEED)


class IQLPolicyAndNetwork(SACPolicy):
    '''IQLPolicyAndNetwork: using the implement of SAC and adding ValueNet'''
    class ValueNet(nn.Cell):
        '''ValueNet: the state value network of IQL algorithm'''
        def __init__(self, input_size, hidden_sizes, output_size, hidden_act=nn.ReLU,
                     compute_type=mindspore.float32):
            super(IQLPolicyAndNetwork.ValueNet, self).__init__()
            model_list = []
            in_size = input_size
            for _, out_size in enumerate(hidden_sizes):
                model_list.append(nn.Dense(in_size, out_size, weight_init='XavierUniform').to_float(compute_type))
                model_list.append(hidden_act())
                in_size = out_size
            self.model = nn.SequentialCell(model_list)
            self.last_fc = nn.Dense(in_size, output_size)

        def construct(self, obs):
            """predict state value"""
            x = obs
            y = self.model(x)
            y = self.last_fc(y)
            return y

    def __init__(self, params):
        super(IQLPolicyAndNetwork, self).__init__(params)
        self.model_1 = self.critic_net1
        self.model_2 = self.critic_net2
        self.target_model_1 = self.target_critic_net1
        self.target_model_2 = self.target_critic_net2
        self.policy = self.actor_net
        self.eval_policy = self.eval_policy
        compute_type = params.get('compute_type', mindspore.float32)
        self.value_net = self.ValueNet(input_size=params['state_space_dim'],
                                       hidden_sizes=params['hidden_sizes'],
                                       output_size=1,
                                       compute_type=compute_type)


# pylint: disable=W0223
class IQLActor(Actor):
    '''IQL Actor'''

    def __init__(self, params=None):
        super(IQLActor, self).__init__()
        self.policy = params['actor_net']
        self.eval_policy = params['eval_policy']
        self._environment = params['collect_environment']
        self.dist = msd.Normal()
        self.cast = ops.Cast()
        self.sigmoid = ops.Sigmoid()
        self.max_ = Tensor(0., mindspore.float32)
        self.min_ = Tensor(-6., mindspore.float32)

    # pylint: disable=W0221
    def act(self, phase, state):
        '''act with environment'''
        _, _, logstd, mean_tanh = self.policy(state)
        logstd = self.sigmoid(logstd)
        logstd = self.min_ + logstd * (self.max_ - self.min_)
        std = logstd.exp()
        esp = ops.standard_normal(mean_tanh.shape)
        x_t = mean_tanh + esp * std
        actions = x_t.clip(-1, 1)

        next_obs, rewards, done = self._environment.step(actions)
        done = self.cast(done, mindspore.float32).expand_dims(0)
        rewards = rewards.expand_dims(0)
        next_obs = next_obs.expand_dims(0)
        return (actions, rewards, next_obs), done

    # pylint: disable=W0221
    def get_action(self, obs):
        '''return action'''
        action = self.eval_policy(obs)
        return action


class IQLLearner(Learner):
    '''IQL Learner'''

    class CriticLoss(nn.Cell):
        '''Critic loss'''

        def __init__(self, model_1, model_2, policy, value, gamma):
            super(IQLLearner.CriticLoss, self).__init__(auto_prefix=False)
            self.model_1 = model_1
            self.model_2 = model_2
            self.policy = policy
            self.value = value
            self.gamma = gamma

            self.dist = msd.Normal()
            self.one = Tensor(1.0, mindspore.float32)
            self.mse_loss = nn.MSELoss(reduction='none')
            self.sigmoid = ops.Sigmoid()
            self.max_ = Tensor(0., mindspore.float32)
            self.min_ = Tensor(-6., mindspore.float32)
            self.alpha = Tensor([1e-20], mindspore.float32)

        def sample(self, obs):
            '''sample an action'''
            _, _, logstd, act_mean = self.policy(obs)
            esp = ops.standard_normal(act_mean.shape)
            logstd = self.sigmoid(logstd)
            logstd = self.min_ + logstd * (self.max_ - self.min_)
            std = logstd.exp()
            x_t = act_mean + esp * std
            act = x_t.clip(-1., 1.)
            log_pi_row = self.dist.log_prob(act, act_mean, std)
            log_pi = log_pi_row.sum(-1, keepdims=True)
            return act, log_pi

        def compute_target(self, next_obs, reward, terminal):
            '''comput target'''
            next_v = self.value(next_obs)
            target = reward + self.gamma * (self.one - terminal) * next_v
            return target

        def construct(self, obs, action, target):
            '''Calculate critic loss'''
            cur_q1 = self.model_1(obs, action)
            cur_q2 = self.model_2(obs, action)
            qf1_loss = self.mse_loss(cur_q1, target).mean()
            qf2_loss = self.mse_loss(cur_q2, target).mean()
            critic_loss = qf1_loss + qf2_loss
            return critic_loss

    class ActorLoss(nn.Cell):
        '''Actor loss'''

        def __init__(self, target_model_1, target_model_2, policy, value, temperature):
            super(IQLLearner.ActorLoss, self).__init__(auto_prefix=False)
            self.target_model_1 = target_model_1
            self.target_model_2 = target_model_2
            self.policy = policy
            self.value = value
            self.temperature = temperature
            self.min = ops.Minimum()
            self.dist = msd.Normal()

            self.softmax = ops.Softmax(axis=0)
            self.sigmoid = ops.Sigmoid()
            self.max_ = Tensor(0., mindspore.float32)
            self.min_ = Tensor(-6., mindspore.float32)

        def get_std(self, obs):
            _, _, logstd, _ = self.policy(obs)
            logstd = self.sigmoid(logstd)
            logstd = self.min_ + logstd * (self.max_ - self.min_)
            std = logstd.exp()
            return std

        def construct(self, obs, action):
            '''Calculate actor loss'''

            # offline q value
            q1 = self.target_model_1(obs, action)
            q2 = self.target_model_2(obs, action)
            min_q = self.min(q1, q2)
            # state value
            v = self.value(obs)
            # exp(q-v)b
            exp_a = ((min_q - v) * self.temperature).exp()

            # action distribution
            _, _, logstd, act_mean = self.policy(obs)

            logstd = self.sigmoid(logstd)
            logstd = self.min_ + logstd * (self.max_ - self.min_)
            std = logstd.exp()

            log_probs = self.dist.log_prob(action, act_mean, std)

            actor_loss = -(log_probs * exp_a).mean()
            return actor_loss

    class ValueLoss(nn.Cell):
        '''Value loss'''

        def __init__(self, target_model_1, target_model_2, value, quantile):
            super(IQLLearner.ValueLoss, self).__init__(auto_prefix=False)
            self.target_model_1 = target_model_1
            self.target_model_2 = target_model_2
            self.value = value
            self.min = ops.Minimum()
            self.quantile = quantile

            self.dist = msd.Normal()
            self.one = Tensor(1.0, mindspore.float32)
            self.mse_loss = nn.MSELoss(reduction='none')
            self.sigmoid = ops.Sigmoid()
            self.max_ = Tensor(0., mindspore.float32)
            self.min_ = Tensor(-6., mindspore.float32)
            self.alpha = Tensor([1e-20], mindspore.float32)

        def construct(self, obs, action):
            '''Calculate value loss'''

            q1 = self.target_model_1(obs, action)
            q2 = self.target_model_2(obs, action)
            q_min = self.min(q1, q2)
            v = self.value(obs)
            v_err = v - q_min
            v_sign = (v_err > 0)
            v_weight = (1 - v_sign) * self.quantile + v_sign * (1 - self.quantile)
            value_loss = (v_weight * (v_err ** 2)).mean()
            return value_loss

    def __init__(self, params):
        super(IQLLearner, self).__init__()
        self._params_config = params
        self.model_1 = params['critic_net1']
        self.model_2 = params['critic_net2']
        target_model_1 = params['target_critic_net1']
        target_model_2 = params['target_critic_net2']
        self.policy = params['actor_net']
        self.value_model = params['value_net']
        gamma = Tensor(self._params_config['gamma'], mindspore.float32)
        temperature = Tensor(self._params_config['temperature'], mindspore.float32)
        quantile = Tensor(self._params_config['quantile'], mindspore.float32)
        tau = self._params_config['update_factor']
        self.cell_list = nn.CellList()
        self.cell_list.append(self.model_1)
        self.cell_list.append(self.model_2)
        self.cell_list.append(target_model_1)
        self.cell_list.append(target_model_2)
        critic_trainable_params = self.cell_list[0].trainable_params() + self.cell_list[1].trainable_params()
        critic_target_trainable_params = self.cell_list[2].trainable_params() + self.cell_list[3].trainable_params()
        actor_trainable_params = self.policy.trainable_params()
        value_trainable_params = self.value_model.trainable_params()
        self.soft_update = SoftUpdate(tau, 1, critic_trainable_params, critic_target_trainable_params)
        ## train net
        self.critic_loss_net = IQLLearner.CriticLoss(self.model_1, self.model_2, self.policy, self.value_model, gamma)
        self.actor_loss_net = IQLLearner.ActorLoss(target_model_1, target_model_2, self.policy, self.value_model,
                                                   temperature)
        self.value_loss_net = IQLLearner.ValueLoss(target_model_1, target_model_2, self.value_model, quantile)

        # set optimizers and update weights
        critic_optim = nn.Adam(critic_trainable_params, learning_rate=params['critic_lr'])
        actor_optim = nn.Adam(actor_trainable_params, learning_rate=params['actor_lr'], weight_decay=1e-4)
        value_optim = nn.Adam(value_trainable_params, learning_rate=params['value_lr'], weight_decay=1e-4)
        self.critic_train = nn.TrainOneStepCell(self.critic_loss_net, critic_optim)
        self.actor_train = nn.TrainOneStepCell(self.actor_loss_net, actor_optim)
        self.value_train = nn.TrainOneStepCell(self.value_loss_net, value_optim)

    def learn(self, experience):
        '''Calculate the loss and update the target'''
        obs, action, reward, next_obs, terminal = experience
        # update value-net
        value_loss = self.value_train(obs, action)
        # use new value-net to update policy-net
        actor_loss = self.actor_train(obs, action)
        # use new value-net to update critic-net
        target = self.critic_loss_net.compute_target(next_obs, reward, terminal)
        critic_loss = self.critic_train(obs, action, target)
        # soft_update
        self.soft_update()

        return (critic_loss, actor_loss, value_loss), self.actor_loss_net.get_std(obs).mean()
