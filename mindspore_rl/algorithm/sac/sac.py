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
"""SAC Agent"""
import numpy as np
import mindspore
from mindspore import Tensor, ops
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter

from mindspore_rl.agent.actor import Actor
from mindspore_rl.agent.learner import Learner
from mindspore_rl.utils import SoftUpdate
from mindspore_rl.algorithm.sac.tanh_normal import TanhMultivariateNormalDiag


class SACPolicy():
    """
    This is SACPolicy class. You should define your networks (SACActorNet and SACCriticNet here)
    which you prepare to use in the algorithm. Moreover, you should also define you loss function
    (SACLossCell here) which calculates the loss between policy and your ground truth value.
    """
    class SACActorNet(nn.Cell):
        """
        SACActorNet is the actor network of SAC algorithm. It takes a set of state as input
        and outputs miu, sigma of a normal distribution
        """
        def __init__(self, input_size, hidden_sizes, output_size, hidden_act=nn.ReLU,
                     conditioned_std=False, compute_type=mindspore.float32):
            super(SACPolicy.SACActorNet, self).__init__()
            in_size = input_size
            model_list = []
            for _, out_size in enumerate(hidden_sizes):
                model_list.append(nn.Dense(in_size, out_size, weight_init='XavierUniform').to_float(compute_type))
                model_list.append(hidden_act())
                in_size = out_size
            self.model = nn.SequentialCell(model_list)
            self.last_fc = nn.Dense(in_size, output_size)

            self.conditioned_std = conditioned_std
            if self.conditioned_std:
                self.last_fc_log_std = nn.Dense(in_size,
                                                output_size,
                                                weight_init='XavierUniform').to_float(compute_type)
            else:
                self.action_log_std = Parameter(Tensor(np.zeros((1, output_size)), mindspore.float32),
                                                name="action_log_std",
                                                requires_grad=True)
            self.exp = ops.Exp()
            self.tanh = ops.Tanh()


        def construct(self, obs):
            """calculate miu and sigma"""
            h = self.model(obs)
            mean = self.last_fc(h)

            if self.conditioned_std:
                log_std = self.last_fc_log_std(h)
                log_std = log_std.clip(-20, 2)
            else:
                log_std = self.action_log_std.broadcast_to(mean.shape)
            std = self.exp(log_std)

            action = self.tanh(mean)
            return mean, std, log_std, action

    class SACCriticNet(nn.Cell):
        """
        SACCriticNet is the critic network of SAC algorithm. It takes a set of states as input
        and outputs the value of input state
        """
        def __init__(self, obs_size, action_size, hidden_sizes, output_size,
                     hidden_act=nn.ReLU, compute_type=mindspore.float32):
            super(SACPolicy.SACCriticNet, self).__init__()
            self.concat = P.Concat(axis=1)
            in_size = obs_size + action_size
            model_list = []
            for _, out_size in enumerate(hidden_sizes):
                model_list.append(nn.Dense(in_size, out_size, weight_init='XavierUniform').to_float(compute_type))
                model_list.append(hidden_act())
                in_size = out_size
            self.model = nn.SequentialCell(model_list)
            self.last_fc = nn.Dense(in_size, output_size)

        def construct(self, obs, action):
            """predict value"""
            x = self.concat((obs, action))
            y = self.model(x)
            y = self.last_fc(y)
            return y

    class RandomPolicy(nn.Cell):
        def __init__(self, action_space_dim):
            super(SACPolicy.RandomPolicy, self).__init__()
            self.uniform = P.UniformReal()
            self.shape = (action_space_dim,)

        def construct(self):
            return self.uniform(self.shape) * 2 - 1

    class CollectPolicy(nn.Cell):
        """Collect Policy"""
        def __init__(self, actor_net):
            super(SACPolicy.CollectPolicy, self).__init__()
            self.actor_net = actor_net
            self.dist = TanhMultivariateNormalDiag(reduce_axis=-1)

        def construct(self, obs):
            means, stds, _, _ = self.actor_net(obs)
            actions = self.dist.sample((), means, stds)
            return actions


    class EvalPolicy(nn.Cell):
        """Eval Policy"""
        def __init__(self, actor_net):
            super(SACPolicy.EvalPolicy, self).__init__()
            self.actor_net = actor_net

        def construct(self, obs):
            _, _, _, action = self.actor_net(obs)
            return action

    def __init__(self, params):
        compute_type = params.get('compute_type', mindspore.float32)
        self.actor_net = self.SACActorNet(input_size=params['state_space_dim'],
                                          hidden_sizes=params['hidden_sizes'],
                                          output_size=params['action_space_dim'],
                                          conditioned_std=params['conditioned_std'],
                                          compute_type=compute_type)
        self.critic_net1 = self.SACCriticNet(obs_size=params['state_space_dim'],
                                             action_size=params['action_space_dim'],
                                             hidden_sizes=params['hidden_sizes'],
                                             output_size=1,
                                             compute_type=compute_type)
        self.critic_net2 = self.SACCriticNet(obs_size=params['state_space_dim'],
                                             action_size=params['action_space_dim'],
                                             hidden_sizes=params['hidden_sizes'],
                                             output_size=1,
                                             compute_type=compute_type)
        self.target_critic_net1 = self.SACCriticNet(obs_size=params['state_space_dim'],
                                                    action_size=params['action_space_dim'],
                                                    hidden_sizes=params['hidden_sizes'],
                                                    output_size=1,
                                                    compute_type=compute_type)
        self.target_critic_net2 = self.SACCriticNet(obs_size=params['state_space_dim'],
                                                    action_size=params['action_space_dim'],
                                                    hidden_sizes=params['hidden_sizes'],
                                                    output_size=1,
                                                    compute_type=compute_type)

        self.init_policy = self.RandomPolicy(params['action_space_dim'])
        self.collect_policy = self.CollectPolicy(self.actor_net)
        self.eval_policy = self.EvalPolicy(self.actor_net)


class SACActor(Actor):
    """
    This is an actor class of SAC algorithm, which is used to interact with environment, and
    generate/insert experience (data)
    """

    def __init__(self, params=None):
        super(SACActor, self).__init__()
        self._params_config = params
        self._environment = params['collect_environment']
        self._eval_env = params['eval_environment']
        self.init_policy = params['init_policy']
        self.collect_policy = params['collect_policy']
        self.eval_policy = params['eval_policy']
        self.expand_dims = P.ExpandDims()
        self.squeeze = P.Squeeze(axis=0)

    def act(self, phase, params):
        """collect experience and insert to replay buffer (used during training)"""
        if phase == 1:
            action = self.init_policy()
            new_state, reward, done = self._environment.step(action)
            return new_state, action, reward, done
        if phase == 2:
            params = self.expand_dims(params, 0)
            action = self.collect_policy(params)
            action = self.squeeze(action)
            new_state, reward, done = self._environment.step(action)
            return new_state, action, reward, done
        if phase == 3:
            params = self.expand_dims(params, 0)
            action = self.eval_policy(params)
            new_state, reward, _ = self._eval_env.step(action)
            return reward, new_state
        self.print("Phase is incorrect")
        return 0

    def get_action(self, phase, params):
        """get action"""
        obs = self.expand_dims(params, 0)
        action = self.eval_policy(obs)
        return action


class SACLearner(Learner):
    """This is the learner class of SAC algorithm, which is used to update the policy net"""

    class CriticLossCell(nn.Cell):
        """CriticLossCell"""
        def __init__(self, gamma, log_alpha, critic_loss_weight, reward_scale_factor, actor_net, target_critic_net1,
                     target_critic_net2, critic_net1, critic_net2):
            super(SACLearner.CriticLossCell, self).__init__(auto_prefix=True)
            self.gamma = gamma
            self.log_alpha = log_alpha
            self.reward_scale_factor = reward_scale_factor
            self.critic_loss_weight = critic_loss_weight
            self.actor_net = actor_net
            self.target_critic_net1 = target_critic_net1
            self.target_critic_net2 = target_critic_net2
            self.critic_net1 = critic_net1
            self.critic_net2 = critic_net2
            self.dist = TanhMultivariateNormalDiag(reduce_axis=-1)
            self.min = P.Minimum()
            self.exp = P.Exp()
            self.mse = nn.MSELoss(reduction='none')

        def construct(self, next_state, reward, state, action, done):
            """Calculate critic loss"""
            next_means, next_stds, _, _ = self.actor_net(next_state)
            next_action, next_log_prob = self.dist.sample_and_log_prob((), next_means, next_stds)

            target_q_value1 = self.target_critic_net1(next_state, next_action).squeeze(axis=-1)
            target_q_value2 = self.target_critic_net2(next_state, next_action).squeeze(axis=-1)
            target_q_value = self.min(target_q_value1, target_q_value2) - self.exp(self.log_alpha) * next_log_prob
            td_target = self.reward_scale_factor * reward + self.gamma * (1 - done.squeeze(axis=-1)) * target_q_value

            pred_td_target1 = self.critic_net1(state, action).squeeze(axis=-1)
            pred_td_target2 = self.critic_net2(state, action).squeeze(axis=-1)
            critic_loss1 = self.mse(td_target, pred_td_target1)
            critic_loss2 = self.mse(td_target, pred_td_target2)
            critic_loss = (critic_loss1 + critic_loss2).mean()
            return critic_loss * self.critic_loss_weight


    class ActorLossCell(nn.Cell):
        """ActorLossCell"""
        def __init__(self, log_alpha, actor_loss_weight, actor_net, critic_net1, critic_net2, actor_mean_std_reg,
                     actor_mean_reg_weight, actor_std_reg_weight):
            super(SACLearner.ActorLossCell, self).__init__(auto_prefix=False)
            self.log_alpha = log_alpha
            self.actor_net = actor_net
            self.actor_loss_weight = actor_loss_weight
            self.critic_net1 = critic_net1
            self.critic_net2 = critic_net2
            self.dist = TanhMultivariateNormalDiag(reduce_axis=-1)
            self.min = P.Minimum()
            self.exp = P.Exp()

            self.actor_mean_std_reg = actor_mean_std_reg
            if self.actor_mean_std_reg:
                self.actor_mean_reg_weight = actor_mean_reg_weight
                self.actor_std_reg_weight = actor_std_reg_weight

        def construct(self, state):
            """Calculate actor loss"""
            means, stds, log_std, _ = self.actor_net(state)
            action, log_prob = self.dist.sample_and_log_prob((), means, stds)

            target_q_value1 = self.critic_net1(state, action)
            target_q_value2 = self.critic_net2(state, action)
            target_q_value = self.min(target_q_value1, target_q_value2).squeeze(axis=-1)
            actor_loss = (self.exp(self.log_alpha) * log_prob - target_q_value).mean()

            actor_reg_loss = 0.
            if self.actor_mean_std_reg:
                mean_reg_loss = self.actor_mean_reg_weight * (means ** 2).mean()
                std_reg_loss = self.actor_std_reg_weight * (log_std ** 2).mean()
                actor_reg_loss = mean_reg_loss + std_reg_loss

            return actor_loss * self.actor_loss_weight + actor_reg_loss

    class AlphaLossCell(nn.Cell):
        """AlphaLossCell"""
        def __init__(self, log_alpha, target_entropy, alpha_loss_weight, actor_net):
            super(SACLearner.AlphaLossCell, self).__init__(auto_prefix=False)
            self.log_alpha = log_alpha
            self.target_entropy = target_entropy
            self.alpha_loss_weight = alpha_loss_weight
            self.actor_net = actor_net
            self.dist = TanhMultivariateNormalDiag(reduce_axis=-1)

        def construct(self, state_list):
            means, stds, _, _ = self.actor_net(state_list)
            _, log_prob = self.dist.sample_and_log_prob((), means, stds)
            entropy_diff = -log_prob - self.target_entropy
            alpha_loss = self.log_alpha * entropy_diff
            alpha_loss = alpha_loss.mean()
            return alpha_loss * self.alpha_loss_weight


    def __init__(self, params):
        super(SACLearner, self).__init__()
        self._params_config = params
        gamma = Tensor(self._params_config['gamma'], mindspore.float32)
        actor_net = params['actor_net']
        critic_net1 = params['critic_net1']
        critic_net2 = params['critic_net2']
        target_critic_net1 = params['target_critic_net1']
        target_critic_net2 = params['target_critic_net2']

        log_alpha = params['log_alpha']
        log_alpha = Parameter(Tensor([log_alpha,], mindspore.float32), name='log_alpha', requires_grad=True)

        critic_loss_net = SACLearner.CriticLossCell(gamma,
                                                    log_alpha,
                                                    params['critic_loss_weight'],
                                                    params['reward_scale_factor'],
                                                    actor_net,
                                                    target_critic_net1,
                                                    target_critic_net2,
                                                    critic_net1,
                                                    critic_net2)
        actor_loss_net = SACLearner.ActorLossCell(log_alpha,
                                                  params['actor_loss_weight'],
                                                  actor_net,
                                                  critic_net1,
                                                  critic_net2,
                                                  params.get('actor_mean_std_reg', False),
                                                  params.get('actor_mean_reg_weight', 0.),
                                                  params.get('actor_std_reg_weight', 0.))

        critic_trainable_params = critic_net1.trainable_params() + critic_net2.trainable_params()
        critic_optim = nn.Adam(critic_trainable_params, learning_rate=params['critic_lr'])
        actor_optim = nn.Adam(actor_net.trainable_params(), learning_rate=params['actor_lr'])


        self.critic_train = nn.TrainOneStepCell(critic_loss_net, critic_optim)
        self.actor_train = nn.TrainOneStepCell(actor_loss_net, actor_optim)

        self.train_alpha_net = params['train_alpha_net']
        if self.train_alpha_net:
            alpha_loss_net = SACLearner.AlphaLossCell(log_alpha,
                                                      params['target_entropy'],
                                                      params['alpha_loss_weight'],
                                                      actor_net)
            alpha_optim = nn.Adam([log_alpha], learning_rate=params['alpha_lr'])
            self.alpha_train = nn.TrainOneStepCell(alpha_loss_net, alpha_optim)

        factor, interval = params['update_factor'], params['update_interval']
        params = critic_net1.trainable_params() + critic_net2.trainable_params()
        target_params = target_critic_net1.trainable_params() + target_critic_net2.trainable_params()
        self.soft_updater = SoftUpdate(factor, interval, params, target_params)

    def learn(self, experience):
        """learn"""
        state, action, reward, next_state, done = experience
        reward = reward.squeeze(axis=-1)

        critic_loss = self.critic_train(next_state, reward, state, action, done)
        actor_loss = self.actor_train(state)

        alpha_loss = 0.
        if self.train_alpha_net:
            alpha_loss = self.alpha_train(state)

        self.soft_updater()
        loss = critic_loss + actor_loss + alpha_loss
        return loss
