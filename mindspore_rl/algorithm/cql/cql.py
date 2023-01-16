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
'''Conservative Q-Learning for Offline Reinforcement Learning'''

from mindspore_rl.agent.learner import Learner
from mindspore_rl.agent.actor import Actor
from mindspore_rl.utils import SoftUpdate
from mindspore_rl.algorithm.sac import SACPolicy
import mindspore
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.parameter import Parameter
import mindspore.ops as ops
import mindspore.nn.probability.distribution as msd
from mindspore.ops import functional as F
import numpy as np

SEED = 10
np.random.seed(SEED)


class CQLPolicyAndNetwork(SACPolicy):
    '''CQLPolicyAndNetwork: using the implement of SAC'''
    def __init__(self, params):
        super(CQLPolicyAndNetwork, self).__init__(params)
        self.model_1 = self.critic_net1
        self.model_2 = self.critic_net2
        self.target_model_1 = self.target_critic_net1
        self.target_model_2 = self.target_critic_net2
        self.policy = self.actor_net
        self.eval_policy = self.eval_policy


#pylint: disable=W0223
class CQLActor(Actor):
    '''CQL Actor'''

    def __init__(self, params=None):
        super(CQLActor, self).__init__()
        self.eval_policy = params['eval_policy']

    #pylint: disable=W0221
    def act(self, obs):
        '''return action'''
        action = self.eval_policy(obs)
        return action


class CQLLearner(Learner):
    '''CQL Learner'''

    class CriticLoss(nn.Cell):
        '''Critic loss'''

        def __init__(self, model_1, model_2, target_model_1, target_model_2, policy, gamma, act_dim, num_random):
            super(CQLLearner.CriticLoss, self).__init__(auto_prefix=False)
            self.model_1 = model_1
            self.model_2 = model_2
            self.target_model_1 = target_model_1
            self.target_model_2 = target_model_2
            self.policy = policy
            self.gamma = gamma
            self.dist = msd.Normal(seed=SEED)
            self.tanh = ops.Tanh()
            self.log = ops.Log()
            self.cat = ops.Concat(1)
            self.alpha_value = Tensor(5.0, mindspore.float32)
            self.one = Tensor(1.0, mindspore.float32)
            self.eps = Tensor(1e-6, mindspore.float32)
            self.random_density = Tensor(np.log(0.5**act_dim), mindspore.float32)
            self.minval = Tensor(-1, mindspore.float32)
            self.maxval = Tensor(1, mindspore.float32)
            self.num_random = num_random
            self.mse_loss = nn.MSELoss()

        def sample(self, obs):
            '''sample an action'''
            act_mean, std, _, action = self.policy(obs)
            x_t = self.dist.sample((), act_mean, std)
            action = self.tanh(x_t).clip(min=-0.995, max=0.995)

            log_prob_row = self.dist.log_prob(x_t, act_mean, std)
            log_prob_new = log_prob_row - self.log((self.one - action.pow(2)) + self.eps)
            log_prob = log_prob_new.sum(-1, keepdims=True)
            return action, log_prob

        def get_tensor_values(self, obs, actions):
            '''Get value'''
            obs_temp = obs.expand_dims(1).tile((1, self.num_random, 1)).reshape(\
                (obs.shape[0] * self.num_random, obs.shape[1]))
            q1 = self.model_1(obs_temp, actions)
            q2 = self.model_2(obs_temp, actions)
            q1 = q1.reshape((obs.shape[0], self.num_random, 1))
            q2 = q2.reshape((obs.shape[0], self.num_random, 1))
            return q1, q2

        def construct(self, obs, action, reward, next_obs, terminal):
            '''Calculate critic loss'''
            next_action, _ = self.sample(next_obs)
            q1_next = self.target_model_1(next_obs, next_action)
            q2_next = self.target_model_2(next_obs, next_action)
            target_q = ops.minimum(q1_next, q2_next)
            target_q = reward + self.gamma * (self.one - terminal) * target_q
            target_q = ops.stop_gradient(target_q)

            cur_q1 = self.model_1(obs, action)
            cur_q2 = self.model_2(obs, action)
            qf1_loss = self.mse_loss(cur_q1, target_q)
            qf2_loss = self.mse_loss(cur_q2, target_q)

            ## CQL Start
            random_actions_tensor = F.uniform((cur_q2.shape[0] * self.num_random, action.shape[-1]),\
                 self.minval, self.maxval, seed=SEED)
            random_actions_tensor = ops.stop_gradient(random_actions_tensor)
            temp_cur_obs = obs.expand_dims(1).tile((1, self.num_random, 1)).reshape(\
                (obs.shape[0] * self.num_random, obs.shape[1]))
            curr_actions_tensor, cur_obs_log_pi = self.sample(temp_cur_obs)
            curr_log_pis = cur_obs_log_pi.reshape((obs.shape[0], self.num_random, 1))
            temp_next_obs = next_obs.expand_dims(1).tile((1, self.num_random, 1)).reshape(\
                (next_obs.shape[0] * self.num_random, next_obs.shape[1]))
            new_curr_actions_tensor, next_obs_log_pi = self.sample(temp_next_obs)
            new_log_pis = next_obs_log_pi.reshape((next_obs.shape[0], self.num_random, 1))

            q1_rand, q2_rand = self.get_tensor_values(obs, random_actions_tensor)
            q1_curr_actions, q2_curr_actions = self.get_tensor_values(obs, curr_actions_tensor)
            q1_rand = ops.stop_gradient(q1_rand)
            q2_rand = ops.stop_gradient(q2_rand)

            q1_next_actions, q2_next_actions = self.get_tensor_values(obs, new_curr_actions_tensor)
            curr_log_pis_d = ops.stop_gradient(curr_log_pis)
            new_log_pis_d = ops.stop_gradient(new_log_pis)

            cat_q1 = self.cat((q1_rand - self.random_density, q1_next_actions - new_log_pis_d,\
                 q1_curr_actions - curr_log_pis_d))
            cat_q2 = self.cat((q2_rand - self.random_density, q2_next_actions - new_log_pis_d,\
                 q2_curr_actions - curr_log_pis_d))

            min_qf1_loss = ops.logsumexp(cat_q1, axis=1).mean() * self.alpha_value
            min_qf2_loss = ops.logsumexp(cat_q2, axis=1).mean() * self.alpha_value
            min_qf1_loss_f = min_qf1_loss - (cur_q1.mean() * self.alpha_value)
            min_qf2_loss_f = min_qf2_loss - (cur_q2.mean() * self.alpha_value)

            qf1_loss_final = qf1_loss + min_qf1_loss_f
            qf2_loss_final = qf2_loss + min_qf2_loss_f
            ### CQL done
            critic_loss = qf1_loss_final + qf2_loss_final
            return critic_loss

    class ActorLoss(nn.Cell):
        '''Actor loss'''

        def __init__(self, model_1, model_2, policy):
            super(CQLLearner.ActorLoss, self).__init__(auto_prefix=False)
            self.model_1 = model_1
            self.model_2 = model_2
            self.policy = policy
            self.min = ops.Minimum()
            self.dist = msd.Normal(seed=SEED)
            self.one = Tensor(1.0, mindspore.float32)
            self.eps = Tensor(1e-6, mindspore.float32)
            self.log = ops.Log()
            self.tanh = ops.Tanh()
            self.print = ops.Print()
            self.alpha = Tensor(1.0, mindspore.float32)
            self.half = Tensor(0.5, mindspore.float32)
            self.start = Tensor(40000, mindspore.float32)
            self.curr_step = Parameter(Tensor(0, mindspore.float32), name="cur_step", requires_grad=False)

        def atanh(self, x):
            one_plus_x = (self.one + x).clip(min=1e-6, max=None)
            one_minus_x = (self.one - x).clip(min=1e-6, max=None)
            ans = 0.5 * self.log(one_plus_x / one_minus_x)
            return ans

        def construct(self, obs, action):
            '''Calculate actor loss'''
            ## sample
            act_mean, std, _, _ = self.policy(obs)
            x_t = self.dist.sample((), act_mean, std)
            act = self.tanh(x_t).clip(min=-0.995, max=0.995)
            log_pi_row = self.dist.log_prob(x_t, act_mean, std)
            log_pi_new = log_pi_row - self.log((self.one - act.pow(2)) + self.eps)
            log_pi = log_pi_new.sum(-1, keepdims=True)

            ## value
            q1_pi = self.model_1(obs, act)
            q2_pi = self.model_2(obs, act)
            min_q_pi = self.min(q1_pi, q2_pi)
            actor_loss = (self.alpha * log_pi - min_q_pi).mean()
            ## using BC to warm up
            if self.curr_step < self.start:
                raw_action = self.atanh(action)
                mean, std = self.policy(obs)
                log_prob = self.dist.log_prob(raw_action, mean, std.exp())
                log_prob = log_prob - self.log((self.one - action.pow(2)) + self.eps)
                policy_log_prob = log_prob.sum(-1)
                actor_loss = (log_pi - policy_log_prob).mean()
            self.curr_step += 1
            return actor_loss

    def __init__(self, params):
        super(CQLLearner, self).__init__()
        self._params_config = params
        self.model_1 = params['critic_net1']
        self.model_2 = params['critic_net2']
        target_model_1 = params['target_critic_net1']
        target_model_2 = params['target_critic_net2']
        self.policy = params['actor_net']
        self.action_space_dim = params['action_space_dim']
        gamma = Tensor(self._params_config['gamma'], mindspore.float32)
        num_random = self._params_config['num_random']
        tau = self._params_config['update_factor']
        self.cell_list = nn.CellList()
        self.cell_list.append(self.model_1)
        self.cell_list.append(self.model_2)
        self.cell_list.append(target_model_1)
        self.cell_list.append(target_model_2)
        critic_trainable_params = self.cell_list[0].trainable_params() + self.cell_list[1].trainable_params()
        critic_target_trainable_params = self.cell_list[2].trainable_params() + self.cell_list[3].trainable_params()
        actor_trainable_params = self.policy.trainable_params()
        self.soft_update = SoftUpdate(tau, 1, critic_trainable_params, critic_target_trainable_params)
        ### train net
        critic_loss_net = CQLLearner.CriticLoss(self.model_1, self.model_2, target_model_1, target_model_2,
                                                self.policy, gamma, self.action_space_dim, num_random)
        actor_loss_net = CQLLearner.ActorLoss(self.model_1, self.model_2, self.policy)

        critic_optim = nn.Adam(critic_trainable_params, learning_rate=params['critic_lr'])
        actor_optim = nn.Adam(actor_trainable_params, learning_rate=params['actor_lr'])
        self.critic_train = nn.TrainOneStepCell(critic_loss_net, critic_optim)
        self.actor_train = nn.TrainOneStepCell(actor_loss_net, actor_optim)

    def learn(self, experience):
        '''Calculate the loss and update the target'''
        obs, action, reward, next_obs, terminal = experience
        critic_loss = self.critic_train(obs, action, reward, next_obs, terminal)
        actor_loss = self.actor_train(obs, action)
        self.soft_update()
        return critic_loss, actor_loss
