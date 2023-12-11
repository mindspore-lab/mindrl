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
#pylint: disable=C0103
"""
Implementation of Agent base class.
"""

import numpy as np

import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore.ops import operations as P
import mindspore as ms
import mindspore.nn.probability.distribution as msd
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.common.initializer import initializer, Orthogonal
import mindspore.ops as ops

from mindspore_rl.agent.actor import Actor
from mindspore_rl.agent.learner import Learner
from mindspore_rl.network import GruNet


def _reshape(x, N, L):
    return x.reshape(N, L, *x.shape[1:])


def _cast1(x):
    return F.transpose(x, (1, 0, 2)).reshape((-1, x.shape[2]))


class MAPPOPolicy():
    """This is MAPPO policy class, it contains the network definition and policy implementation"""
    class MAPPOActorNet(nn.Cell):
        """This is the actor net definition"""

        def __init__(self, params):
            super().__init__()
            state_space_dim = params['state_space_dim']
            action_space_dim = params['action_space_dim']
            self.compute_type = params.get('compute_type', ms.float32)
            self.linear1_actor = nn.Dense(state_space_dim,
                                          64,
                                          weight_init=initializer(Orthogonal(
                                              gain=5/3), [64, state_space_dim]),
                                          bias_init=0,
                                          activation=nn.Tanh()).to_float(self.compute_type)

            self.linear2_actor = nn.Dense(64,
                                          64,
                                          weight_init=initializer(
                                              Orthogonal(gain=5/3), [64, 64]),
                                          bias_init=0,
                                          activation=nn.Tanh()).to_float(self.compute_type)
            self.layer_norm_input = nn.LayerNorm(
                normalized_shape=(state_space_dim,), epsilon=1e-5).to_float(self.compute_type)
            self.layer_norm_hidden1 = nn.LayerNorm(
                normalized_shape=(64,), epsilon=1e-5).to_float(self.compute_type)
            self.layer_norm_hidden2 = nn.LayerNorm(
                normalized_shape=(64,), epsilon=1e-5).to_float(self.compute_type)
            self.layer_norm_hidden3 = nn.LayerNorm(
                normalized_shape=(64,), epsilon=1e-5).to_float(self.compute_type)
            self.linear3_actor = nn.Dense(64,
                                          action_space_dim,
                                          weight_init=initializer(Orthogonal(gain=0.01), [action_space_dim, 64]),
                                          bias_init=0).to_float(self.compute_type)
            self.ones_like = P.OnesLike()
            self.gru = GruNet(input_size=64,
                              hidden_size=64).to_float(self.compute_type)
            self.expand_dims = P.ExpandDims()
            self.transpose = P.Transpose()
            self.reshape = P.Reshape()
            self.reduce_any = P.ReduceAny()
            self.concat = P.Concat()
            self.stack = P.Stack()
            self.reduce_sum = P.ReduceSum(keep_dims=False)
            self.zeros = P.Zeros()
            self.cast = P.Cast()

        def construct(self, x, hn, masks):
            """The forward calculation of actor net"""
            x = self.cast(x, self.compute_type)
            hn = self.cast(hn, self.compute_type)
            masks = self.cast(masks, self.compute_type)
            # Feature Extraction
            x = self.layer_norm_input(x)
            x = self.linear1_actor(x)
            x = self.layer_norm_hidden1(x)
            x = self.linear2_actor(x)
            x = self.layer_norm_hidden2(x)

            # RNN
            if x.shape[0] == hn.shape[0]:
                x, hn = self.gru(self.expand_dims(x, 0), self.transpose(
                    (hn * self.expand_dims(masks.repeat(1, 1), -1)), (1, 0, 2)))
                x = x.squeeze(0)
                hn = self.transpose(hn, (1, 0, 2))
            else:
                x = self.reshape(x, (10, 320, 64))
                masks = self.reshape(masks, (10, 320))
                hn = hn.transpose(1, 0, 2)
                rnn_output = []

                has_zero_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                for i in range(len(has_zero_index) - 1):
                    start_idx = has_zero_index[i]
                    end_idx = has_zero_index[i + 1]
                    temp = (hn * self.reshape(masks[start_idx], (1, -1, 1)))
                    rnn_out, hn = self.gru(x[start_idx:end_idx], temp)
                    rnn_output.append(rnn_out[0])

                rnn_output = self.stack(rnn_output)

                x = rnn_output.reshape(3200, -1)
                hn = hn.transpose(1, 0, 2)

            x = self.layer_norm_hidden3(x)

            # output layer for categorical
            x = self.linear3_actor(x)
            out_x = self.cast(x, ms.float32)
            out_hn = self.cast(hn, ms.float32)
            return out_x, out_hn

    class MAPPOCriticNet(nn.Cell):
        """This is the critic net definition"""

        def __init__(self, params):
            super().__init__()
            global_dim = params['environment_config']['global_observation_dim']
            self.compute_type = params.get('compute_type', ms.float32)
            self.linear1_critic = nn.Dense(global_dim,
                                           64,
                                           weight_init=initializer(
                                               Orthogonal(gain=5/3), [64, global_dim]),
                                           bias_init=0,
                                           activation=nn.Tanh()).to_float(self.compute_type)

            self.linear2_critic = nn.Dense(64,
                                           64,
                                           weight_init=initializer(
                                               Orthogonal(gain=5/3), [64, 64]),
                                           bias_init=0,
                                           activation=nn.Tanh()).to_float(self.compute_type)
            self.layer_norm_input_critic = nn.LayerNorm(
                normalized_shape=(global_dim,), epsilon=1e-5).to_float(self.compute_type)
            self.layer_norm_hidden1_critic = nn.LayerNorm(
                normalized_shape=(64,), epsilon=1e-5).to_float(self.compute_type)
            self.layer_norm_hidden2_critic = nn.LayerNorm(
                normalized_shape=(64,), epsilon=1e-5).to_float(self.compute_type)
            self.layer_norm_hidden3_critic = nn.LayerNorm(
                normalized_shape=(64,), epsilon=1e-5).to_float(self.compute_type)
            self.linear3_critic = nn.Dense(64,
                                           1,
                                           weight_init=initializer(
                                               Orthogonal(gain=0.01), [1, 64]),
                                           bias_init=0).to_float(self.compute_type)
            self.gru_critic = GruNet(input_size=64,
                                     hidden_size=64).to_float(self.compute_type)
            self.expand_dims = P.ExpandDims()
            self.transpose = P.Transpose()
            self.stack = P.Stack()
            self.reshape = P.Reshape()
            self.concat = P.Concat()
            self.reduce_any = P.ReduceAny()
            self.reduce_sum = P.ReduceSum(keep_dims=False)
            self.zeros = P.Zeros()

        def construct(self, x, hn, masks):
            """The forward calculation of critic net"""
            x = self.cast(x, self.compute_type)
            hn = self.cast(hn, self.compute_type)
            masks = self.cast(masks, self.compute_type)
            # Feature Extraction
            x = self.layer_norm_input_critic(x)
            x = self.linear1_critic(x)
            x = self.layer_norm_hidden1_critic(x)
            x = self.linear2_critic(x)
            x = self.layer_norm_hidden2_critic(x)
            # RNN

            if x.shape[0] == hn.shape[0]:
                x, hn = self.gru_critic(self.expand_dims(x, 0), self.transpose(
                    (hn * self.expand_dims(masks.repeat(1, 1), -1)), (1, 0, 2)))
                x = x.squeeze(0)
                hn = self.transpose(hn, (1, 0, 2))
            else:
                x = self.reshape(x, (10, 320, 64))
                masks = self.reshape(masks, (10, 320))
                hn = hn.transpose(1, 0, 2)
                rnn_output = []

                has_zero_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                for i in range(len(has_zero_index) - 1):
                    start_idx = has_zero_index[i]
                    end_idx = has_zero_index[i + 1]
                    temp = (hn * self.reshape(masks[start_idx], (1, -1, 1)))
                    rnn_out, hn = self.gru_critic(x[start_idx:end_idx], temp)
                    rnn_output.append(rnn_out[0])

                rnn_output = self.stack(rnn_output)

                x = rnn_output.reshape(3200, -1)
                hn = hn.transpose(1, 0, 2)

            x = self.layer_norm_hidden3_critic(x)

            # output layer for categorical
            x = self.linear3_critic(x)
            out_x = self.cast(x, ms.float32)
            out_hn = self.cast(hn, ms.float32)
            return out_x, out_hn

    class CollectPolicy(nn.Cell):
        """The collect policy of MAPPO"""

        def __init__(self, actor_net):
            super().__init__()
            self.categorical_dist = msd.Categorical()
            self.multinomial = P.Multinomial().add_prim_attr("primitive_target", "CPU")
            self.actor_net = actor_net
            self.expand_dims = P.ExpandDims()
            self.exp = P.Exp()
            self.log = P.Log()
            self.reduce_sum = P.ReduceSum(keep_dims=True)

        def construct(self, inputs_data):
            """calculate the action, log_prob and ht_actor"""
            local_obs, hn_actor, masks = inputs_data
            log_categorical_x, ht_actor = self.actor_net(
                local_obs, hn_actor, masks)
            # transfer log prob to prob
            categorical_x = self.exp(log_categorical_x)
            # normalize
            norm_log_categorical_x = log_categorical_x - \
                self.log(self.reduce_sum(categorical_x, -1))
            norm_action_prob = self.exp(norm_log_categorical_x)

            actions = self.multinomial(norm_action_prob, 1).squeeze(-1)
            log_prob = self.categorical_dist.log_prob(
                actions, norm_action_prob)

            actions = self.expand_dims(actions, 1)

            log_prob = self.expand_dims(log_prob, 1)

            return actions, log_prob, ht_actor

    def __init__(self, params):

        self.actor_net = MAPPOPolicy.MAPPOActorNet(params)
        self.critic_net = MAPPOPolicy.MAPPOCriticNet(params)
        self.collect_policy = MAPPOPolicy.CollectPolicy(self.actor_net)


class MAPPOAgent(nn.Cell):
    """
    The base class for the Agent.
    """

    def __init__(self, actor, learner):
        super(MAPPOAgent, self).__init__(auto_prefix=True)
        self.actor = actor
        self.learner = learner

    def construct(self, command, input_value):
        if command == 1:
            return self.actor.act(input_value)
        if command == 2:
            return self.learner.learn(input_value)
        return 0


#pylint: disable=W0223
class MAPPOActor(Actor):
    """This is the actor class of MAPPO, which is used to obtain the actions of each agent"""

    def __init__(self, params):
        super().__init__()
        self.collect_policy = params['collect_policy']
        self.critic_net = params['critic_net']
        self.onehot = P.OneHot()
        self.assign = P.Assign()
        self.true = Tensor(True, ms.bool_)
        self.zero_float = Tensor(0, ms.float32)
        self.one_float = Tensor(1, ms.float32)

    #pylint: disable=W0221
    def act(self, inputs_data):
        """Use collect policy to calculate the action"""
        local_obs, global_obs, hn_actor, hn_critic, masks = inputs_data

        collect_data = (local_obs, hn_actor, masks)
        actions, log_prob, ht_actor = self.collect_policy(collect_data)
        value_prediction, ht_critic = self.critic_net(
            global_obs, hn_critic, masks)

        onehot_action = self.onehot(actions, 5, self.one_float, self.zero_float).squeeze(1)

        return onehot_action, actions, log_prob, ht_actor, value_prediction, ht_critic


class MAPPOLearner(Learner):
    """This is the learner class of MAPPO, which is used to calculate loss and do the backpropagation of network"""
    class MAPPOActorLossCell(nn.Cell):
        """MAPPOLossCell calculates the loss of MAPPO algorithm"""

        def __init__(self, actor_net):
            super(MAPPOLearner.MAPPOActorLossCell,
                  self).__init__(auto_prefix=False)
            self.actor_net = actor_net
            self.epsilon = 0.2

            self.reduce_mean = P.ReduceMean()
            self.reduce_sum = P.ReduceSum(keep_dims=True)
            self.div = P.Div()
            self.mul = P.Mul()
            self.minimum = P.Minimum()
            self.maximum = P.Maximum()
            self.add = P.Add()
            self.sub = P.Sub()
            self.square = P.Square()
            self.exp = P.Exp()
            self.squeeze = P.Squeeze()
            self.categorical_dist = msd.Categorical()
            self.expand_dims = P.ExpandDims()
            self.huber_delta = 10.0
            self.log = P.Log()
            self.cast = P.Cast()
            self.print = P.Print()

        def construct(self, actions, local_obs, hn_actor, masks, advantage, log_prob_old):
            """calculate the actor loss"""
            # Actor Loss

            # logits
            log_categorical_x, _ = self.actor_net(local_obs, hn_actor, masks)
            # transfer log prob to prob
            categorical_x = self.exp(log_categorical_x)
            # normalize
            norm_log_categorical_x = log_categorical_x - \
                self.log(self.reduce_sum(categorical_x, -1))
            norm_categorical_x = self.exp(norm_log_categorical_x)

            # log problem
            log_prob_new = self.categorical_dist.log_prob(
                actions.squeeze(-1), norm_categorical_x)
            log_prob_new = self.expand_dims(log_prob_new, -1)
            dist_entropy = self.reduce_mean(
                self.categorical_dist.entropy(norm_categorical_x), -1)
            importance_ratio = self.exp(log_prob_new - log_prob_old)

            surr = self.mul(importance_ratio, advantage)
            clip_surr = self.mul(
                C.clip_by_value(importance_ratio, 1. - self.epsilon,
                                1. + self.epsilon), advantage)
            actor_loss = self.reduce_mean(-self.minimum(surr, clip_surr))
            # may need gradient normalization and clip
            entropy_actor_loss = actor_loss - dist_entropy * 0.01
            return entropy_actor_loss

    class MAPPOCriticLossCell(nn.Cell):
        """MAPPOLossCell calculates the loss of MAPPO algorithm"""

        def __init__(self, critic_net, value_normalizer):
            super(MAPPOLearner.MAPPOCriticLossCell,
                  self).__init__(auto_prefix=False)
            self.critic_net = critic_net
            self.epsilon = 0.2

            self.reduce_mean = P.ReduceMean()
            self.reduce_sum = P.ReduceSum()
            self.div = P.Div()
            self.mul = P.Mul()
            self.minimum = P.Minimum()
            self.maximum = P.Maximum()
            self.add = P.Add()
            self.sub = P.Sub()
            self.square = P.Square()
            self.exp = P.Exp()
            self.squeeze = P.Squeeze()
            self.value_normalizer = value_normalizer
            self.huber_delta = 10.0
            self.select = P.Select()
            self.print = P.Print()

        def construct(self, global_obs, hn_critic, masks, discounted_r, value_old):
            """calculate the critic loss"""
            def huber_loss(error, delta):
                a = (error.abs() <= delta).astype(ms.float32)
                b = (error > delta).astype(ms.float32)
                value_1 = self.square(error) * 0.5
                value_2 = delta * (error.abs() - delta * 0.5)
                return a * value_1 + b * value_2
            # Critic Loss
            value_prediction, _ = self.critic_net(global_obs, hn_critic, masks)
            value_clip = C.clip_by_value(
                value_prediction - value_old, -self.epsilon, self.epsilon) + value_old

            self.value_normalizer.update(discounted_r)
            error_clip = self.value_normalizer.normalize(
                discounted_r) - value_clip
            error = self.value_normalizer.normalize(
                discounted_r) - value_prediction

            value_loss_clip = huber_loss(error_clip, self.huber_delta)
            value_loss = huber_loss(error, self.huber_delta)

            critic_loss = self.reduce_mean(
                self.maximum(value_loss_clip, value_loss))
            return critic_loss

    def __init__(self, params) -> None:
        super().__init__()
        self.actor_net = params['actor_net']
        self.critic_net = params['critic_net']
        self.gamma = Tensor(params['gamma'], ms.float32)
        self.td_lambda = params['td_lambda']
        self.iter_time = params['iter_time']
        self.zeros_like = P.ZerosLike()
        self.zero = Tensor(0, ms.float32)
        self.one = Tensor(1, ms.float32)
        self.reshape = P.Reshape()
        self.stack = P.Stack()
        self.concat = P.Concat()
        self.value_normalizer = ValueNormalizer()
        actor_optimizer = nn.Adam(self.actor_net.trainable_params(),
                                  learning_rate=params['learning_rate'])
        critic_optimizer = nn.Adam(self.critic_net.trainable_params(),
                                   learning_rate=params['learning_rate'])
        actor_loss_cell = self.MAPPOActorLossCell(self.actor_net)
        self.actor_train = nn.TrainOneStepCell(
            actor_loss_cell, actor_optimizer)
        critic_loss_cell = self.MAPPOCriticLossCell(
            self.critic_net, self.value_normalizer)
        self.critic_train = nn.TrainOneStepCell(
            critic_loss_cell, critic_optimizer)
        self.expand_dims = P.ExpandDims()
        self.gather = P.Gather()
        self.print = P.Print()
        self.slice = P.Slice()

    #pylint: disable=W0221
    def learn(self, samples):
        """The learn method of MAPPO, it will calculate the loss and update the neural network"""
        local_obs, hn_actor, hn_critic, mask, actions, log_prob, value, reward, global_obs = samples

        def reshape_tensor_2d(tensor):
            reshaped_tensor = self.reshape(tensor.transpose(
                1, 0, 2), (tensor.shape[0] * tensor.shape[1], tensor.shape[2]))
            return reshaped_tensor

        def reshape_tensor_3d(tensor):
            reshaped_tensor = self.reshape(tensor.transpose(
                1, 0, 2, 3), (tensor.shape[0] * tensor.shape[1],) + tensor.shape[2:])
            return reshaped_tensor

        last_global_obs = global_obs[-1]
        last_hn_critic = hn_critic[-1]
        last_episode_mask = mask[-1]
        last_value_prediction = value[-1]
        last_value, _ = self.critic_net(
            last_global_obs, last_hn_critic, last_episode_mask)

        gae = self.zeros_like(last_value)
        discounted_r = self.zeros_like(value)
        weighted_discount = self.gamma * self.td_lambda
        temp_1 = self.value_normalizer.denormalize(last_value)
        temp_2 = self.value_normalizer.denormalize(last_value_prediction)
        delta = reward[-1] + self.gamma * temp_1 * last_episode_mask - temp_2
        gae = delta + weighted_discount * last_episode_mask * gae

        discounted_r[-1] = gae + \
            self.value_normalizer.denormalize(last_value_prediction)

        iter_num = self.zero
        while iter_num < 24:
            step = (24 - iter_num).astype(ms.int32)
            delta = reward[step] + self.gamma * self.value_normalizer.denormalize(
                value[step + 1]) * mask[step] - self.value_normalizer.denormalize(value[step])
            gae = delta + self.gamma * self.td_lambda * mask[step] * gae
            discounted_r[step] = gae + \
                self.value_normalizer.denormalize(value[step])
            iter_num += 1

        advantage = (discounted_r -
                     self.value_normalizer.denormalize(value))[1:]
        norm_advantage = (advantage - advantage.mean()) / \
            (advantage.std() + 1e-5)
        local_obs = reshape_tensor_2d(local_obs[:-1])
        global_obs = reshape_tensor_2d(global_obs[:-1])
        hn_actor = reshape_tensor_3d(hn_actor[:-1])
        hn_critic = reshape_tensor_3d(hn_critic[:-1])
        mask = reshape_tensor_2d(mask[:-1])

        actions = reshape_tensor_2d(actions[1:])
        log_prob = reshape_tensor_2d(log_prob[1:])
        value = reshape_tensor_2d(value[1:])
        norm_advantage = reshape_tensor_2d(norm_advantage)
        discounted_r = reshape_tensor_2d(discounted_r[1:])
        L, N = 10, 320

        global_obs = _reshape(global_obs, N, L)
        local_obs = _reshape(local_obs, N, L)
        actions = _reshape(actions, N, L)
        value = _reshape(value, N, L)
        discounted_r = _reshape(discounted_r, N, L)
        mask = _reshape(mask, N, L)
        log_prob = _reshape(log_prob, N, L)
        norm_advantage = _reshape(norm_advantage, N, L)

        global_obs = _cast1(global_obs)
        local_obs = _cast1(local_obs)
        actions = _cast1(actions)
        value = _cast1(value)
        discounted_r = _cast1(discounted_r)
        mask = _cast1(mask)
        log_prob = _cast1(log_prob)
        norm_advantage = _cast1(norm_advantage)

        iter_learn_time = self.zero
        actor_loss = self.zero
        critic_loss = self.zero

        while iter_learn_time < self.iter_time:
            actor_loss += self.actor_train(actions, local_obs,
                                           hn_actor, mask, norm_advantage, log_prob)
            critic_loss += self.critic_train(global_obs,
                                             hn_critic, mask, discounted_r, value)
            iter_learn_time += 1
        output_loss = actor_loss + critic_loss
        return output_loss


class ValueNormalizer(nn.Cell):
    """The value normalizer"""
    def __init__(self):
        super().__init__()
        self.mean_value = Parameter(
            Tensor(0.0, ms.float32), requires_grad=False)
        self.mean_square_value = Parameter(
            Tensor(0.0, ms.float32), requires_grad=False)
        self.debiasing = Parameter(
            Tensor(0.0, ms.float32), requires_grad=False)
        self.beta = 0.99999
        self.reduce_mean = P.ReduceMean()
        self.square = P.Square()
        self.sqrt = P.Sqrt()
        self.print = P.Print()
        self.inf = Tensor(np.inf, ms.float32)

    def update(self, input_value):
        """Update the mean, variance and debiasing"""
        mean = self.reduce_mean(input_value)
        mean_square = self.reduce_mean(self.square(input_value))
        self.mean_value = self.mean_value * self.beta + mean * (1 - self.beta)
        self.mean_square_value = self.mean_square_value * \
            self.beta + mean_square * (1 - self.beta)
        self.debiasing = self.debiasing * self.beta + (1 - self.beta)
        return True

    def calculate_mean_var(self):
        """Calculate the mean and variance"""
        mean = self.mean_value / \
            C.clip_by_value(self.debiasing, 1e-5, self.inf)
        mean_square = self.mean_square_value / \
            C.clip_by_value(self.debiasing, 1e-5, self.inf)
        variance = C.clip_by_value(
            (mean_square - self.square(mean)), 1e-2, self.inf)
        return mean, variance

    def normalize(self, input_value):
        """Do the normalize for input value"""
        mean, var = self.calculate_mean_var()
        normlized_tensor = (input_value - mean) / self.sqrt(var)
        return normlized_tensor

    def denormalize(self, input_value):
        """Do the denormalized for the input value"""
        mean, var = self.calculate_mean_var()
        denormlized_tensor = input_value * self.sqrt(var) + mean
        return denormlized_tensor
