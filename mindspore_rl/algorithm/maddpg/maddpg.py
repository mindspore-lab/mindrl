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
"""MADDPG"""
import mindspore
import mindspore.nn as nn
from mindspore import ops
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore import Tensor

from mindspore_rl.agent.actor import Actor
from mindspore_rl.agent.learner import Learner
from mindspore_rl.utils import SoftUpdate


class MADDPGPolicy():
    """This is MADDPGPolicy class. You should define your networks (MADDPGActorNet and MADDPGCriticNet here)
     which you prepare to use in the algorithm. Moreover, you should also define you loss function
     (MADDPGLossCell here) which calculates the loss between policy and your ground truth value.
    """
    class MADDPGActorNet(nn.Cell):
        """MADDPGActorNet is the actor network of MADDPG algorithm. It takes a set of state as input
         and outputs logits"""

        def __init__(self, input_size, hidden_size, output_size, compute_type=mindspore.float32, is_continuous=False):
            super(MADDPGPolicy.MADDPGActorNet, self).__init__()
            self.dense1 = nn.Dense(input_size, hidden_size, weight_init='XavierUniform').to_float(compute_type)
            self.dense2 = nn.Dense(hidden_size, hidden_size, weight_init='XavierUniform').to_float(compute_type)
            self.dense3 = nn.Dense(hidden_size, output_size, weight_init='XavierUniform').to_float(compute_type)
            self.relu = P.ReLU()
            self.continuous = is_continuous
            if self.continuous:
                self.std_fc = nn.Dense(hidden_size, output_size, weight_init='XavierUniform').to_float(compute_type)

        def construct(self, x):
            """calculate """
            x = self.relu(self.dense1(x))
            x = self.relu(self.dense2(x))
            mean = self.dense3(x)
            if self.continuous:
                std = self.std_fc(x)
                return mean, std
            return mean

    class MADDPGCriticNet(nn.Cell):
        """MADDPGCriticNet is the critic network of MADDPG algorithm. It takes a set of states as input
        and outputs the value of input state"""

        def __init__(self, obs_size, action_size, hidden_size, output_size, num_agent,
                     compute_type=mindspore.float32):
            super(MADDPGPolicy.MADDPGCriticNet, self).__init__()
            self.dense1 = nn.Dense((obs_size + action_size) * num_agent, hidden_size, weight_init='XavierUniform')\
                .to_float(compute_type)
            self.dense2 = nn.Dense(hidden_size, hidden_size, weight_init='XavierUniform').to_float(compute_type)
            self.dense3 = nn.Dense(hidden_size, output_size, weight_init='XavierUniform').to_float(compute_type)
            self.concat = P.Concat(axis=1)
            self.relu = P.ReLU()

        def construct(self, observation, action):
            """predict value"""
            x = self.concat((observation, action))
            x = self.relu(self.dense1(x))
            x = self.relu(self.dense2(x))
            x = self.dense3(x)
            return x

    def __init__(self, params):
        # nn.Cell do not support clone or deepcopy. Create target network manually.
        self.actor_net = self.MADDPGActorNet(params.get('state_space_dim'),
                                             params.get('hidden_size'),
                                             params.get('action_space_dim'),
                                             params.get('compute_type'),
                                             params.get('continuous_actions'))

        self.target_actor_net = self.MADDPGActorNet(params.get('state_space_dim'),
                                                    params.get('hidden_size'),
                                                    params.get('action_space_dim'),
                                                    params.get('compute_type'),
                                                    params.get('continuous_actions'))

        self.critic_net = self.MADDPGCriticNet(params.get('state_space_dim'),
                                               params.get('action_space_dim'),
                                               params.get('hidden_size'),
                                               1,
                                               params.get('num_agent'),
                                               params.get('compute_type'))
        self.target_critic_net = self.MADDPGCriticNet(params.get('state_space_dim'),
                                                      params.get('action_space_dim'),
                                                      params.get('hidden_size'),
                                                      1,
                                                      params.get('num_agent'),
                                                      params.get('compute_type'))


class MADDPGAgent(nn.Cell):
    """
    The base class for the Agent.
    """

    def __init__(self, actor, learner):
        super(MADDPGAgent, self).__init__(auto_prefix=True)
        self.actor = actor
        self.learner = learner

    def construct(self, command, input_value):
        if command == 1:
            return self.actor.act(input_value)
        if command == 2:
            return self.learner.learn(input_value)
        return 0


class MADDPGActor(Actor):
    """This is an actor class of MADDPG algorithm, which is used to compute actions"""

    def __init__(self, params=None):
        super(MADDPGActor, self).__init__()
        self.actor_net = params.get('actor_net')
        self.target_actor_net = params.get('target_actor_net')
        self.softmax = P.Softmax(axis=-1)
        self.log = P.Log()
        self.exp = P.Exp()
        self.tanh = P.Tanh()
        self.minval = Tensor(0.0001, mindspore.float32)
        self.maxval = Tensor(0.9999, mindspore.float32)

    def act(self, phase, params):
        '''default act'''
        return

    def get_action(self, phase, params):
        """return action for predict"""
        obs = params
        if phase:
            mean, _ = self.actor_net(obs)
            action = self.tanh(mean)
        else:
            logits = self.actor_net(obs)
            action = self.softmax(logits)
        return action

    def sample_action(self, phase, params, use_target=False):
        """sample action"""
        obs = params
        if use_target:
            policy = self.target_actor_net(obs)
        else:
            policy = self.actor_net(obs)
        # phase = True : continous_actions
        if phase:
            random = ops.standard_normal(shape=policy[0].shape)
            action = policy[0] + self.exp(policy[1]) * random
            action = self.tanh(action)
        else:
            u = F.uniform(policy.shape, self.minval, self.maxval)
            soft_u = self.log(-1.0 * self.log(u))
            action = self.softmax(policy - soft_u)
        return action


class MADDPGLearner(Learner):
    """This is the learner class of MADDPG algorithm, which is used to update the policy net"""

    class CriticLossCell(nn.Cell):
        """MADDPGLossCell calculates the loss of MADDPG algorithm"""
        def __init__(self, critic_net):
            super(MADDPGLearner.CriticLossCell, self).__init__(auto_prefix=True)
            self.critic_net = critic_net
            self.reduce_mean = P.ReduceMean()
            self.square = P.Square()

        def construct(self, obs_n, act_n, target_q):
            """calculate the total loss"""
            # critic Loss
            q = self.critic_net(obs_n, act_n)
            critic_loss = self.reduce_mean(self.square(q - target_q))
            return critic_loss

    class ActorLossCell(nn.Cell):
        """ActorLossCell calculates the loss of MADDPG algorithm"""
        def __init__(self, actor_net, critic_net, continuous_actions):
            super(MADDPGLearner.ActorLossCell, self).__init__(auto_prefix=True)
            self.actor_net = actor_net
            self.critic_net = critic_net
            self.reduce_mean = P.ReduceMean()
            self.square = P.Square()
            self.cat = P.Concat(axis=-1)
            self.continuous_actions = continuous_actions
            self.exp = P.Exp()
            self.tanh = P.Tanh()
            self.log = P.Log()
            self.softmax = P.Softmax(axis=-1)
            self.minval = Tensor(0.0001, mindspore.float32)
            self.maxval = Tensor(0.9999, mindspore.float32)

        def sample_action(self, obs):
            '''sample action'''
            policy = self.actor_net(obs)
            if self.continuous_actions:
                random = ops.standard_normal(shape=policy[0].shape)
                action = policy[0] + self.exp(policy[1]) * random
                action = self.tanh(action)
            else:
                u = F.uniform(policy.shape, self.minval, self.maxval)
                soft_u = self.log(-1.0 * self.log(u))
                action = self.softmax(policy - soft_u)
            return action

        def construct(self, obs_n, act_n, agent_id):
            """calculate the total loss"""
            act_this = self.sample_action(obs_n[:, agent_id, :])
            act_n[:, agent_id, :] = act_this
            q = self.critic_net(obs_n.reshape((obs_n.shape[0], -1)), act_n.reshape((act_n.shape[0], -1)))
            pg_loss = -1.0 * self.reduce_mean(q)
            logits = self.actor_net(obs_n[:, agent_id, :])
            if self.continuous_actions:
                logits = self.cat(logits)
            p_reg = self.reduce_mean(self.square(logits))
            actor_loss = pg_loss + p_reg * 0.001
            return actor_loss


    def __init__(self, params):
        super(MADDPGLearner, self).__init__()
        self.gamma = params.get('gamma')
        self.continuous_actions = params.get('continuous_actions')
        self.critic_net = params.get('critic_net')
        self.actor_net = params.get('actor_net')

        # optimizer network
        critic_optimizer = nn.Adam(self.critic_net.trainable_params(), learning_rate=params.get('learning_rate'))
        actor_optimizer = nn.Adam(self.actor_net.trainable_params(), learning_rate=params.get('learning_rate'))

        # loss network
        self.target_actor_net = params.get('target_actor_net')
        self.target_critic_net = params.get('target_critic_net')
        critic_loss_cell = self.CriticLossCell(self.critic_net)
        actor_loss_cell = self.ActorLossCell(self.actor_net, self.critic_net, self.continuous_actions)

        self.critic_train = nn.TrainOneStepCell(critic_loss_cell, critic_optimizer)
        self.actor_train = nn.TrainOneStepCell(actor_loss_cell, actor_optimizer)
        self.critic_train.set_train(mode=True)
        self.actor_train.set_train(mode=True)

        # soft update network
        factor, interval = params.get('update_factor'), params.get('update_interval')
        cell_list = nn.CellList()
        cell_list.append(self.critic_net)
        cell_list.append(self.actor_net)
        cell_list.append(self.target_critic_net)
        cell_list.append(self.target_actor_net)
        params = cell_list[0].trainable_params() + cell_list[1].trainable_params()
        target_params = cell_list[2].trainable_params() + cell_list[3].trainable_params()
        self.soft_updater = SoftUpdate(factor, interval, params, target_params)
        SoftUpdate(0, 1, params, target_params)()
        self.cast = P.Cast()

    def learn(self, experience):
        """MADDPG learners"""
        obs_n, act_n, obs_next_n, rew, done, target_act_next_n, agent_id = experience

        # train p net
        act_n = ops.stop_gradient(act_n)
        actor_loss = self.actor_train(obs_n, act_n, agent_id)

        # target q
        target_q_next = self.target_critic_net(obs_next_n.reshape((obs_next_n.shape[0], -1)), target_act_next_n)
        target_q = rew + self.gamma * (1.0 - self.cast(done, mindspore.float32)) * target_q_next
        target_q = ops.stop_gradient(target_q)
        # train q net
        critic_loss = self.critic_train(obs_n.reshape((obs_n.shape[0], -1)), \
            act_n.reshape((act_n.shape[0], -1)), target_q)

        # update target network parameters.
        self.soft_updater()
        return critic_loss + actor_loss
