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
"""DDPG"""
import mindspore
from mindspore import Tensor, nn
from mindspore.common.initializer import Uniform, VarianceScaling
from mindspore.ops import composite as C
from mindspore.ops import operations as P

from mindspore_rl.agent.actor import Actor
from mindspore_rl.agent.learner import Learner
from mindspore_rl.utils import OUNoise, SoftUpdate


class HuberLoss(nn.Cell):
    """Huber Loss"""

    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = Tensor(delta, mindspore.float32)
        self.abs = P.Abs()
        self.square = P.Square()
        self.select = P.Select()
        self.reduce_sum = P.ReduceSum()

    def construct(self, predict, label):
        abs_error = self.abs(predict - label)
        cond = abs_error <= self.delta
        loss = self.select(
            cond,
            0.5 * self.square(abs_error),
            self.delta * abs_error - 0.5 * self.square(self.delta),
        )
        return self.reduce_sum(loss)


class DDPGPolicy:
    """This is DDPGPolicy class. You should define your networks (DDPGActorNet and DDPGCriticNet here)
    which you prepare to use in the algorithm. Moreover, you should also define you loss function
    (DDPGLossCell here) which calculates the loss between policy and your ground truth value.
    """

    class DDPGActorNet(nn.Cell):
        """DDPGActorNet is the actor network of DDPG algorithm. It takes a set of state as input
        and outputs miu, sigma of a normal distribution"""

        def __init__(
            self,
            input_size,
            hidden_size1,
            hidden_size2,
            output_size,
            compute_type=mindspore.float32,
        ):
            super(DDPGPolicy.DDPGActorNet, self).__init__()
            weight_init = VarianceScaling(
                scale=1.0 / 3, mode="fan_in", distribution="uniform"
            )
            self.dense1 = nn.Dense(
                input_size, hidden_size1, weight_init=weight_init
            ).to_float(compute_type)
            self.dense2 = nn.Dense(
                hidden_size1, hidden_size2, weight_init=weight_init
            ).to_float(compute_type)
            last_weight_init = Uniform(scale=0.003)
            self.dense3 = nn.Dense(
                hidden_size2, output_size, weight_init=last_weight_init
            ).to_float(compute_type)
            self.tanh = P.Tanh()
            self.relu = P.ReLU()

        def construct(self, x):
            """calculate"""
            x = self.relu(self.dense1(x))
            x = self.relu(self.dense2(x))
            x = self.tanh(self.dense3(x))
            return x

    class DDPGCriticNet(nn.Cell):
        """DDPGCriticNet is the critic network of DDPG algorithm. It takes a set of states as input
        and outputs the value of input state"""

        def __init__(
            self,
            obs_size,
            action_size,
            hidden_size1,
            hidden_size2,
            output_size,
            compute_type=mindspore.float32,
        ):
            super(DDPGPolicy.DDPGCriticNet, self).__init__()
            weight_init = VarianceScaling(
                scale=1.0 / 3, mode="fan_in", distribution="uniform"
            )
            self.dense1 = nn.Dense(
                obs_size, hidden_size1, weight_init=weight_init
            ).to_float(compute_type)
            self.dense2 = nn.Dense(
                hidden_size1 + action_size, hidden_size2, weight_init=weight_init
            ).to_float(compute_type)
            last_weight_init = Uniform(scale=0.003)
            self.dense3 = nn.Dense(
                hidden_size2, output_size, weight_init=last_weight_init
            ).to_float(compute_type)
            self.concat = P.Concat(axis=-1)
            self.relu = P.ReLU()
            self.cast = P.Cast()

        def construct(self, observation, action):
            """predict value"""
            x = self.relu(self.dense1(observation))
            action = self.cast(action, x.dtype)
            x = self.concat((x, action))
            x = self.relu(self.dense2(x))
            x = self.dense3(x)
            return x

    def __init__(self, params):
        # nn.Cell do not support clone or deepcopy. Create target network manually.
        self.actor_net = self.DDPGActorNet(
            params["state_space_dim"],
            params["hidden_size1"],
            params["hidden_size2"],
            params["action_space_dim"],
            params["compute_type"],
        )
        self.target_actor_net = self.DDPGActorNet(
            params["state_space_dim"],
            params["hidden_size1"],
            params["hidden_size2"],
            params["action_space_dim"],
            params["compute_type"],
        )

        self.critic_net = self.DDPGCriticNet(
            params["state_space_dim"],
            params["action_space_dim"],
            params["hidden_size1"],
            params["hidden_size2"],
            1,
            params["compute_type"],
        )
        self.target_critic_net = self.DDPGCriticNet(
            params["state_space_dim"],
            params["action_space_dim"],
            params["hidden_size1"],
            params["hidden_size2"],
            1,
            params["compute_type"],
        )


class DDPGActor(Actor):
    """This is an actor class of DDPG algorithm, which is used to interact with environment, and
    generate/insert experience (data)"""

    def __init__(self, params=None):
        super().__init__()
        self.actor_net = params["actor_net"]
        self.env = params["collect_environment"]
        self.expand_dims = P.ExpandDims()
        self.squeeze = P.Squeeze()
        low, high = self.env.action_space.boundary
        self.clip_value_min = Tensor(low)
        self.clip_value_max = Tensor(high)
        self.noise = OUNoise(
            params["stddev"], params["damping"], self.env.action_space.shape
        )

    def act(self, phase, params):
        """collect experience and insert to replay buffer (used during training)"""
        actions = self.get_action(phase, params)
        next_obs, rewards, done = self.env.step(actions)
        rewards = self.expand_dims(rewards, 0)
        done = self.expand_dims(done, 0)
        return next_obs, actions, rewards, done

    def get_action(self, phase, params):
        """get action"""
        obs = self.expand_dims(params, 0)
        actions = self.actor_net(obs)
        actions = self.squeeze(actions)
        if phase != 3:
            actions = self.noise(actions)
            actions = C.clip_by_value(actions, self.clip_value_min, self.clip_value_max)
        return actions


class DDPGLearner(Learner):
    """This is the learner class of DDPG algorithm, which is used to update the policy net"""

    class CriticLossCell(nn.Cell):
        """DDPGLossCell calculates the loss of DDPG algorithm"""

        def __init__(self, gamma, target_actor_net, target_critic_net, critic_net):
            super(DDPGLearner.CriticLossCell, self).__init__(auto_prefix=True)
            self.gamma = gamma
            self.target_actor_net = target_actor_net
            self.target_critic_net = target_critic_net
            self.critic_net = critic_net
            self.huber_loss = HuberLoss()

        def construct(self, obs, actions, rewards, next_obs, done):
            """calculate the total loss"""
            # critic Loss
            target_actions = self.target_actor_net(next_obs)
            target_q_values = self.target_critic_net(next_obs, target_actions)
            # One step td error.
            td_targets = rewards + self.gamma * (1.0 - done) * target_q_values
            q_values = self.critic_net(obs, actions)
            critic_loss = self.huber_loss(td_targets, q_values)
            return critic_loss

    class ActorLossCell(nn.Cell):
        """ActorLossCell calculates the loss of DDPG algorithm"""

        def __init__(self, actor_net, critic_net):
            super(DDPGLearner.ActorLossCell, self).__init__(auto_prefix=True)
            self.actor_net = actor_net
            self.critic_net = critic_net
            self.reduce_mean = P.ReduceMean()

        def construct(self, obs):
            """calculate the total loss"""
            actions = self.actor_net(obs)
            q_values = self.critic_net(obs, actions)
            actor_loss = -self.reduce_mean(q_values)
            return actor_loss

    def __init__(self, params):
        super().__init__()
        gamma = params["gamma"]
        self.critic_net = params["critic_net"]
        self.actor_net = params["actor_net"]

        # optimizer network
        critic_optimizer = nn.Adam(
            self.critic_net.trainable_params(), learning_rate=params["critic_lr"]
        )
        actor_optimizer = nn.Adam(
            self.actor_net.trainable_params(), learning_rate=params["actor_lr"]
        )

        # loss network
        self.target_actor_net = params["target_actor_net"]
        self.target_critic_net = params["target_critic_net"]
        critic_loss_cell = self.CriticLossCell(
            gamma, self.target_actor_net, self.target_critic_net, self.critic_net
        )
        critic_loss_cell = self.CriticLossCell(
            gamma, self.target_actor_net, self.target_critic_net, self.critic_net
        )
        actor_loss_cell = self.ActorLossCell(self.actor_net, self.critic_net)

        self.critic_train = nn.TrainOneStepCell(critic_loss_cell, critic_optimizer)
        self.actor_train = nn.TrainOneStepCell(actor_loss_cell, actor_optimizer)
        self.critic_train.set_train(mode=True)
        self.actor_train.set_train(mode=True)

        # soft update network
        factor, interval = params["update_factor"], params["update_interval"]
        params = self.actor_net.trainable_params() + self.critic_net.trainable_params()
        target_params = (
            self.target_actor_net.trainable_params()
            + self.target_critic_net.trainable_params()
        )
        self.soft_updater = SoftUpdate(factor, interval, params, target_params)

    def learn(self, experience):
        """DDPG learners"""
        obs, actions, rewards, next_obs, done = experience
        critic_loss = self.critic_train(obs, actions, rewards, next_obs, done)
        actor_loss = self.actor_train(obs)

        # update target network parameters.
        self.soft_updater()
        return critic_loss + actor_loss
