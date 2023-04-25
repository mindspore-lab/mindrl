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
"""PPO algorithm"""
import mindspore
import mindspore.nn as nn
import mindspore.nn.probability.distribution as msd
import mindspore.ops as ops
import numpy as np
from mindspore import Tensor
from mindspore.common.api import ms_function
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.ops import composite as C
from mindspore.ops import operations as P

from mindspore_rl.agent.actor import Actor
from mindspore_rl.agent.learner import Learner
from mindspore_rl.agent.trainer import Trainer


# pylint: disable=W0613
class PPOPolicy:
    r"""
    This is PPOPolicy class. You should define your networks (PPOActorNet and PPOCriticNet here)
    which you prepare to use in the algorithm. Moreover, you should also define you loss function
    (PPOLossCell here) which calculates the loss between policy and your ground truth value.
    """

    class PPOActorNet(nn.Cell):
        r"""
        PPOActorNet is the actor network of PPO algorithm. It takes a set of state as input
        and outputs miu, sigma of a normal distribution
        """

        def __init__(
            self, input_size, hidden_size1, hidden_size2, output_size, sigma_init_std
        ):
            super(PPOPolicy.PPOActorNet, self).__init__()
            self.linear1_actor = nn.Dense(
                input_size, hidden_size1, weight_init="XavierUniform"
            )
            self.linear2_actor = nn.Dense(
                hidden_size1, hidden_size2, weight_init="XavierUniform"
            )

            self.linear_miu_actor = nn.Dense(hidden_size2, output_size)
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

        def construct(self, x):
            """calculate miu and sigma"""
            x = self.tanh_actor(self.linear1_actor(x))
            x = self.tanh_actor(self.linear2_actor(x))
            miu = self.tanh_actor(self.linear_miu_actor(x))
            miu_shape = miu.shape
            miu = self.reshape(miu, (-1, 6))
            sigma = self.softplus_actor(
                self.bias_add(self.zeros_like(miu), self.bias_sigma_actor)
            )
            miu = self.reshape(miu, miu_shape)
            sigma = self.reshape(sigma, miu_shape)
            return miu, sigma

    class PPOCriticNet(nn.Cell):
        r"""
        PPOCriticNet is the critic network of PPO algorithm. It takes a set of states as input
        and outputs the value of input state.
        """

        def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
            super(PPOPolicy.PPOCriticNet, self).__init__()
            self.linear1_critic = nn.Dense(
                input_size, hidden_size1, weight_init="XavierUniform"
            )
            self.linear2_critic = nn.Dense(
                hidden_size1, hidden_size2, weight_init="XavierUniform"
            )
            self.linear3_critic = nn.Dense(hidden_size2, output_size)
            self.tanh_critic = nn.Tanh()

        def construct(self, x):
            """predict value"""
            x = self.tanh_critic(self.linear1_critic(x))
            x = self.tanh_critic(self.linear2_critic(x))
            x = self.linear3_critic(x)
            return x

    class PPOLossCell(nn.Cell):
        r"""
        PPOLossCell calculates the loss of PPO algorithm
        """

        def __init__(self, actor_net, critic_net, epsilon, critic_coef):
            super(PPOPolicy.PPOLossCell, self).__init__(auto_prefix=False)
            self._actor_net = actor_net
            self._critic_net = critic_net
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
            miu_new, sigma_new = self._actor_net(states)
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
            value_prediction = self._critic_net(states)
            value_prediction = self.squeeze(value_prediction)
            squared_advantage_critic = self.square(discounted_r - value_prediction)
            critic_loss = self.reduce_mean(squared_advantage_critic) * self.critic_coef

            # Total Loss
            total_loss = actor_loss + critic_loss
            return total_loss

    def __init__(self, params):
        self.actor_net = self.PPOActorNet(
            params["state_space_dim"],
            params["hidden_size1"],
            params["hidden_size2"],
            params["action_space_dim"],
            params["sigma_init_std"],
        )
        self.critic_net = self.PPOCriticNet(
            params["state_space_dim"], params["hidden_size1"], params["hidden_size2"], 1
        )
        trainable_parameter = (
            self.critic_net.trainable_params() + self.actor_net.trainable_params()
        )
        optimizer_ppo = nn.Adam(trainable_parameter, learning_rate=params["lr"])
        ppo_loss_net = self.PPOLossCell(
            self.actor_net,
            self.critic_net,
            Tensor(params["epsilon"], mindspore.float32),
            Tensor(params["critic_coef"], mindspore.float32),
        )
        self.ppo_net_train = nn.TrainOneStepCell(ppo_loss_net, optimizer_ppo)
        self.ppo_net_train.set_train(mode=True)


class PPOActor(Actor):
    r"""
    This is an actor class of PPO algorithm, which is used to interact with environment, and
    generate/insert experience (data)
    """

    def __init__(self, params=None):
        super(PPOActor, self).__init__()
        self._params_config = params
        self._environment = params["collect_environment"]
        self._eval_env = params["eval_environment"]
        self._actor_net = params["actor_net"]
        self.norm_dist = msd.Normal()
        self.expand_dims = P.ExpandDims()

    def act(self, phase, state):
        """collect experience and insert to replay buffer (used during training)"""
        if phase != 3:
            miu, sigma = self._actor_net(state)
            action = self.norm_dist.sample((), miu, sigma)
            new_state, reward, _ = self._environment.step(action)
            return reward, new_state, action, miu, sigma

        action, _ = self._actor_net(state)
        new_state, reward, _ = self._eval_env.step(action)
        return reward, new_state


class PPOLearner(Learner):
    r"""
    This is the learner class of PPO algorithm, which is used to update the policy net
    """

    def __init__(self, params):
        super(PPOLearner, self).__init__()
        self._params_config = params
        self.gamma = Tensor(self._params_config["gamma"], mindspore.float32)
        self.iter_times = params["iter_times"]
        self._ppo_net_train = params["ppo_net_train"]
        self._critic_net = params["critic_net"]
        self._actor_net = params["actor_net"]

        self.sub = P.Sub()
        self.mul = P.Mul()
        self.add = P.Add()
        self.zeros_like = P.ZerosLike()
        self.reshape = P.Reshape()
        self.expand_dims = P.ExpandDims()
        self.concat = P.Concat()
        self.stack = P.Stack()
        self.assign = P.Assign()
        self.squeeze = P.Squeeze(2)
        self.squeeze_r = P.Squeeze(3)
        self.moments = nn.Moments(axis=(0, 1, 2), keep_dims=False)
        self.sqrt = P.Sqrt()
        self.norm_dist_old = msd.Normal()
        self.reduce_sum = P.ReduceSum()
        self.zero = Tensor(0, mindspore.float32)
        self.zero_int = Tensor(0, mindspore.int32)

    def learn(self, samples):
        """prepare for the value (advantage, discounted reward), which is used to calculate the loss"""

        def discounted_reward(rewards, v_last, gamma):
            """Compute discounter reward"""
            discounted_r = self.zeros_like(rewards)
            iter_num = self.zero_int
            iter_end = len(rewards[0][0])
            while iter_num < iter_end:
                i = iter_end - iter_num - 1
                v_last = self.add(rewards[:, :, i], self.mul(gamma, v_last))
                discounted_r[:, :, i] = v_last
                iter_num += 1
            return discounted_r

        def gae(
            reward_list, next_state_list, critic_value, v_last, gamma, td_lambda=0.95
        ):
            """Compute advantage"""

            next_critic_value = self._critic_net(next_state_list)
            delta = self.squeeze_r(
                reward_list + gamma * next_critic_value - critic_value
            )
            weighted_discount = gamma * td_lambda
            advantage = self.zeros_like(delta)
            v_last = self.zeros_like(v_last)
            iter_num = self.zero_int
            iter_end = len(delta[0][0])

            while iter_num < iter_end:
                i = iter_end - iter_num - 1
                v_last = self.add(delta[:, :, i], self.mul(weighted_discount, v_last))
                advantage[:, :, i] = v_last
                iter_num += 1
            return advantage

        def _normalized_advantage(advantage, epsilon=1e-8):
            """Normalize the advantage"""
            adv_mean, adv_variance = self.moments(advantage)
            normalized_advantage = (advantage - adv_mean) / (
                self.sqrt(adv_variance) + epsilon
            )
            return normalized_advantage

        (
            state_list,
            action_list,
            reward_list,
            next_state_list,
            miu_list,
            sigma_list,
        ) = samples

        last_state = next_state_list[:, :, -1]
        rewards = self.squeeze_r(reward_list)

        last_value_prediction = self._critic_net(last_state)
        last_value_prediction = self.squeeze(last_value_prediction)
        last_value_prediction = self.zeros_like(last_value_prediction)

        discounted_r = discounted_reward(rewards, last_value_prediction, self.gamma)

        value_prediction = self._critic_net(state_list)
        advantage = gae(
            reward_list,
            next_state_list,
            value_prediction,
            last_value_prediction,
            self.gamma,
        )

        normalized_advantage = _normalized_advantage(advantage)
        log_prob = self.norm_dist_old.log_prob(action_list, miu_list, sigma_list)
        log_prob_old = self.reduce_sum(log_prob, -1)

        i = self.zero
        loss = self.zero

        while i < self.iter_times:
            loss_iter = self._ppo_net_train(
                action_list,
                state_list,
                normalized_advantage,
                log_prob_old,
                discounted_r,
            )
            loss += loss_iter
            i += 1
        return loss / self.iter_times


class PPOTrainer(Trainer):
    r"""
    This is the trainer class of PPO algorithm. It arranges the PPO algorithm
    """

    def __init__(self, msrl, params=None):
        nn.Cell.__init__(self, auto_prefix=False)
        self.all_ep_r = []
        self.all_eval_ep_r = []
        self.zero = Tensor(0, mindspore.float32)
        self.assign = P.Assign()
        self.squeeze = P.Squeeze()
        self.mod = P.Mod()
        self.equal = P.Equal()
        self.less = P.Less()
        self.reduce_mean = P.ReduceMean()
        self.transpose = P.Transpose()
        self.duration = params["duration"]
        self.batch_size = params["batch_size"]
        self.eval_interval = params["eval_interval"]
        self.num_eval_episode = params["num_eval_episode"]
        self.print = P.Print()
        self.slice = P.Slice()
        self.depend = P.Depend()

        self.state = Parameter(Tensor(np.zeros((30, 17)), mindspore.float32))
        self.new_state = Parameter(Tensor(np.zeros((30, 17)), mindspore.float32))
        self.reward = Parameter(Tensor(np.zeros((30, 1)), mindspore.float32))
        self.action = Parameter(Tensor(np.zeros((30, 6)), mindspore.float32))
        self.miu = Parameter(Tensor(np.zeros((30, 6)), mindspore.float32))
        self.sigma = Parameter(Tensor(np.zeros((30, 6)), mindspore.float32))
        super(PPOTrainer, self).__init__(msrl)

    def train(self, episode, callbacks=None, ckpt_path=None):
        """The main function which arranges the algorithm"""
        for i in range(episode):
            _, training_reward = self.train_one_episode()
            self.all_ep_r.append(training_reward.asnumpy())
            print(
                f"Episode {i}, steps: {self.duration}, "
                f"reward: {training_reward.asnumpy():.3f}"
            )

    @mindspore.jit
    def train_one_episode(self):
        """the algorithm in one episode"""
        training_loss = self.zero
        training_reward = self.zero
        j = self.zero
        self.state = self.msrl.collect_environment.reset()

        while self.less(j, self.duration):
            (
                self.reward,
                self.new_state,
                self.action,
                self.miu,
                self.sigma,
            ) = self.msrl.agent_act(self.state)
            self.msrl.replay_buffer_insert(
                [
                    self.state,
                    self.action,
                    self.reward,
                    self.new_state,
                    self.miu,
                    self.sigma,
                ]
            )
            self.state = self.new_state
            reward = self.reduce_mean(self.reward)
            training_reward += reward
            j += 1

        replay_buffer_elements = self.msrl.get_replay_buffer_elements(
            transpose=True, shape=(1, 2, 0, 3)
        )
        state_list = replay_buffer_elements[0]
        action_list = replay_buffer_elements[1]
        reward_list = replay_buffer_elements[2]
        next_state_list = replay_buffer_elements[3]
        miu_list = replay_buffer_elements[4]
        sigma_list = replay_buffer_elements[5]

        training_loss += self.msrl.agent_learn(
            (
                state_list,
                action_list,
                reward_list,
                next_state_list,
                miu_list,
                sigma_list,
            )
        )
        self.msrl.replay_buffer_reset()
        self.print("loss", training_loss)
        self.print("reward", training_reward)
        return training_loss, training_reward

    @mindspore.jit
    def evaluation(self):
        """evaluation function"""
        total_eval_reward = self.zero
        num_eval = self.zero
        while num_eval < self.num_eval_episode:
            eval_reward = self.zero
            state, _ = self.msrl.agent_reset_eval()
            j = self.zero
            while self.less(j, self.duration):
                reward, state = self.msrl.agent_act(state)
                reward = self.msrl.reduce_mean(reward)
                eval_reward += reward
                j += 1
            num_eval += 1
            total_eval_reward += eval_reward
        avg_eval_reward = total_eval_reward / self.num_eval_episode
        return avg_eval_reward
