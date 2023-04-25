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
"""PPO data parallel"""
import argparse
import time

import mindspore
import mindspore.nn as nn
import mindspore.nn.probability.distribution as msd
import mindspore.ops as ops
import numpy as np
from mindspore import Tensor, context
from mindspore.common.api import ms_function
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.communication import get_group_size
from mindspore.communication.management import init
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P

from mindspore_rl.agent.actor import Actor
from mindspore_rl.agent.learner import Learner
from mindspore_rl.agent.trainer import Trainer
from mindspore_rl.core import Session, UniformReplayBuffer
from mindspore_rl.environment import GymEnvironment

context.set_context(mode=context.GRAPH_MODE, device_target="GPU", max_call_depth=100000)

init("nccl")

parser = argparse.ArgumentParser(description="Set the number of actor workers.")
parser.add_argument(
    "-n",
    "--num_actor",
    type=int,
    default=2,
    required=True,
    help="The number of actor workers. Default: 2.",
)
parser.add_argument(
    "-e",
    "--num_collect_environment",
    type=int,
    default=30,
    required=True,
    help="The number of collect_environment. Default: 30.",
)
parser.add_argument(
    "-ep",
    "--num_episode",
    type=int,
    default=100,
    required=True,
    help="The number of episodes. Default: 100.",
)
parser.add_argument(
    "-duration",
    "--duration",
    type=int,
    default=1000,
    required=True,
    help="The duration. Default: 1000.",
)
args = parser.parse_args()
actor_number = int(args.num_actor)
environment_number = int(args.num_collect_environment)
EPISODE = int(args.num_episode)
DURATION = int(args.duration)


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
        and outputs the value of input state
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

    class GradNorm(nn.Cell):
        r"""
        Gradient normalization
        """

        def __init__(self, clip_value):
            super(PPOPolicy.GradNorm, self).__init__(auto_prefix=False)
            self.stack = P.Stack()
            self.norm = nn.Norm()
            self.hyper_map = C.HyperMap()
            self.mul = P.Mul()
            self.select = P.Select()
            self.clip_value = Tensor(clip_value, mindspore.float32)
            self.eps = Tensor(1e-7, mindspore.float32)
            self.one = Tensor(1.0, mindspore.float32)

        def construct(self, grads):
            """calculate gradient normlization"""
            grads_norm = self.hyper_map(self.norm, grads)
            total_norm = self.norm(self.stack(grads_norm))

            cond = total_norm > self.clip_value
            coef = self.clip_value / (total_norm + self.eps)
            coef = self.select(cond, coef, self.one)

            grads = self.hyper_map(F.partial(self.mul, coef), grads)
            return grads

    class AllReduce(nn.Cell):
        r"""
        Grads allreduce
        """

        def __init__(self):
            super(PPOPolicy.AllReduce, self).__init__(auto_prefix=False)
            self.all_reduce = P.AllReduce()
            self.mul = P.Mul()
            self.hyper_map = C.HyperMap()
            self.coef = 1 / get_group_size()

        def construct(self, grads):
            # reduce sum
            grads = self.hyper_map(self.all_reduce, grads)
            # mean
            grads = self.hyper_map(F.partial(self.mul, self.coef), grads)
            return grads

    class TrainOneStepCell(nn.Cell):
        r"""
        Train one step cell
        """

        def __init__(self, network, optimizer, clip_value, sens=1.0):
            super(PPOPolicy.TrainOneStepCell, self).__init__(auto_prefix=False)
            self.network = network
            self.network.set_grad()
            self.optimizer = optimizer
            self.weights = self.optimizer.parameters
            self.grad = C.GradOperation(get_by_list=True, sens_param=True)
            self.grad_norm = PPOPolicy.GradNorm(clip_value)
            self.sens = sens
            self.grad_reducer = PPOPolicy.AllReduce()

        def construct(self, *inputs):
            """Train one step"""
            loss = self.network(*inputs)
            sens = F.fill(loss.dtype, loss.shape, self.sens)
            grads = self.grad(self.network, self.weights)(*inputs, sens)

            # Data Parallel: Apply ReduceMean for gradients across all devices.
            grads = self.grad_reducer(grads)

            loss = F.depend(loss, self.optimizer(grads))
            return loss

    def __init__(self, params):
        """Init for PPOPolicy"""
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
        clip_value = Tensor(Tensor(params["grad_clip"], mindspore.float32))
        self.ppo_net_train = self.TrainOneStepCell(
            ppo_loss_net, optimizer_ppo, clip_value
        )
        self.ppo_net_train.set_train(mode=True)


class PPOActor(Actor):
    r"""
    This is an actor class of PPO algorithm, which is used to interact with environment, and
    generate/insert experience (data)
    """

    def __init__(self, params=None):
        """Init for PPOActor"""
        super(PPOActor, self).__init__()
        self._params_config = params
        self._environment = params["collect_environment"]
        self._eval_env = params["eval_environment"]
        self._buffer = params["replay_buffer"]
        self._actor_net = params["actor_net"]
        self.norm_dist = msd.Normal()
        self.expand_dims = P.ExpandDims()

    def act(self, state):
        """collect experience and insert to replay buffer (used during training)"""
        miu, sigma = self._actor_net(state)
        action = self.norm_dist.sample((), miu, sigma)
        new_state, reward, _ = self._environment.step(action)
        return reward, new_state, action, miu, sigma

    def evaluate(self, state):
        """collect experience (used during evaluation)"""
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

    def learn(self, samples):
        """prepare for the value (advantage, discounted reward), which is used to calculate the loss"""

        def discounted_reward(rewards, v_last, gamma):
            """Compute discounter reward"""
            discounted_r = self.zeros_like(rewards)
            iter_num = self.zero_int
            iter_end = len(rewards[0])
            while iter_num < iter_end:
                i = iter_end - iter_num - 1
                v_last = self.add(rewards[:, i], self.mul(gamma, v_last))
                discounted_r[:, i] = v_last
                iter_num += 1
            return discounted_r

        def gae(
            reward_list, next_state_list, critic_value, v_last, gamma, td_lambda=0.95
        ):
            """Compute advantage"""
            next_critic_value = self._critic_net(next_state_list)
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
        last_state = next_state_list[:, -1]
        rewards = self.squeeze(reward_list)

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
        super(PPOTrainer, self).__init__(msrl)

    def train(self, episode, callbacks=None, ckpt_path=None):
        """The main function which arranges the algorithm"""
        for i in range(episode):
            st = time.time()
            _, training_reward = self.train_one_episode()
            et = time.time()
            print(
                f"Episode {i}, steps: {self.duration}, "
                f"reward: {training_reward.asnumpy():.3f}"
            )
            print("training time ", et - st)

    @mindspore.jit
    def train_one_episode(self):
        """the algorithm in one episode"""
        training_loss = self.zero
        training_reward = self.zero
        j = self.zero
        state = self.msrl.collect_environment.reset()

        while self.less(j, self.duration):
            reward, new_state, action, miu, sigma = self.msrl.agent_act(state)
            self.msrl.replay_buffer_insert(
                [state, action, reward, new_state, miu, sigma]
            )
            state = new_state
            reward = self.reduce_mean(reward)
            training_reward += reward
            j += 1

        replay_buffer_elements = self.msrl.get_replay_buffer_elements(
            transpose=True, shape=(1, 0, 2)
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
                reward, state = self.msrl.agent_evaluate(state)
                reward = self.reduce_mean(reward)
                eval_reward += reward
                j += 1
            num_eval += 1
            total_eval_reward += eval_reward
        avg_eval_reward = total_eval_reward / self.num_eval_episode
        return avg_eval_reward


env_params = {"name": "HalfCheetah-v2"}
eval_env_params = {"name": "HalfCheetah-v2"}

ACT_NUM = actor_number
COLLECT_ENV_NUM = int(environment_number / actor_number)
print(COLLECT_ENV_NUM)

policy_params = {
    "epsilon": 0.2,
    "lr": 1e-3,
    "grad_clip": 3.0,
    "hidden_size1": 200,
    "hidden_size2": 100,
    "sigma_init_std": 0.35,
    "critic_coef": 0.5,
}

learner_params = {
    "gamma": 0.99,
    "state_space_dim": 0,
    "action_space_dim": 0,
    "iter_times": 25,
}

trainer_params = {
    "duration": DURATION,
    "batch_size": 1,
    "eval_interval": 20,
    "num_eval_episode": 3,
}

ppo_algorithm_config = {
    "actor": {
        "number": 1,
        "type": PPOActor,
        "params": None,
        "policies": [],
        "networks": ["actor_net"],
        "environment": True,
        "eval_environment": True,
    },
    "learner": {
        "number": 1,
        "type": PPOLearner,
        "params": learner_params,
        "networks": ["critic_net", "ppo_net_train"],
    },
    "replay_buffer": {
        "type": UniformReplayBuffer,
        "number": 1,
        "capacity": DURATION,
        "data_shape": [
            (environment_number, 17),
            (environment_number, 6),
            (environment_number, 1),
            (environment_number, 17),
            (environment_number, 6),
            (environment_number, 6),
        ],
        "data_type": [
            mindspore.float32,
            mindspore.float32,
            mindspore.float32,
            mindspore.float32,
            mindspore.float32,
            mindspore.float32,
        ],
        "sample_size": 1,
    },
    "policy_and_network": {"type": PPOPolicy, "params": policy_params},
    "collect_environment": {
        "number": environment_number,
        "type": GymEnvironment,
        "params": env_params,
    },
    "eval_environment": {
        "number": 1,
        "type": GymEnvironment,
        "params": eval_env_params,
    },
}

ppo_session = Session(ppo_algorithm_config, params=trainer_params)
ppo_session.run(class_type=PPOTrainer, episode=EPISODE, duration=DURATION)
