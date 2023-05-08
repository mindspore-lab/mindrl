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
"""COMA"""
import copy
import math

import mindspore as ms
import mindspore.nn.probability.distribution as msd
import numpy as np
from mindspore import Tensor, jit, nn, value_and_grad
from mindspore.common.api import cells_compile_cache
from mindspore.common.initializer import HeUniform, initializer
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P

from mindspore_rl.agent import Actor, Learner


class COMAPolicy:
    """COMA Policy"""

    class COMACriticNet(nn.Cell):
        """COMACriticNet"""

        def __init__(self, params):
            super().__init__()
            input_shape = params["environment_config"]["global_observation_dim"]
            input_shape += params["state_space_dim"]
            input_shape += (
                params["action_space_dim"]
                * params["environment_config"]["num_agent"]
                * 2
            )
            input_shape += params["environment_config"]["num_agent"]
            self.fc1 = nn.Dense(input_shape, 128, activation="relu")
            self.fc2 = nn.Dense(128, 128, activation="relu")
            self.fc3 = nn.Dense(128, params["action_space_dim"])

            self.num_env = 8
            self.num_agent = params["environment_config"]["num_agent"]
            self.agent_id = Tensor(
                np.eye(self.num_agent),
                ms.float32,
            )

        def construct(self, inputs):
            x = self.fc1(inputs)
            x = self.fc2(x)
            x = self.fc3(x)
            return x

        def build_batch_inputs(self, state, obs, last_actions_onehot):
            """build batch inputs"""
            num_env = obs.shape[0]
            max_length = obs.shape[1]
            num_agents = obs.shape[2]
            n_actions = last_actions_onehot.shape[-1]

            inputs = []
            inputs.append(state.unsqueeze(2).repeat(num_agents, axis=2))
            inputs.append(obs)

            actions = (
                last_actions_onehot[:, 1:]
                .view(num_env, max_length, 1, -1)
                .repeat(num_agents, 2)
            )
            agent_mask = 1 - self.agent_id
            agent_mask = (
                agent_mask.view(-1, 1).repeat(n_actions, 1).view(num_agents, -1)
            )
            actions = actions * agent_mask.unsqueeze(0).unsqueeze(0)
            inputs.append(actions)

            last_actions = last_actions_onehot[:, :-1]
            last_actions = last_actions.view(num_env, max_length, 1, -1).repeat(
                num_agents, 2
            )
            inputs.append(last_actions)

            inputs.append(
                self.agent_id.unsqueeze(0)
                .repeat(max_length, 0)
                .unsqueeze(0)
                .repeat(num_env, 0)
            )

            inputs = F.concat(inputs, axis=-1)
            return inputs

        def build_step_inputs(self, state, obs, last_actions_onehot, t):
            """build step inputs"""
            num_env = obs.shape[0]
            num_agents = obs.shape[2]
            n_actions = last_actions_onehot.shape[-1]

            inputs = []
            inputs.append(
                state[:, t].unsqueeze(1).unsqueeze(2).repeat(num_agents, axis=2)
            )
            inputs.append(obs[:, t].unsqueeze(1))

            actions = (
                last_actions_onehot[:, t + 1]
                .unsqueeze(1)
                .view(num_env, 1, 1, -1)
                .repeat(num_agents, 2)
            )
            agent_mask = 1 - self.agent_id
            agent_mask = (
                agent_mask.view(-1, 1).repeat(n_actions, 1).view(num_agents, -1)
            )
            actions = actions * agent_mask.unsqueeze(0).unsqueeze(0)
            inputs.append(actions)

            last_actions = (
                last_actions_onehot[:, t].view(num_env, 1, 1, -1).repeat(num_agents, 2)
            )
            inputs.append(last_actions)

            inputs.append(
                self.agent_id.unsqueeze(0).repeat(1, 0).unsqueeze(0).repeat(num_env, 0)
            )

            inputs = F.concat(inputs, axis=-1)
            return inputs

        def clone(self):
            new_obj = copy.deepcopy(self)
            cells_compile_cache[id(new_obj)] = new_obj.compile_cache
            return new_obj

    class COMAPolicyNet(nn.Cell):
        """The policy net implementation"""

        def __init__(self, params):
            super().__init__()
            self.hidden_size = params["hidden_size"]
            self.output_size = params["action_space_dim"]
            self.num_agent = params["environment_config"]["num_agent"]
            self.input_size = (
                params["state_space_dim"] + self.num_agent + self.output_size
            )

            self.fc1 = nn.Dense(
                self.input_size,
                self.hidden_size,
                activation=nn.ReLU(),
                weight_init=initializer(
                    HeUniform(negative_slope=math.sqrt(5)),
                    [self.hidden_size, self.input_size],
                    ms.float32,
                ),
            )
            self.gru_net = nn.GRU(self.hidden_size, self.hidden_size)
            self.fc2 = nn.Dense(
                self.hidden_size,
                self.output_size,
                weight_init=initializer(
                    HeUniform(negative_slope=math.sqrt(5)),
                    [self.output_size, self.hidden_size],
                    ms.float32,
                ),
            )

            self.concat = P.Concat()
            self.reshape = P.Reshape()
            self.expand_dims = P.ExpandDims()
            self.squeeze = P.Squeeze(axis=0)

        def construct(self, x, hx):
            x = self.fc1(x)
            x = self.expand_dims(x, 0)
            hx = self.expand_dims(hx, 0)
            hy, _ = self.gru_net(x, hx)
            hy = self.squeeze(hy)
            x = self.fc2(hy)
            return x, hy

        def clone(self):
            new_obj = copy.deepcopy(self)
            cells_compile_cache[id(new_obj)] = new_obj.compile_cache
            return new_obj

    class CollectPolicy(nn.Cell):
        """Collect policy"""

        def __init__(self, network, params):
            super().__init__()
            self.epsi_high = Tensor(params["epsi_high"], ms.float32)
            self.epsi_low = Tensor(params["epsi_low"], ms.float32)
            self.num_agent = params["environment_config"]["num_agent"]
            time_length = params["time_length"]
            self.delta = (self.epsi_high - self.epsi_low) / time_length
            self.network = network

            self.dist = msd.Categorical()

        def construct(self, params, step):
            """consturct"""
            agent_inputs, hx, avail_action = params
            agent_outs, hy = self.network(agent_inputs, hx)
            agent_outs = F.softmax(agent_outs)

            # epsilon-decay exploration
            decayed_value = self.epsi_high - self.delta * step
            action_prob = (1 - decayed_value) * agent_outs + F.ones_like(agent_outs) * (
                decayed_value / agent_outs.shape[-1]
            )

            # mask unavailable action
            action_prob = action_prob.reshape(-1, self.num_agent, agent_outs.shape[-1])
            action_prob[avail_action == 0] = 0

            # normalize
            action_prob = action_prob / action_prob.sum(-1, keepdims=True)
            random_action = self.dist.sample((), action_prob)

            # action_prob = action_prob.reshape(-1, action_prob.shape[-1])

            # random_action = []
            # for i in range(action_prob.shape[0]):
            #     np.random.seed(42)
            #     random_action.append(
            #         np.random.choice(11, 1, p=action_prob.asnumpy()[i]).item()
            #     )
            # random_action = Tensor(np.array(random_action, np.int32))
            # random_action = random_action.reshape(8, 5)

            return random_action, hy

    class EvalPolicy(nn.Cell):
        """Eval policy"""

        def __init__(self, network):
            super().__init__()
            self.network = network
            self.categorical = msd.Categorical()

        def construct(self, params):
            x, hx, avail_action = params
            x, hy = self.network(x, hx)
            x[avail_action == 0] = -9999999

            greed_action = self.categorical.mode(x)
            greed_action = greed_action.unsqueeze(1)
            return greed_action, hy

    def __init__(self, params):
        self.policy_net = self.COMAPolicyNet(params)
        self.critic_net = self.COMACriticNet(params)
        self.target_critic_net = self.critic_net.clone()

        self.collect_policy = self.CollectPolicy(self.policy_net, params)
        self.eval_policy = self.EvalPolicy(self.policy_net)


class COMAActor(Actor):
    """The actor implementation"""

    def __init__(self, params):
        super().__init__()
        self.collect_policy = params["collect_policy"]
        self.collect_environment = params["collect_environment"]

    def act(self, phase, params):
        """How to get action and interact with environment"""
        local_obs, hx, avail_action, epsilon_steps = params
        action, hx = self.collect_policy((local_obs, hx, avail_action), epsilon_steps)
        (
            new_state,
            reward,
            done,
            global_obs,
            new_avail_action,
        ) = self.collect_environment.step(action)
        return new_state, done, reward, action, hx, global_obs, new_avail_action

    def get_action(self, phase, params):
        local_obs, hx, avail_action, epsilon_steps = params
        action, hx = self.collect_policy((local_obs, hx, avail_action), epsilon_steps)
        return action, hx


class CriticLossCell(nn.Cell):
    """Critic loss cell"""

    def __init__(self, critic):
        super().__init__()
        self.critic = critic

    def construct(
        self, state, obs, actions, last_actions_onehot, q_vals, targets, mask_t, t
    ):
        """construct"""
        bs = obs.shape[0]
        num_agents = obs.shape[2]
        n_actions = last_actions_onehot.shape[-1]

        inputs = self.critic.build_step_inputs(state, obs, last_actions_onehot, t)
        q_t = self.critic(inputs)
        q_vals[:, t] = q_t.view(bs, num_agents, n_actions)
        q_taken = (
            q_t.gather_elements(3, actions[:, t].unsqueeze(1)).squeeze(3).squeeze(1)
        )
        target_t = targets[:, t]

        td_error = q_taken - target_t

        masked_td_error = td_error * mask_t
        loss = (masked_td_error**2).sum() / mask_t.sum()
        return loss, q_vals


class ActorLossCell(nn.Cell):
    """Actor loss cell"""

    def __init__(self, actor):
        super().__init__()
        self.mac = actor
        self.epsi_high = 0.5
        self.epsi_low = 0.01
        time_length = 100000
        self.delta = (self.epsi_high - self.epsi_low) / time_length

    def construct(
        self, obs, actions, last_actions_onehot, q_vals, avail_actions, mask, max_t
    ):
        """construct"""
        bs = obs.shape[0]
        num_agents = obs.shape[2]
        n_actions = last_actions_onehot.shape[-1]

        mac_out = F.zeros_like(q_vals)
        hx = F.zeros((bs * num_agents, 64))
        t = Tensor(0)
        while t < max_t:
            inputs = []
            inputs.append(obs[:, t])
            inputs.append(last_actions_onehot[:, t])
            inputs.append(F.eye(num_agents, num_agents).unsqueeze(0).repeat(bs, 0))

            inputs = F.concat(inputs, axis=-1)
            inputs = inputs.reshape(-1, inputs.shape[-1])
            agent_outs, hx = self.mac(inputs, hx)
            agent_outs = F.softmax(agent_outs)

            decayed_value = self.epsi_high - self.delta * t
            agent_outs = (1 - decayed_value) * agent_outs + F.ones_like(agent_outs) * (
                decayed_value / agent_outs.shape[-1]
            )
            agent_outs = agent_outs.reshape(bs, num_agents, -1)
            mac_out[:, t] = agent_outs
            t += 1

        mac_out[avail_actions == 0] = 0
        mac_out = mac_out / mac_out.sum(-1, keepdims=True)
        mac_out[avail_actions == 0] = 0

        q_vals = q_vals.reshape(-1, n_actions)
        pi = mac_out.view(-1, n_actions)
        baseline = (pi * q_vals).sum(-1)
        q_taken = q_vals.gather_elements(1, actions.reshape(-1, 1)).squeeze(1)
        pi_taken = pi.gather_elements(1, actions.reshape(-1, 1)).squeeze(1)

        pi_taken[mask == 0] = 1.0
        log_pi_taken = F.log(pi_taken)

        advantages = q_taken - baseline
        coma_loss = -((advantages * log_pi_taken) * mask).sum() / mask.sum()
        return coma_loss


class COMALearner(Learner):
    """COMA Learner"""

    def __init__(self, params):
        super().__init__()
        self.policy = params["policy_net"]
        self.critic = params["critic_net"]
        self.target_critic = params["target_critic_net"]
        self.gamma = 0.99
        self.td_lambda = 0.8
        self.clip_norm = params["clip_norm"]

        critic_loss_net = CriticLossCell(self.critic)
        self.critic_grad_fn = value_and_grad(
            critic_loss_net,
            grad_position=None,
            weights=self.critic.trainable_params(),
            has_aux=True,
        )
        self.critic_optim = nn.RMSProp(
            self.critic.trainable_params(),
            learning_rate=params["critic_lr"],
            decay=params["decay"],
            epsilon=params["epsilon"],
        )

        actor_loss_net = ActorLossCell(self.policy)
        self.actor_grad_fn = value_and_grad(
            actor_loss_net,
            grad_position=None,
            weights=self.policy.trainable_params(),
            has_aux=False,
        )
        self.actor_optim = nn.RMSProp(
            self.policy.trainable_params(),
            learning_rate=params["actor_lr"],
            decay=params["decay"],
            epsilon=params["epsilon"],
        )

    @jit
    def learn(self, experience):
        (
            state,
            obs,
            actions,
            avail_actions,
            rewards,
            terminated,
            last_actions_onehot,
            filled,
        ) = experience
        num_agents = obs.shape[2]
        max_t = filled.sum(1).max()

        terminated = terminated.float()
        mask = filled.to(ms.float32)
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        critic_loss, q_vals = self._train_critic(
            state, obs, rewards, terminated, actions, last_actions_onehot, mask, max_t
        )

        mask = mask.repeat(num_agents, 2).view(-1)

        actor_loss, grads = self.actor_grad_fn(
            obs, actions, last_actions_onehot, q_vals, avail_actions, mask, max_t
        )
        grads = C.clip_by_global_norm(grads, self.clip_norm)
        actor_loss = F.depend(actor_loss, self.actor_optim(grads))
        return critic_loss, actor_loss

    def _train_critic(
        self, state, obs, rewards, terminated, actions, last_actions_onehot, mask, max_t
    ):
        """train critic"""
        num_agents = obs.shape[2]
        batch_inputs = self.target_critic.build_batch_inputs(
            state, obs, last_actions_onehot
        )
        target_q_vals = self.target_critic(batch_inputs)
        targets_taken = target_q_vals.gather_elements(3, actions).squeeze(3)

        targets = self.build_td_lambda_targets(
            rewards, terminated, mask, targets_taken, self.gamma, self.td_lambda, max_t
        )

        q_vals = F.zeros_like(target_q_vals)
        loss = 0.0

        t = max_t - 1
        while t >= 0:
            mask_t = mask[:, t].broadcast_to((-1, num_agents))

            (loss, q_vals), grads = self.critic_grad_fn(
                state, obs, actions, last_actions_onehot, q_vals, targets, mask_t, t
            )
            grads = C.clip_by_global_norm(grads, self.clip_norm)
            loss = F.depend(loss, self.critic_optim(grads))
            t -= 1
        return loss / (max_t - 1), q_vals

    def build_td_lambda_targets(
        self, rewards, terminated, mask, target_qs, gamma, td_lambda, max_t
    ):
        """build td lambda target"""
        ret = F.zeros_like(target_qs)
        t = max_t - 1
        while t >= 0:
            ret[:, t] = td_lambda * gamma * ret[:, t + 1] + mask[:, t] * (
                rewards[:, t]
                + (1 - td_lambda) * gamma * target_qs[:, t + 1] * (1 - terminated[:, t])
            )
            t -= 1
        return ret
