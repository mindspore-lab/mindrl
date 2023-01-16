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
"""GAIL Agent"""
import numpy as np
import mindspore
from mindspore import Tensor, ops
import mindspore.nn as nn
from mindspore.ops import functional as F
from mindspore_rl.agent.learner import Learner
from mindspore_rl.core.uniform_replay_buffer import UniformReplayBuffer


class ExpertReplayBuffer(nn.Cell):
    """
    The replay buffer contains expert experience data.

    Args:
        traj_list (list[dict[numpy.array]]): expert data contains 'observations', 'actions' and 'rewards'.
        capacity (int): the capacity of the buffer.
        shapes (list[int]): the shape of each tensor in a buffer element.
        types (list[mindspore.dtype]): the data type of each tensor in a buffer element.

    """
    def __init__(self, traj_list, batch_size, shapes, dtypes):
        super(ExpertReplayBuffer, self).__init__()
        capacity = sum([len(traj['observations']) for traj in traj_list])
        self.buffer = UniformReplayBuffer(batch_size, capacity, shapes, dtypes)
        episode_reward = []
        for traj in traj_list:
            self.buffer.insert([Tensor(traj['observations'], dtypes[0]), Tensor(traj['actions'], dtypes[1])])
            episode_reward.append(traj['rewards'].sum())
        print('Expert mean reward:', sum(episode_reward) / len(episode_reward))

    def sample(self):
        """
        Randomly choose a batch of expert data from the replay buffer.
        """
        return self.buffer.sample()


class Discriminator(nn.Cell):
    """
    Discriminator.

    Args:
        input (int): input dimension.
        hidden_size (list[int]): hidden layer dimensions.
        hid_act (list[int]): activate function used in hidden layer.
        use_bn (bool): whether use batch normal in hidden layer
        clamp_magtitude (float): Symmetric clip value.
    """
    def __init__(self, input_size, hidden_size, hid_act='relu', use_bn=True, clamp_magtitude=10.0):
        super(Discriminator, self).__init__()
        if hid_act == 'relu':
            hid_act_class = nn.ReLU
        elif hid_act == 'tanh':
            hid_act_class = nn.Tanh
        else:
            raise NotImplementedError("The hid_act should be 'relu' or 'tanh'.")

        self.clamp_magtitude = clamp_magtitude

        in_size = input_size
        model_list = []
        out_size = input_size
        for _, out_size in enumerate(hidden_size):
            model_list.append(nn.Dense(in_size, out_size))
            if use_bn:
                model_list.append(nn.BatchNorm1d(out_size))
            model_list.append(hid_act_class())
            in_size = out_size

        model_list.append(nn.Dense(out_size, 1))
        self.model = nn.SequentialCell(model_list)

    def construct(self, batch):
        """Discriminator."""
        output = self.model(batch)
        output = output.clip(-1.0 * self.clamp_magtitude, self.clamp_magtitude)
        return output


class GradientWithInput(nn.Cell):
    """GradientWithInput."""
    def __init__(self, discriminator):
        super(GradientWithInput, self).__init__()
        self.reduce_sum = ops.ReduceSum()
        self.discriminator = discriminator

    def construct(self, interpolates):
        """return reduce sum of discriminator"""
        decision_interpolate = self.discriminator(interpolates)
        decision_interpolate = self.reduce_sum(decision_interpolate, 0)
        return decision_interpolate


class GradientPenalty(nn.Cell):
    """GradientPenalty."""
    def __init__(self, grad_penalty_weight, discriminator):
        super(GradientPenalty, self).__init__()
        self.gradient_op = ops.GradOperation()
        self.grad_penalty_weight = grad_penalty_weight
        self.gradient_with_input = GradientWithInput(discriminator)
        self.uniform_real = ops.UniformReal()

    def construct(self, expert_disc_input, policy_dict_input):
        """Compute gradient penalty loss"""
        eps = self.uniform_real((expert_disc_input.shape[0], 1))
        interp_obs = eps * expert_disc_input + (1. - eps) * policy_dict_input
        gradient = self.gradient_op(self.gradient_with_input)(interp_obs)
        gradient_penalty = ((gradient.norm(1, 2) - 1) ** 2).mean() * self.grad_penalty_weight
        return gradient_penalty


class BCEFocalLoss(nn.Cell):
    """BCEFocalLoss."""
    def __init__(self, gamma):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.one = Tensor(1.0, mindspore.float32)
        self.zero = Tensor(0.0, mindspore.float32)

    def construct(self, inputs, targets):
        """Compute BCEFocal loss"""
        targets = targets.squeeze().to(mindspore.int64)
        prob_1 = F.sigmoid(inputs)
        prob_0 = 1. - prob_1
        soft_inputs = F.concat([prob_0, prob_1], -1)
        target_one_hot = F.one_hot(targets, 2, self.one, self.zero, -1)
        weight = F.pow(1.0 - soft_inputs, self.gamma)
        focal = -weight * F.log(soft_inputs)
        loss = (target_one_hot * focal).sum(-1).mean()
        return loss


class DiscriminatorLossCell(nn.Cell):
    """Discriminator loss cell."""
    def __init__(self,
                 expert_replay_buffer,
                 policy_replay_buffer,
                 discriminator,
                 use_grad_pen=True,
                 grad_pen_weight=10.,
                 disc_focal_loss_gamma=0.0):
        super(DiscriminatorLossCell, self).__init__()
        self.expert_replay_buffer = expert_replay_buffer
        self.policy_replay_buffer = policy_replay_buffer

        self.discriminator = discriminator
        self.bce_loss = BCEFocalLoss(disc_focal_loss_gamma)
        self.bce_target = Tensor(np.concatenate([np.ones((expert_replay_buffer.buffer.buffer_sample.batch_size, 1)),
                                                 np.zeros((policy_replay_buffer.buffer_sample.batch_size, 1))],
                                                0), mindspore.float32)
        self.use_grad_pen = use_grad_pen
        self.gradient_penalty = GradientPenalty(grad_pen_weight, discriminator)

    def construct(self):
        """Compute discriminator loss."""
        expert_obs, expert_acts = self.expert_replay_buffer.sample()
        policy_obs, policy_acts, _, _ = self.policy_replay_buffer.sample()

        expert_disc_input = F.concat([expert_obs, expert_acts], 1)
        policy_disc_input = F.concat([policy_obs, policy_acts], 1)
        disc_input = F.concat([expert_disc_input, policy_disc_input], 0)
        disc_logits = self.discriminator(disc_input)
        disc_total_loss = self.bce_loss(disc_logits, self.bce_target)

        if self.use_grad_pen:
            disc_grad_pen_loss = self.gradient_penalty(expert_disc_input, policy_disc_input)
            disc_total_loss += disc_grad_pen_loss
        return disc_total_loss


class PolicyTrainer(nn.Cell):
    """
    Policy trainer.
    """
    def __init__(self, learner, discriminator, policy_replay_buffer, mode):
        super(PolicyTrainer, self).__init__()
        assert mode in ['gail', 'gail2']
        self.learner = learner
        self.discriminator = discriminator
        self.policy_replay_buffer = policy_replay_buffer
        self.softplus = ops.Softplus()
        self.mode = mode

    def construct(self):
        """Update policy used in generator."""
        obs, action, next_obs, done = self.policy_replay_buffer.sample()
        disc_inputs = F.concat([obs, action], 1)
        disc_logits = self.discriminator(disc_inputs)
        if self.mode == 'gail':
            reward = self.softplus(disc_logits)
        else: # For gail2. Graph compiler requires 'else' branch.
            reward = -self.softplus(-disc_logits)
        loss = self.learner.learn((obs, action, reward, next_obs, done))
        return loss


class GAILLearner(Learner):
    """This is the learner class of GAIL algorithm, which is used to update the policy net"""
    def __init__(self,
                 policy_learner,
                 discriminator,
                 discriminator_trainer,
                 expert_replay_buffer,
                 policy_replay_buffer,
                 num_update_loops_per_train_call,
                 num_disc_updates_per_loop_iter,
                 num_policy_updates_per_loop_iter,
                 mode):
        super(GAILLearner, self).__init__()
        self.policy_trainer = PolicyTrainer(policy_learner, discriminator, policy_replay_buffer, mode)
        self.discriminator = discriminator
        self.discriminator_trainer = discriminator_trainer
        self.expert_replay_buffer = expert_replay_buffer
        self.policy_replay_buffer = policy_replay_buffer

        self.num_update_loops_per_train_call = Tensor(num_update_loops_per_train_call, mindspore.int32)
        self.num_disc_updates_per_loop_iter = Tensor(num_disc_updates_per_loop_iter, mindspore.int32)
        self.num_policy_updates_per_loop_iter = Tensor(num_policy_updates_per_loop_iter, mindspore.int32)

    def learn(self):     # pylint: disable=W0221
        """learn"""
        disc_loss = Tensor(0.)
        policy_loss = Tensor(0.)
        # Loop unrolling(for-range) and higher-order derivative results compiling fail.
        # Use while-Tensor style avoid the issue.
        i = Tensor(0)
        while i < self.num_update_loops_per_train_call:
            i += 1
            j = Tensor(0)
            while j < self.num_disc_updates_per_loop_iter:
                disc_loss = self.discriminator_trainer()
                j += 1

            k = Tensor(0)
            while k < self.num_policy_updates_per_loop_iter:
                policy_loss = self.policy_trainer()
                k += 1
        return disc_loss, policy_loss
