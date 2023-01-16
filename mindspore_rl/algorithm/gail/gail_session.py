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
"""
GAIL session.
"""
import pickle
import random
import mindspore
from mindspore import nn
from mindspore_rl.core import Session
from mindspore_rl.utils.callback import CheckpointCallback, LossCallback, EvaluateCallback, TimeCallback
from mindspore_rl.core.uniform_replay_buffer import UniformReplayBuffer
from mindspore_rl.algorithm.gail.gail import ExpertReplayBuffer, Discriminator, DiscriminatorLossCell, GAILLearner


class GAILSession(Session):
    '''GAIL session'''
    def __init__(self,
                 collect_environment,
                 eval_environment,
                 expert_data_path,
                 expert_sample_size,
                 expert_batch_size,
                 policy_batch_size,
                 policy_buffer_size,
                 policy,
                 policy_trainer,
                 disc_config,
                 gail_learn_config):
        with open(expert_data_path, 'rb') as f:
            traj_list = pickle.load(f)
            traj_list = random.sample(traj_list, expert_sample_size)

        obs_space = collect_environment.observation_space
        action_space = collect_environment.action_space
        expert_replay_buffer = ExpertReplayBuffer(traj_list=traj_list,
                                                  batch_size=expert_batch_size,
                                                  shapes=[obs_space.shape, action_space.shape],
                                                  dtypes=[obs_space.ms_dtype, action_space.ms_dtype])

        policy_replay_buffer = UniformReplayBuffer(batch_size=policy_batch_size,
                                                   capacity=policy_buffer_size,
                                                   shapes=[obs_space.shape, action_space.shape, obs_space.shape, (1,)],
                                                   types=[obs_space.ms_dtype, action_space.ms_dtype,
                                                          obs_space.ms_dtype, mindspore.bool_])


        discriminator = Discriminator(input_size=obs_space.shape[0] + action_space.shape[0],
                                      hidden_size=disc_config['disc_hid_dim'],
                                      hid_act=disc_config['disc_hid_act'],
                                      use_bn=disc_config['disc_use_bn'],
                                      clamp_magtitude=disc_config['disc_clamp_magnitude'])
        disc_optim = nn.Adam(discriminator.trainable_params(), learning_rate=disc_config['disc_lr'])
        disc_loss = DiscriminatorLossCell(expert_replay_buffer,
                                          policy_replay_buffer,
                                          discriminator,
                                          disc_config['use_grad_pen'],
                                          disc_config['grad_pen_weight'],
                                          disc_config['disc_focal_loss_gamma'])
        discriminator_trainer = nn.TrainOneStepCell(disc_loss, disc_optim)

        learner = GAILLearner(policy_trainer,
                              discriminator,
                              discriminator_trainer,
                              expert_replay_buffer,
                              policy_replay_buffer,
                              gail_learn_config['num_update_loops_per_train_call'],
                              gail_learn_config['num_disc_updates_per_loop_iter'],
                              gail_learn_config['num_policy_updates_per_loop_iter'],
                              gail_learn_config['mode'])

        self.msrl = nn.Cell()
        self.msrl.collect_environment = collect_environment
        self.msrl.eval_environment = eval_environment
        self.msrl.policy_replay_buffer = policy_replay_buffer
        self.msrl.collect_policy = policy.collect_policy
        self.msrl.eval_policy = policy.eval_policy
        self.msrl.learner = learner

        loss_cb = LossCallback()
        ckpt_cb = CheckpointCallback(save_per_episode=10, directory='./ckpt')
        eval_cb = EvaluateCallback(eval_rate=10)
        time_cb = TimeCallback()
        cbs = [loss_cb, ckpt_cb, eval_cb, time_cb]

        trainer_params = {
            'save_per_episode': 100,
            'ckpt_path': './ckpt',
            'num_eval_episode': 10,
            'min_step_before_training': 5000,
        }
        super().__init__(alg_config=None, deploy_config=None, params=trainer_params, callbacks=cbs)
