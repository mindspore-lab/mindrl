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
SAC training example.
"""

#pylint: disable=C0413
import argparse
import math
from mindspore import context
from mindspore import dtype as mstype
from mindspore_rl.environment.gym_environment import GymEnvironment
from mindspore_rl.algorithm.sac.sac import SACPolicy
from mindspore_rl.algorithm.sac.sac import SACLearner
from mindspore_rl.algorithm.gail.gail_session import GAILSession
from mindspore_rl.algorithm.gail.gail_trainer import GAILTrainer


parser = argparse.ArgumentParser(description='MindSpore Reinforcement GAIL')
parser.add_argument('--episode', type=int, default=500, help='total episode numbers.')
parser.add_argument('--device_target', type=str, default='Auto', choices=['Ascend', 'CPU', 'GPU', 'Auto'],
                    help='Choose a device to run the sac example(Default: Auto).')
parser.add_argument('--precision_mode', type=str, default='fp32', choices=['fp32', 'fp16'],
                    help='Precision mode')
parser.add_argument('--expert_data_path', type=str,
                    default='./mujoco-experts/HalfCheetah/seed-0/exp_trajs_sac_50.pkl',
                    help='expert data path.')
args, _ = parser.parse_known_args()


def train(episode=args.episode):
    """start to train GAIL algorithm"""
    if args.device_target != 'Auto':
        context.set_context(device_target=args.device_target)
    if context.get_context('device_target') in ['CPU']:
        context.set_context(enable_graph_kernel=True)

    context.set_context(mode=context.GRAPH_MODE)

    compute_type = mstype.float32 if args.precision_mode == 'fp32' else mstype.float16
    if compute_type == mstype.float16 and args.device_target != 'Ascend':
        raise ValueError("Fp16 mode is supported by Ascend backend.")

    # These objects will be instance by MSRL latter.
    collect_environment = GymEnvironment({'name': 'HalfCheetah-v2', 'seed': 0})
    eval_environment = GymEnvironment({'name': 'HalfCheetah-v2', 'seed': 0})
    obs_space = collect_environment.observation_space
    action_space = collect_environment.action_space

    policy_params = {
        'state_space_dim': obs_space.shape[0],
        'action_space_dim': action_space.shape[0],
        'hidden_sizes': [256, 256],
        'conditioned_std': True,
        'compute_type': compute_type
    }
    sac_policy = SACPolicy(policy_params)

    learner_params = {
        'gamma': 0.99,
        'critic_lr': 3e-4,
        'actor_lr': 3e-4,
        'reward_scale_factor': 2.0,
        'critic_loss_weight': 0.5,
        'actor_loss_weight': 1.0,
        'actor_mean_std_reg': True,
        'actor_mean_reg_weight': 1e-3,
        'actor_std_reg_weight': 1e-3,
        'log_alpha': math.log(2),
        'train_alpha_net': True,
        'alpha_loss_weight': 1.0,
        'alpha_lr': 3e-4,
        'target_entropy': -3.0,
        'update_factor': 0.005,
        'update_interval': 1,
        'actor_net': sac_policy.actor_net,
        'critic_net1': sac_policy.critic_net1,
        'critic_net2': sac_policy.critic_net2,
        'target_critic_net1': sac_policy.target_critic_net1,
        'target_critic_net2': sac_policy.target_critic_net2
    }
    sac_learner = SACLearner(learner_params)

    disc_config = {}
    disc_config['disc_hid_dim'] = [128, 128]
    disc_config['disc_hid_act'] = 'relu'
    disc_config['disc_use_bn'] = False
    disc_config['disc_clamp_magnitude'] = 10.0
    disc_config['disc_lr'] = 3e-4
    disc_config['use_grad_pen'] = True
    disc_config['grad_pen_weight'] = 0.5
    disc_config['disc_focal_loss_gamma'] = 0.0
    disc_config['compute_type'] = compute_type

    gail_learn_config = {}
    gail_learn_config['num_update_loops_per_train_call'] = 1000
    gail_learn_config['num_disc_updates_per_loop_iter'] = 1
    gail_learn_config['num_policy_updates_per_loop_iter'] = 1
    gail_learn_config['mode'] = 'gail2'

    gail_session = GAILSession(collect_environment=collect_environment,
                               eval_environment=eval_environment,
                               expert_data_path=args.expert_data_path,
                               expert_sample_size=4,
                               expert_batch_size=256,
                               policy_batch_size=256,
                               policy_buffer_size=20000,
                               policy=sac_policy,
                               policy_trainer=sac_learner,
                               disc_config=disc_config,
                               gail_learn_config=gail_learn_config)
    gail_session.run(class_type=GAILTrainer, episode=episode)

if __name__ == "__main__":
    train()
