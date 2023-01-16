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
DQN session.
"""
from mindspore_rl.core import Session
from mindspore_rl.utils.utils import update_config
from mindspore_rl.utils.callback import CheckpointCallback, LossCallback, Callback
from mindspore_rl.algorithm.qmix import config


class StepInfoCollectCallback(Callback):
    """Step info call back for collect, provides detail information getting from sc2"""

    def __init__(self, interval):
        self.interval = interval
        self.won_counter = []
        self.dead_allies_counter = []
        self.dead_enemies_counter = []

    def episode_end(self, params):
        """Step info stats during training"""
        battle_won, dead_allies, dead_enemies = params.others
        self.won_counter.append(battle_won.asnumpy()[0])
        self.dead_allies_counter.append(dead_allies.asnumpy()[0])
        self.dead_enemies_counter.append(dead_enemies.asnumpy()[0])
        if (params.cur_episode + 1) % self.interval == 0:
            win_rate = sum(self.won_counter) / self.interval
            avg_dead_allies = sum(self.dead_allies_counter) / self.interval
            avg_dead_enemies = sum(self.dead_enemies_counter) / self.interval
            self.won_counter = []
            self.dead_allies_counter = []
            self.dead_enemies_counter = []

            print("---------------------------------------------------")
            print("The average statical results of these {} episodes during training is ".format(self.interval),
                  flush=True)
            print("Win Rate: {:.3}".format(win_rate), flush=True)
            print("Average Dead Allies: {:.3}".format(avg_dead_allies), flush=True)
            print("Average Dead Enemies: {:.3}".format(avg_dead_enemies), flush=True)
            print("---------------------------------------------------")


class StepInfoEvalCallback(Callback):
    """Step info call back for evaluation, provides detail information getting from sc2"""

    def __init__(self, eval_rate, times):
        super(StepInfoEvalCallback, self).__init__()
        if not isinstance(eval_rate, int) or eval_rate < 0:
            raise ValueError("The arg of 'evaluation_frequency' must be int and >= 0, but get ", eval_rate)
        self._eval_rate = eval_rate
        self.won_counter = []
        self.dead_allies_counter = []
        self.dead_enemies_counter = []
        self.times = times

    def begin(self, params):
        """Store the eval rate in the begin of training, run once."""
        params.eval_rate = self._eval_rate

    def episode_end(self, params):
        """Run evaluate in the end of episode, and print the rewards."""
        if self._eval_rate != 0 and params.cur_episode > 0 and \
                params.cur_episode % self._eval_rate == 0:
            # Call the `evaluate` function provided by user.
            for _ in range(self.times):
                battle_won, dead_allies, dead_enemies = params.evaluate()
                self.won_counter.append(battle_won.asnumpy()[0])
                self.dead_allies_counter.append(dead_allies.asnumpy()[0])
                self.dead_enemies_counter.append(dead_enemies.asnumpy()[0])

            win_rate = sum(self.won_counter) / self.times
            avg_dead_allies = sum(self.dead_allies_counter) / self.times
            avg_dead_enemies = sum(self.dead_enemies_counter) / self.times
            self.won_counter = []
            self.dead_allies_counter = []
            self.dead_enemies_counter = []

            print("---------------------------------------------------")
            print("The average statical results of these {} episodes during evaluation is ".format(self.times),
                  flush=True)
            print("Win Rate: {:.3}".format(win_rate), flush=True)
            print("Average Dead Allies: {:.3}".format(avg_dead_allies), flush=True)
            print("Average Dead Enemies: {:.3}".format(avg_dead_enemies), flush=True)
            print("---------------------------------------------------")


class QMIXSession(Session):
    '''QMIX session'''
    def __init__(self, env_yaml=None, algo_yaml=None):
        update_config(config, env_yaml, algo_yaml)
        env_config = config.algorithm_config.get('collect_environment')
        env = env_config.get('type')(env_config.get('params'))
        num_agent = env.config.get('num_agent')
        epsode_limit = env.config.get('episode_limit')
        global_obs_dim = env.config.get('global_observation_dim')
        action_dim = env.action_space.num_values
        local_obs_shape, local_obs_type = (epsode_limit + 1, num_agent,
                                           (env.observation_space.shape[-1] + num_agent + \
                                            env.action_space.num_values)), env.observation_space.ms_dtype
        global_obs_shape, global_obs_type = (epsode_limit + 1, global_obs_dim), env.observation_space.ms_dtype
        action_shape, action_type = (epsode_limit + 1, num_agent, 1), env.action_space.ms_dtype
        avail_action_shape, avail_action_type = (epsode_limit + 1, num_agent, action_dim), env.action_space.ms_dtype
        reward_shape, reward_type = (epsode_limit + 1, env.reward_space.shape[-1]), env.reward_space.ms_dtype
        done_shape, done_type = (epsode_limit + 1, env.done_space.shape[-1]), env.done_space.ms_dtype
        filled_shape, filled_type = (epsode_limit + 1, env.done_space.shape[-1]), env.action_space.ms_dtype
        hy_shape, hy_type = (epsode_limit + 1, num_agent, config.policy_params.get('hypernet_embed')),\
             env.reward_space.ms_dtype
        replay_buffer_config = config.algorithm_config.get('replay_buffer')
        replay_buffer_config['data_shape'] = [local_obs_shape, global_obs_shape, action_shape,
                                              avail_action_shape, reward_shape, done_shape, filled_shape, hy_shape]
        replay_buffer_config['data_type'] = [local_obs_type, global_obs_type, action_type,
                                             avail_action_type, reward_type, done_type, filled_type, hy_type]

        params = config.trainer_params
        loss_cb = LossCallback()
        step_info_train_cb = StepInfoCollectCallback(100)
        step_info_eval_cb = StepInfoEvalCallback(200, 20)
        ckpt_cb = CheckpointCallback(config.trainer_params.get('save_per_episode'),
                                     config.trainer_params.get('ckpt_path'))
        cbs = [step_info_train_cb, step_info_eval_cb, loss_cb, ckpt_cb]
        super().__init__(config.algorithm_config, None, params=params, callbacks=cbs)
