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
"""MADDPG Trainer"""
#pylint: disable=W0613
import time
import numpy as np
import mindspore as ms
from mindspore.common.api import jit
from mindspore import Tensor, Parameter
from mindspore.ops import operations as P

from mindspore_rl.utils.callback import CallbackParam, CallbackManager
from mindspore_rl.agent.trainer import Trainer



class MADDPGTrainer(Trainer):
    """This is the trainer class of MADDPG algorithm. It arranges the MADDPG algorithm"""

    def __init__(self, msrl, params):
        super(MADDPGTrainer, self).__init__(msrl)
        self.msrl = msrl
        self.agent = self.msrl.agent
        self.params = params
        self.zero = Tensor(0, ms.int32)
        self.zero_float = Tensor(0.0, ms.float32)
        self.false = Tensor(False, ms.bool_)
        self.true = Tensor(True, ms.bool_)
        self.duration_config = Tensor(25, ms.int32)
        self.assign = P.Assign()
        self.squeeze = P.Squeeze()
        self.less = P.Less()
        self.reduce_mean = P.ReduceMean()
        self.stack_first = P.Stack(axis=0)
        self.cat = P.Concat(axis=1)
        self.continuous_actions = params.get('continuous_actions')
        self.num_agent = params.get('num_agent')
        self.num_eval_episode = params.get('num_eval_episode')
        self.fill_value = Tensor(params.get('init_size'), ms.int32)
        self.episode_rewards = Parameter(Tensor([0.], ms.float32), requires_grad=False)
        self.train_step = Parameter(Tensor(0, ms.int32), requires_grad=False)

    def train(self, episodes, callbacks=None, ckpt_path=None):
        """
        The interface of the train function. User will implement
        this function.

        Args:
            episodes(int): the number of training episodes.
            callbacks(Optional[list[Callback]]): List of callback objects. Default: None
            ckpt_path(Optional[string]): The checkpoint file to init or restore net. Default: None.
        """

        cb_params = CallbackParam()
        cb_params.episodes_num = episodes
        self.init()
        # 1 Using `CallbackManager` to traverse each callback.
        with CallbackManager(callbacks) as callback_list:

            # 2 Init or restore the variables if the checkpoint files exist.
            cb_params.cur_episode = 0
            if self.vars:
                cb_params.vars = self.vars

            callback_list.begin(cb_params)

            # 3 Get `evaluate` function if meet the conditions.
            if 'eval_rate' in cb_params and cb_params.eval_rate > 0:
                cb_params.evaluate = self.evaluate
            episode_rewards = []
            start = time.time()
            for i in range(episodes):
                callback_list.episode_begin(cb_params)
                ans = self.train_one_episode()
                loss, rewards, steps, others = [], [], [], []
                if len(ans) == 3:
                    loss, rewards, steps = ans
                elif len(ans) == 4:
                    loss, rewards, steps, others = ans
                else:
                    raise RuntimeError("The output number of function `train_one_episode` must be 3 or 4, \
                        and represent for `loss, rewards, steps, [optional]others.` in order")
                episode_rewards.append(float(rewards.asnumpy()))
                if i % 1000 == 0:
                    print("-----------------------------------------")
                    print("In episode {}, mean episode reward is {} , cost {} s.".format(\
                        i, np.mean(episode_rewards[-1000:]), round(time.time()-start, 3)))
                    start = time.time()
                cb_params.loss = loss
                cb_params.total_rewards = rewards
                cb_params.steps = steps
                cb_params.others = others
                callback_list.episode_end(cb_params)
                cb_params.cur_episode = i + 1

            callback_list.end(cb_params)

    @jit
    def init(self):
        """Init method, it will be called once"""
        # ---------------------- initialize ------------------------- #
        obs_n = self.msrl.collect_environment.reset()
        i = self.zero
        done = self.false
        while self.less(i, self.fill_value):
            act_n = []
            agent_id = 0
            while agent_id < self.num_agent:
                act_n.append(self.agent[agent_id].actor.sample_action(self.continuous_actions,\
                     obs_n[agent_id, :].squeeze()))
                agent_id += 1
            action_list = self.stack_first(act_n)
            # step for each agent's env
            new_obs_n, rew_n, done_n = self.msrl.collect_environment.step(action_list)

            self.msrl.replay_buffer_insert([obs_n, action_list, rew_n, new_obs_n, done_n])
            obs_n = new_obs_n
            done = done_n.all()
            if done:
                obs_n = self.msrl.collect_environment.reset()
                done = self.false
            i += 1
        return self.true

    @jit
    def train_one_episode(self):
        """the algorithm in one episode"""
        # ----------------------------------------- actor -------------------------------------------
        obs_n = self.msrl.collect_environment.reset()

        training_reward = self.zero_float
        duration = self.zero
        loss = self.zero_float
        while self.less(duration, 25):
            self.train_step += 1
            act_n = []
            agent_id = 0
            while agent_id < self.num_agent:
                act_n.append(self.agent[agent_id].actor.sample_action(self.continuous_actions,\
                     obs_n[agent_id, :].squeeze()))
                agent_id += 1
            action_list = self.stack_first(act_n)
            # step for each agent's env
            new_obs_n, rew_n, done_n = self.msrl.collect_environment.step(action_list)

            dones = done_n.all()

            self.msrl.replay_buffer_insert([obs_n, action_list, rew_n, new_obs_n, done_n])

            obs_n = new_obs_n
            duration += 1
            training_reward += rew_n.sum()

            # ----------------------------------------- learner -------------------------------------------
            if self.train_step % 100 == 0:
                obs_n_batch, act_n_batch, rew_n_batch, obs_next_n_batch, done_n_batch \
                    = self.msrl.replay_buffer_sample()
                agent_id = 0
                while agent_id < self.num_agent:
                    loss += self._learn(agent_id, obs_n_batch, act_n_batch, rew_n_batch, \
                        obs_next_n_batch, done_n_batch)
                    agent_id += 1

            if dones:
                break
        return loss, training_reward, duration

    def trainable_variables(self):
        """Trainable variables for saving."""
        return

    @jit
    def evaluate(self):
        eval_episode = self.zero
        eval_episode_rewards = self.zero_float
        while eval_episode < self.num_eval_episode:
            eval_episode += 1
            obs_n = self.msrl.eval_environment.reset()
            duration = self.zero
            training_reward = self.zero_float
            while self.less(duration, 25):
                act_n = []
                agent_id = 0
                while agent_id < self.num_agent:
                    act_n.append(self.agent[agent_id].actor.get_action(self.continuous_actions, obs_n[agent_id, :]))
                    agent_id += 1
                action_list = self.stack_first(act_n)
                obs_n, rew_n, done_n = self.msrl.eval_environment.step(action_list)
                dones = done_n.all()
                training_reward += rew_n.sum()
                duration += 1
                if dones:
                    break
            eval_episode_rewards += training_reward
        return eval_episode_rewards / self.num_eval_episode

    def _learn(self, agent_id, obs_n, act_n, rew_n, obs_next_n, done_n):
        '''learn func for each agent'''
        rew, done = rew_n[:, agent_id, :], done_n[:, agent_id, :]
        # comput target_act_next_n
        target_act_next_n = []
        i = 0
        while i < self.num_agent:
            target_act_next_n.append(self.agent[i].actor.sample_action(self.continuous_actions,\
                 obs_next_n[:, i, :], use_target=True))
            i += 1
        target_act_next = self.cat(target_act_next_n)
        loss = self.agent[agent_id].learner.learn([obs_n, act_n, obs_next_n, rew, done, target_act_next, agent_id])
        return loss
