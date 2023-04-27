# Copyright 2021 Huawei Technologies Co., Ltd
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
Base Callback and some default callbacks.
"""
import os
import time
from contextlib import ExitStack
import numpy as np

from mindspore import log as logger
from mindspore import Tensor
from mindspore.train.serialization import save_checkpoint
from mindspore.train._utils import _make_directory

class Callback:
    '''
    Base callback.
    '''
    def __enter__(self):
        '''Return self.'''
        return self

    def __exit__(self, *err):
        "Exit and release."

    def begin(self, params):
        '''
        Call once before train.

        Args:
            params (CallbackParam): Parameters for begin.
        '''

    def end(self, params):
        '''
        Call once after train.

        Args:
            params (CallbackParam): Parameters for end.
        '''

    def episode_begin(self, params):
        '''
        Call before each episode begin.

        Args:
            params (CallbackParam): Parameters for episode begin.
        '''

    def episode_end(self, params):
        '''
        Call after each episode finished.

        Args:
            params (CallbackParam): Parameters for episode end.
        '''


class CallbackParam(dict):
    '''It contains the parameters required for the execution of the callback function.'''
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value


class CallbackManager(Callback):
    '''Execute callbacks sequentially

        Args:
            callbacks (list[Callback]): a list of Callbacks.
    '''
    def __init__(self, callbacks):
        self._callbacks, self._stack = [], None
        # 1 single callback
        if isinstance(callbacks, Callback):
            self._callbacks.append(callbacks)
        # 2 callbacks list
        elif isinstance(callbacks, list):
            for cb in callbacks:
                if not isinstance(cb, Callback):
                    raise TypeError("Callback in callbacks must be Callback Function.")
                self._callbacks.append(cb)
        elif callbacks is not None:
            raise TypeError("The callbacks must be Callback.")

    def __enter__(self):
        if self._stack is None:
            callbacks, self._stack = [], ExitStack().__enter__()
            for callback in self._callbacks:
                target = self._stack.enter_context(callback)
                if not isinstance(target, Callback):
                    logger.warning("Error callback type.")
                    callbacks.append(callback)
                else:
                    callbacks.append(target)
            self._callbacks = callbacks
        return self

    def __exit__(self, *err):
        return self._stack.__exit__(*err)

    def begin(self, params):
        '''
        Call only once before training.

        Args:
            params (CallbackParam): Parameters for begin.
        '''
        for cb in self._callbacks:
            cb.begin(params)

    def end(self, params):
        '''
        Call only once after training.

        Args:
            params (CallbackParam): Parameters for end.
        '''
        for cb in self._callbacks:
            cb.end(params)

    def episode_begin(self, params):
        '''
        Call before each episode start

        Args:
            params (CallbackParam): Parameters for episode begin.
        '''
        for cb in self._callbacks:
            cb.episode_begin(params)

    def episode_end(self, params):
        '''
        Call before each episode end.

        Args:
            params (CallbackParam): Parameters for episode end.
        '''
        for cb in self._callbacks:
            cb.episode_end(params)


def _is_tensor(input_):
    '''Inner util func to check input type.'''
    if isinstance(input_, Tensor) or isinstance(input_.asnumpy(), np.ndarray):
        return True
    return False


class LossCallback(Callback):
    r'''
    Print loss in each episode end.

    Args:
        print_rate (int, optional): The frequency to print loss. Default: ``1`` .

    Examples:
        >>> from mindspore_rl.utils.callback import LossCallback
        >>> from mindspore_rl.core import Session
        >>> from mindspore_rl.algorithm.dqn import config
        >>> loss_cb = LossCallback()
        >>> cbs = [loss_cb]
        >>> session = Session(config.algorithm_config, None, None, cbs)
    '''
    def __init__(self, print_rate=1):
        super(LossCallback, self).__init__()
        if not isinstance(print_rate, int) or print_rate < 0:
            raise ValueError("The arg of 'print_rate' must be int and >= 0, but get ", print_rate)
        self._print_rate = print_rate

    def episode_end(self, params):
        '''
        Print loss in the end of episode.

        Args:
            params (CallbackParam): Parameters of the tarining.
        '''
        losses = params.loss
        rewards = params.total_rewards
        rewards_out, losses_out = [], []

        if not (self._print_rate != 0 and params.cur_episode % self._print_rate == 0):
            return

        # 1 Deal with rewards for both tuple and single.
        if isinstance(rewards, (tuple, list)):
            for reward in rewards:
                if _is_tensor(reward):
                    rewards_out.append(round(float(np.mean(reward.asnumpy())), 3))
        if _is_tensor(rewards):
            rewards_out.append(round(float(np.mean(rewards.asnumpy())), 3))

        # 2 Deal with losses for both tuple and single.
        if isinstance(losses, (tuple, list)):
            for loss in losses:
                if _is_tensor(loss):
                    losses_out.append(round(float(np.mean(loss.asnumpy())), 3))
        elif _is_tensor(losses):
            losses_out.append(round(float(np.mean(losses.asnumpy())), 3))
        else:
            raise ValueError("Episode {}: losses should be tensor or tensor of list/tuple.".format(params.cur_episode))

        # 3 Check loss value and stop if it is NAN or INF.
        for loss in losses_out:
            if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
                raise ValueError("Episode {}: Invalid loss {}, training stop.".format(params.cur_episode, loss))

        # 4 Pirnt the loss and rewards.
        print("Episode {}: loss is {}, rewards is {}".format(params.cur_episode, \
            ', '.join([str(x) for x in losses_out]), ', '.join([str(x) for x in rewards_out])), flush=True)


class TimeCallback(Callback):
    r'''
    Time Callback to monitor time costs for each episode.

    Args:
        print_rate (int, optional): The frequency to print time. Default: ``1`` .
        fixed_steps_in_episode (int, optional): If the number of steps in each episode is fixed, this number
            is used to calculate the step time. If ``None`` , the real steps number should be provided in params.
            Default: ``None`` .

    Examples:
        >>> from mindspore_rl.utils.callback import TimeCallback
        >>> from mindspore_rl.core import Session
        >>> from mindspore_rl.algorithm.dqn import config
        >>> time_cb = TimeCallback()
        >>> cbs = [time_cb]
        >>> session = Session(config.algorithm_config, None, None, cbs)
    '''
    def __init__(self, print_rate=1, fixed_steps_in_episode=None):
        super(TimeCallback, self).__init__()
        if not isinstance(print_rate, int) or print_rate < 0:
            raise ValueError("The arg of 'print_rate' must be int and >= 0, but get ", print_rate)
        if fixed_steps_in_episode is not None:
            if not (isinstance(fixed_steps_in_episode, int) and fixed_steps_in_episode > 0):
                raise ValueError("The arg of 'fixed_steps_in_episode' must be int and > 0, but get ", \
                    fixed_steps_in_episode)

        self._print_rate = print_rate
        self._steps_in_episode = fixed_steps_in_episode
        self.epoch_time = time.time()

    def episode_begin(self, params):
        '''
        Get time in the begin of episode.

        Args:
            params (CallbackParam): Parameters of the tarining.
        '''
        self.epoch_time = time.time()

    def episode_end(self, params):
        '''
        Print time in the end of episode.

        Args:
            params (CallbackParam): Parameters of the tarining.
        '''
        epoch_secends = (time.time() - self.epoch_time) * 1000
        steps = self._steps_in_episode

        if steps is None:
            steps = params.steps
            if isinstance(steps, (tuple, list)):
                if isinstance(steps[0], Tensor) or isinstance(steps[0].asnumpy(), np.ndarray):
                    steps = steps[0].asnumpy()
            if isinstance(steps, Tensor) and isinstance(steps.asnumpy(), np.ndarray):
                steps = steps.asnumpy()

        step_seconds = epoch_secends / steps
        if self._print_rate != 0 and params.cur_episode % self._print_rate == 0:
            print("Episode {} has {} steps, cost time: {:5.3f} ms, per step time: {:5.3f} ms" \
                .format(params.cur_episode, steps, epoch_secends, step_seconds), flush=True)


class CheckpointCallback(Callback):
    r'''
    Save the checkpoint file for all the model weights. And keep the latest `max_ckpt_nums` checkpoint files.

    Args:
        save_per_episode (int, optional): The frequency to save checkpoint. Default: ``0`` (not saved).
        directory (str, optional): The directory for saving checkpoints. Default: ``None`` , saving to ``'./'`` .
        max_ckpt_nums (int, optional): Numbers of how many checkpoint files to be kept. Default: ``5`` .

    Examples:
        >>> from mindspore_rl.utils.callback import CheckpointCallback
        >>> from mindspore_rl.core import Session
        >>> from mindspore_rl.algorithm.dqn import config
        >>> ckpt_cb = CheckpointCallback()
        >>> cbs = [ckpt_cb]
        >>> session = Session(config.algorithm_config, None, None, cbs)
    '''
    def __init__(self, save_per_episode=0, directory=None, max_ckpt_nums=5):
        super(CheckpointCallback, self).__init__()
        if not isinstance(save_per_episode, int) or save_per_episode < 0:
            raise ValueError("The arg of 'save_per_episode' must be int and >= 0, but get ", save_per_episode)
        if not isinstance(max_ckpt_nums, int) or max_ckpt_nums < 0:
            raise ValueError("The arg of 'max_ckpt_nums' must be int and > 0, but get ", max_ckpt_nums)

        self._max_ckpt_nums = max_ckpt_nums
        self._save_per_episode = save_per_episode

        # Make directory if provided or get current directory.
        if directory is not None:
            self._save_path = _make_directory(directory)
        else:
            self._save_path = os.getcwd()
        if not os.path.exists(self._save_path):
            os.makedirs(self._save_path)

    def episode_end(self, params):
        '''
        Save checkpoint in the end of episode.

        Args:
            params (CallbackParam): Parameters of the tarining.
        '''
        if not (self._save_per_episode != 0 and (params.cur_episode + 1) % self._save_per_episode == 0):
            return
        if not params.vars:
            raise RuntimeError("The trainable_variables not found.")
        # 1 `(key, var)` in `params.vars` each represent as name and variables for saving.
        for key, var in params.vars.items():
            # 2 `save_path` combined by `self._save_path` and name. Eg: /ckpt/actor_net/ .
            save_path = self._save_path + '/' + key
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # 3 Save checkpoints for each variables. Eg: /path/actor_net_10.ckpt.
            ckpt_name = save_path + '/' + key + '_' + str(params.cur_episode + 1) + '.ckpt'
            save_checkpoint(var, ckpt_name)

            files = os.listdir(save_path)
            ckpt_list = []
            nums = 0

            # 4 Keep `_max_ckpt_nums` checkpoint files, and delete the older one.
            for filename in files:
                if os.path.splitext(filename)[-1] == '.ckpt' and (key in filename):
                    nums += 1
                    ckpt_list.append(save_path + "/" + filename)
                if nums > self._max_ckpt_nums:
                    ckpt_files = sorted(ckpt_list, key=os.path.getmtime)
                    os.remove(ckpt_files[0])
                    ckpt_list.remove(ckpt_files[0])


class EvaluateCallback(Callback):
    r'''
    Evaluate callback.

    Args:
        eval_rate (int, optional): The frequency to eval. Default: ``0`` (will not evaluate).

    Examples:
        >>> from mindspore_rl.utils.callback import EvaluateCallback
        >>> from mindspore_rl.core import Session
        >>> from mindspore_rl.algorithm.dqn import config
        >>> eval_cb = EvaluateCallback()
        >>> cbs = [eval_cb]
        >>> session = Session(config.algorithm_config, None, None, cbs)
    '''
    def __init__(self, eval_rate=0):
        super(EvaluateCallback, self).__init__()
        if not isinstance(eval_rate, int) or eval_rate < 0:
            raise ValueError("The arg of 'eval_rate' must be int and >= 0, but get ", eval_rate)
        self._eval_rate = eval_rate

    def begin(self, params):
        '''
        Store the eval rate in the begin of training, run once.

        Args:
            params (CallbackParam): Parameters for episode begin.
        '''
        params.eval_rate = self._eval_rate

    def episode_end(self, params):
        '''
        Run evaluate in the end of episode, and print the rewards.

        Args:
            params (CallbackParam): Parameters for episode end.
        '''
        if self._eval_rate != 0 and params.cur_episode > 0 and \
            params.cur_episode % self._eval_rate == 0:
            # Call the `evaluate` function provided by user.
            rewards = params.evaluate()
            print("-----------------------------------------")
            print("Evaluate for episode {} total rewards is {:5.3f}" \
                .format(params.cur_episode, rewards.asnumpy()), flush=True)
            print("-----------------------------------------")
