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
Implementation of trainer base class.
"""
import os

from mindspore_rl.utils.callback import CallbackParam, CallbackManager, TimeCallback
import mindspore.nn as nn
from mindspore.train.serialization import load_checkpoint, load_param_into_net


INIT = 1
COLLECT = 2
EVAL = 3


class Trainer(nn.Cell):
    r"""
    The trainer base class. It is a process class that provides the basic mode of training.

    Note:
        Reference to `dqn_trainer.py
        <https://gitee.com/mindspore/docs/blob/master/docs/reinforcement/docs/source_en/dqn.md
        #defining-the-dqntrainer-class>`_.

    Args:
        msrl(MSRL): the function handler class.
    """

    def __init__(self, msrl):
        super(Trainer, self).__init__()
        self.msrl = msrl
        self.vars = {}

    def train(self, episodes, callbacks=None, ckpt_path=None):
        """
        The train method provides a standard training process, including the whole loop and callbacks.
        Users can inherit or overwrite as needed.

        Args:
            episodes(int): the number of training episodes.
            callbacks(Optional[list[Callback]]): List of callback objects. Default: None
            ckpt_path(Optional[str]): The checkpoint file path to init or restore net. Default: None.
        """

        cb_params = CallbackParam()
        cb_params.episodes_num = episodes

        # Move TimeCallback to the first to exclude the time of other callbacks.
        for item in callbacks:
            if isinstance(item, TimeCallback):
                callbacks.remove(item)
                callbacks.insert(0, item)
        # 1 Using `CallbackManager` to traverse each callback.
        with CallbackManager(callbacks) as callback_list:

            # 2 Init or restore the variables if the checkpoint files exist.
            self._init_or_restore(ckpt_path)
            cb_params.cur_episode = 0
            if self.vars:
                cb_params.vars = self.vars

            callback_list.begin(cb_params)

            # 3 Get `evaluate` function if meet the conditions.
            if 'eval_rate' in cb_params and cb_params.eval_rate > 0:
                cb_params.evaluate = self.evaluate

            for i in range(episodes):
                callback_list.episode_begin(cb_params)

                # 4 Get the result of `train_one_episode` func, and deal with three situation:
                #   a) Default using: Three objects in tuple, each stand for `loss`, `rewards` and `steps`.
                #   b) User defined: Four objects in tuple, the first three is same as default using, the last
                #       one `others` can be tuple or single one as user defined.
                #   c) Other situation: Runtime error.
                ans = self.train_one_episode()
                loss, rewards, steps, others = [], [], [], []
                if len(ans) == 3:
                    loss, rewards, steps = ans
                elif len(ans) == 4:
                    loss, rewards, steps, others = ans
                else:
                    raise RuntimeError("The output number of function `train_one_episode` must be 3 or 4, \
                        and represent for `loss, rewards, steps, [optional]others.` in order")

                cb_params.loss = loss
                cb_params.total_rewards = rewards
                cb_params.steps = steps
                cb_params.others = others
                callback_list.episode_end(cb_params)
                cb_params.cur_episode = i + 1
            callback_list.end(cb_params)

    def train_one_episode(self):
        """
        The interface of train one episode function in train.
        And the output of this function must be constricted as `loss, rewards, steps, [optional]others` in order.
        """
        raise NotImplementedError("Method train_one_episode should be overridden by subclass.\
            and the output must be constricted as `loss, rewards, steps, [optional]others` in order")

    def evaluate(self):
        """
        The interface of the evaluate function for evaluate in train.
        """
        raise NotImplementedError("Method evaluate should be overridden by subclass.")

    def _load_ckpt(self, ckpt_path=None, name=None, net=None):
        """load checkpoint"""

        # 1 Deal with the input file or input path.
        if os.path.exists(ckpt_path) and os.path.isfile(ckpt_path):
            ckpt_file = ckpt_path
        else:
            dir_path = ckpt_path + '/' + name
            # 2 If ckpt file does not exist, just return. Otherwise, get the newest one.
            if not os.path.exists(dir_path):
                return
            files = os.listdir(dir_path)
            if files == []:
                return
            ckpt_files = []
            for filename in files:
                if os.path.splitext(filename)[-1] == '.ckpt' and (name in filename):
                    ckpt_files.append(dir_path + '/' + filename)
            ckpt_file = sorted(ckpt_files, key=os.path.getmtime)[-1]

        # 3 Load the checkpoint.
        if os.path.exists(ckpt_file):
            print("Load file ", ckpt_file)
            param_dict = load_checkpoint(ckpt_file)
            _, not_load = load_param_into_net(net, param_dict)
            if not_load:
                raise RuntimeError("Load params into net failed!")
        else:
            print("Warning: missing ckpt file for ", name)

    def _init_or_restore(self, ckpt_path=None):
        '''Init or restore the variables.'''
        if ckpt_path:
            self.vars = self.trainable_variables()
            for key, val in self.vars.items():
                self._load_ckpt(ckpt_path, key, val)


    def trainable_variables(self):
        """
        The variables for saving to checkpoint.
        """
        raise NotImplementedError("Method trainable_variables should be overridden by subclass.")

    def load_and_eval(self, ckpt_path=None):
        """
        The interface of the eval function for offline. A checkpoint must be provided.

        Args:
            ckpt_path (string): The checkpoint file to restore net.
        """
        if ckpt_path is None:
            raise RuntimeError("Please provide a ckpt_path.")
        self._init_or_restore(ckpt_path)
        reward = self.evaluate()
        reward = reward.asnumpy()
        print("-----------------------------------------")
        print(f"Evaluate result is {reward:.3f}, checkpoint file in {ckpt_path}")
        print("-----------------------------------------")
