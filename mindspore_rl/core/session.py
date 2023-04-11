# Copyright 2021-2023 Huawei Technologies Co., Ltd
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
Implementation of the session class.
"""
from mindspore.communication import get_rank, init

from mindspore_rl.core import MSRL
from mindspore_rl.distribution import fragment_generation


class _Workers:
    r"""
    The _Workers class is class for the distributed algorithms.

    Args:
        msrl (MSRL): The MSRL instance.
        fragment_list (dict): All the fragmets for distribution.
        episode (int): The eposide for each training.
    """

    def __init__(self, msrl, fragment_list, duration, episode):
        self.rank_id = get_rank()
        self.fid = str(self.rank_id)
        print("Assign fragment ", self.fid, " on worker ", self.rank_id)
        self.worker = fragment_list[self.rank_id](msrl, self.rank_id, duration, episode)

    def run(self):
        print("Start fragment ", self.fid, " on worker ", self.rank_id)
        self.worker.run()
        print("Finish fragment ", self.fid)


class Session:
    """
    The Session is a class for running MindSpore RL algorithms.

    Args:
        alg_config (dict): the algorithm configuration or the deployment configuration of the algorithm.
        deploy_config (dict): the deployment configuration for distribution. Default: None.
            For more details of configuration of algorithm, please have a look at
            `detail <https://www.mindspore.cn/reinforcement/docs/zh-CN/master/custom_config_info.html>`_.
        params (dict): The algorithm specific training parameters. Default: None.
        callbacks (list[Callback]): The callback list. Default: None.
    """

    def __init__(self, alg_config, deploy_config=None, params=None, callbacks=None):
        if alg_config is not None:
            self.msrl = MSRL(alg_config, deploy_config)
        self.params = params
        self.callbacks = callbacks
        self.dist = False
        self.alg_config = alg_config
        if deploy_config:
            self.dist = True
            self.worker_num = deploy_config["worker_num"]
            self.config = deploy_config["config"]
            self.dist_policy = deploy_config["distribution_policy"]
            self.is_auto = deploy_config["auto_distribution"]
            self.algo_name = deploy_config["algo_name"]

    def run(self, class_type=None, is_train=True, episode=0, duration=0):
        """
        Execute the reinforcement learning algorithm.

        Args:
            class_type (Trainer): The class type of the algorithm"s trainer class. Default: None.
            is_train (bool): Run the algorithm in train mode or eval mode. Default: True
            episode (int): The number of episode of the training. Default: 0.
            duration (int): The number of duration of the training. Default: 0.
        """

        if self.dist:
            init("nccl")
            if self.is_auto:
                fragment_list = fragment_generation(
                    self.algo_name, self.worker_num, self.dist_policy, self.msrl
                )
            else:
                from fragments import get_all_fragments  # pylint: disable=C0415

                fragment_list = get_all_fragments(self.msrl.num_actors)
            workers = _Workers(self.msrl, fragment_list, duration, episode)
            workers.run()
        else:
            if self.params is None:
                trainer = class_type(self.msrl)
            else:
                trainer = class_type(self.msrl, self.params)
            ckpt_path = None
            if self.params and "ckpt_path" in self.params:
                ckpt_path = self.params["ckpt_path"]
            if is_train:
                trainer.train(episode, self.callbacks, ckpt_path)
                print("training end")
            else:
                if ckpt_path:
                    trainer.load_and_eval(ckpt_path)
                    print("eval end")
                else:
                    print("Please provide a ckpt_path for eval.")

        # Close the environment to release the resource
        if self.msrl.collect_environment is not None:
            self.msrl.collect_environment.close()
        if self.msrl.eval_environment is not None:
            self.msrl.eval_environment.close()
