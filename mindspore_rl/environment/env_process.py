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
The class uses the python process to manage and interact with environments.
"""

#pylint: disable=W0212
from multiprocessing import Process
import numpy as np


class EnvironmentProcess(Process):
    r"""
    An independent process responsible for creating and interacting with one or more environments.

    Args:
        proc_no (int): The process number assigned by the caller.
        env_num (int): The number of input environments.
        envs (list(Environment)): A list that contains instance of environment (subclass of Environment).
        actions (Queue): The queue used to pass actions to the environment process.
        observations (Queue): The queue used to pass observations to the caller process.
        initial_states (Queue): The queue used to pass initial states to the caller process.

    Examples:
        >>> from multiprocessing import Queue
        >>> actions = Queue()
        >>> observations = Queue()
        >>> initial_states = Queue()
        >>> proc_no = 1
        >>> env_num = 2
        >>> env_params = {'name': 'CartPole-v0'}
        >>> multi_env = [GymEnvironment(env_params), GymEnvironment(env_params)]
        >>> env_proc = EnvironmentProcess(proc_no, env_num, multi_env, actions, observations, initial_states)
        >>> env_proc.start()
    """

    def __init__(self, proc_no, env_num, envs,
                 actions, observations, initial_states):
        super().__init__()
        self.proc_no = proc_no
        self.actions = actions
        self.observations = observations
        self.initial_states = initial_states

        self.env_num = env_num
        self.envs = envs

    def run(self):
        while True:
            message = self.actions.get()
            if isinstance(message, np.ndarray):
                obs = [self.envs[i]._step(message[i])
                       for i in range(self.env_num)]
                self.observations.put(obs)
            elif message == 'reset':
                init_states = [self.envs[i]._reset()
                               for i in range(self.env_num)]
                self.initial_states.put(init_states)
