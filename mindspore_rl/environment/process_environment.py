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
"""Process Environment"""

# pylint:disable=W0703
# pylint:disable=W0106
import traceback
from multiprocessing import Pipe, Process
from typing import Callable, Iterable, Sequence, Union

import cloudpickle
import numpy as np
from mindspore import Tensor
from mindspore import log as logger

from mindspore_rl.environment.environment import Environment
from mindspore_rl.environment.space import Space

EnvCreator = Callable[[], Environment]


class ProcessEnvironment(Environment):
    """
    Run an environment in a separate process.

    Args:
        env_creators (Union[EnvCreator, Sequence[EnvCreator]): The environment constructor.
        env_id (Iterable): A iterable environment id.
    """

    _CALL = "call"
    _CLOSE = "close"
    _EXCEPTION = "exception"
    _RESULT = "result"
    _GETATTR = "getattr"

    def __init__(
        self, env_creators: Union[EnvCreator, Sequence[EnvCreator]], env_id: Iterable
    ):
        super().__init__()
        self._pickled_env_creators = cloudpickle.dumps(env_creators)
        self._pickled_env_id = cloudpickle.dumps(env_id)
        self._num_env_per_worker = len(env_creators)
        self._local_conn = None
        self._process = None

        self._action_space = None
        self._observation_space = None
        self._reward_space = None
        self._done_space = None
        self._config = None
        self._num_agent = None
        self._num_env_reset_out = None
        self._num_env_step_out = None

    @property
    def action_space(self) -> Space:
        """
        Get the action space of the environment.

        Returns:
            action_space(Space): The action space of environment.
        """
        if not self._action_space:
            self._action_space = self.get_attr("action_space")()
        return self._action_space

    @property
    def observation_space(self) -> Space:
        """
        Get the observation space of the environment.

        Returns:
            observation_space(Space): The observation space of environment.
        """
        if not self._observation_space:
            self._observation_space = self.get_attr("observation_space")()
        return self._observation_space

    @property
    def reward_space(self) -> Space:
        """
        Get the reward space of the environment.

        Returns:
            reward_space(Space): The reward space of environment.
        """
        if not self._reward_space:
            self._reward_space = self.get_attr("reward_space")()
        return self._reward_space

    @property
    def done_space(self) -> Space:
        """
        Get the done space of the environment.

        Returns:
            done_space(Space): The done space of environment.
        """
        if not self._done_space:
            self._done_space = self.get_attr("done_space")()
        return self._done_space

    @property
    def config(self) -> dict:
        """
        Get the config of environment.

        Returns:
            config_dict(dict): A dictionary which contains environment's info.
        """
        if not self._config:
            self._config = self.get_attr("config")()
        return self._config

    @property
    def num_agent(self) -> int:
        """
        Number of agents in the environment.

        Returns:
            num_agent (int), Number of agent in the current environment. If the environment is
                single agent, it will return 1. Otherwise, subclass needs to override this property
                to return correct number of agent. Default: 1.
        """
        if not self._num_agent:
            self._num_agent = self.get_attr("num_agent")()
        return self._num_agent

    @property
    def _num_reset_out(self) -> int:
        """
        Inner method, return the number of return value of reset.

        Returns:
            int, The number of return value of reset.
        """
        if not self._num_env_reset_out:
            self._num_env_reset_out = self.get_attr("_num_reset_out")()
        return self._num_env_reset_out

    @property
    def _num_step_out(self) -> int:
        """
        Inner method, return the number of return value of step.

        Returns:
            int, The number of return value of step.
        """
        if not self._num_env_step_out:
            self._num_env_step_out = self.get_attr("_num_step_out")()
        return self._num_env_step_out

    def render(self) -> Union[Tensor, np.ndarray]:
        """
        Generate the image for current frame of environment.

        Returns:
            img (Union[Tensor, np.ndarray]), The image of environment at current frame.
        """
        raise ValueError("ProcessEnvironment does not support render yet.")

    def set_seed(self, seed_value: Union[int, Sequence[int]]) -> bool:
        r"""
        Set seed to control the randomness of environment.

        Args:
            seed_value (int), The value that is used to set

        Returns:
            Success (np.bool\_), Whether successfully set the seed.
        """
        promise = self.call("set_seed", seed_value)
        success = promise()
        return success

    def start(self):
        """
        Create process and pipe, and start current process.

        Returns:
            bool, Whether start processes successfully.
        """
        self._local_conn, worker_conn = Pipe()
        self._process = Process(target=self._worker, args=(worker_conn,))
        self._process.start()
        return True

    def reset(self):
        """
        Reset the environment to the initial state. It is always used at the beginning of each
        episode. It will return the value of initial state or other initial information.

        Returns:
            - state (Union[np.ndarray, Tensor]), A numpy array or Tensor which states for
                the initial state of environment.
            - args (Union[np.ndarray, Tensor], optional), Support arbitrary outputs, but user needs to ensure the
                dtype. This output is optional.
        """
        promise = self.call("reset")
        return promise

    def step(self, action):
        r"""
        Execute the environment step, which means that interact with environment once.

        Args:
            action (Union[Tensor, np.ndarray]): A tensor that contains the action information.

        Returns:
            - state (Union[np.ndarray, Tensor]), The environment state after performing the action.
            - reward (Union[np.ndarray, Tensor]), The reward after performing the action.
            - done (Union[np.ndarray, Tensor]), Whether the simulation finishes or not.
            - args (Union[np.ndarray, Tensor], optional), Support arbitrary outputs, but user needs to ensure the
                dtype. This output is optional.
        """
        promise = self.call("step", action)
        return promise

    def call(self, name, *args, **kwargs):
        """
        Call the corresponding function in process

        Args:
            name (str): The called function name.
            *args (Any, optional): args for this function.
            **kwargs (Any, optional): kwargs for this function.

        Returns:
            promise (function): A promise of call.
        """
        payload = name, args, kwargs
        self._local_conn.send((self._CALL, payload))
        return self.receive

    def get_attr(self, name: str):
        """
        Get the attr from environment in current process

        Args:
            name (str): The attribute name.

        Returns:
            promise (function): a promise of get_attr.

        """
        self._local_conn.send((self._GETATTR, name))
        return self.receive

    def close(self) -> bool:
        r"""
        Close the environment to release the resource.

        Returns:
            Success (np.bool\_), Whether shutdown the process or threading successfully.
        """
        self._local_conn.send((self._CLOSE, None))
        self._local_conn.close()
        if self._process.is_alive():
            self._process.terminate()
            self._process.join()
        return True

    def send(
        self, action: Union[Tensor, np.ndarray], env_id: Union[Tensor, np.ndarray]
    ):
        r"""
        Execute the environment step asynchronously. User can obtain result by using recv.

        Args:
            action (Union[Tensor, np.ndarray]): A tensor or array that contains the action information.
            env_id (Union[Tensor, np.ndarray]): Which environment these actions will interact with.

        Returns:
            Success (bool): True if the action is successfully executed, otherwise False.
        """
        raise ValueError("ProcessEnvironment does not support send yet")

    def recv(self):
        r"""
        Receive the result of interacting with environment.

        Returns:
            - state (Union[np.ndarray, Tensor]), The environment state after performing the action.
            - reward (Union[np.ndarray, Tensor]), The reward after performing the action.
            - done (Union[np.ndarray, Tensor]), whether the simulation finishes or not.
            - env_id (Union[np.ndarray, Tensor]), Which environments are interacted.env
            - args (Union[np.ndarray, Tensor], optional), Support arbitrary outputs, but user needs to ensure the
                dtype. This output is optional.
        """
        raise ValueError("ProcessEnvironment does not support recv yet")

    def receive(self):
        """Inner receive function"""
        message, payload = self._local_conn.recv()
        if message == self._EXCEPTION:
            trace_back = payload
            raise Exception(trace_back)
        if message == self._RESULT:
            return payload
        self.close()
        raise KeyError(f"Received unknown message {message}")

    def _worker(self, worker_conn):
        """Inner worker for each process"""
        try:
            env_creators = cloudpickle.loads(self._pickled_env_creators)
            env_id = cloudpickle.loads(self._pickled_env_id)
            envs = [
                env_creator(env_id[i]) for i, env_creator in enumerate(env_creators)
            ]
            while True:
                message, payload = worker_conn.recv()
                if message == self._CALL:
                    name, args, kwargs = payload
                    if self._num_env_per_worker > 1 and (name in ("step", "set_seed")):
                        result = [
                            getattr(env, name)(args[0][i]) for i, env in enumerate(envs)
                        ]
                    else:
                        result = [
                            getattr(env, name)(*args, **kwargs)
                            for i, env in enumerate(envs)
                        ]
                    worker_conn.send((self._RESULT, result))
                elif message == self._GETATTR:
                    name = payload
                    result = getattr(envs[0], name)
                    worker_conn.send((self._RESULT, result))
                elif message == self._CLOSE:
                    [env.close() for env in envs]
                    break
                else:
                    raise KeyError(f"Received unknown message {message}")
        except Exception:
            trace_back = traceback.format_exc()
            error_message = f"Error in environment process function {trace_back}"
            logger.error(error_message)
            worker_conn.send((self._EXCEPTION, trace_back))
        finally:
            worker_conn.close()
