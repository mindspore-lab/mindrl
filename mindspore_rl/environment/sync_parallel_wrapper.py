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
"""Sync Parallel Wrapper"""
# pylint:disable=W0106
from typing import Callable, Optional, Sequence, Union

import numpy as np
from mindspore import Tensor

from mindspore_rl.environment.environment import Environment
from mindspore_rl.environment.process_environment import ProcessEnvironment
from mindspore_rl.environment.space import Space

EnvCreator = Callable[[], Environment]


class SyncParallelWrapper(Environment):
    r"""
    Execute environment synchronously in parallel. The result will be returned when all the environment are
    finished.

    Args:
        env_creators (Sequence[EnvCreator]):  A list of env creator.
        num_proc (int, optional): The number of parallel processes. If user does not provided num\_proc, framework
            will use the same number of processes as number of environment. Defaults: 0.
        shared_memory (bool, optional): Whether to use shared memory to share the data between processes. This
            functionality is not implemented yet. Default: False
    """

    def __init__(
        self,
        env_creators: Sequence[EnvCreator],
        num_proc: int = 0,
        shared_memory: bool = False,
    ):
        super().__init__()
        self._shared_memory = shared_memory
        type_check = [not callable(env_creator) for env_creator in env_creators]
        if any(type_check):
            raise TypeError(
                f"The input env_creators must be a list of callable, but got {env_creators}"
            )
        if num_proc < 0:
            raise ValueError(f"num_proc must be a positive integer, but got {num_proc}")
        self._num_env = len(env_creators)
        self._num_proc = self._num_env if num_proc == 0 else num_proc
        if self._num_proc > self._num_env:
            raise ValueError(
                "The number of processes must be less or equal to number of environment, "
                f"but got number of processes {self._num_proc}, number of environment {self._num_env}"
            )
        self._envs = []
        avg_env_per_proc = int(self._num_env / self._num_proc)
        for i in range(self._num_proc):
            assigned_env_num = i * avg_env_per_proc
            if assigned_env_num < self._num_env:
                env_num = avg_env_per_proc
            else:
                env_num = self._num_env - assigned_env_num
            proc_env = ProcessEnvironment(
                env_creators[env_num * i : env_num * (i + 1)],
                range(env_num * i, env_num * (i + 1)),
            )
            self._envs.append(proc_env)
            proc_env.start()

    @property
    def batched(self) -> bool:
        """
        Whether the environment is batched.

        Returns:
            batched (bool), Whether the environment is batched. Default: False.
        """
        return True

    @property
    def action_space(self) -> Space:
        """
        Get the action space of the environment.

        Returns:
            action_space(Space): The action space of environment.
        """
        return self._envs[0].action_space

    @property
    def observation_space(self) -> Space:
        """
        Get the observation space of the environment.

        Returns:
            observation_space(Space): The observation space of environment.
        """
        return self._envs[0].observation_space

    @property
    def reward_space(self) -> Space:
        """
        Get the reward space of the environment.

        Returns:
            reward_space(Space): The reward space of environment.
        """
        return self._envs[0].reward_space

    @property
    def done_space(self) -> Space:
        """
        Get the done space of the environment.

        Returns:
            done_space(Space): The done space of environment.
        """
        return self._envs[0].done_space

    @property
    def config(self) -> dict:
        """
        Get the config of environment.

        Returns:
            config_dict(dict): A dictionary which contains environment's info.
        """
        return self._envs[0].config

    @property
    def num_environment(self) -> Optional[int]:
        """
        Number of environment

        Returns:
            num_env (int),  Number of environment.
        """
        return self._num_env

    @property
    def num_agent(self) -> int:
        """
        Number of agents in the environment.

        Returns:
            num_agent (int), Number of agent in the current environment. If the environment is
                single agent, it will return 1. Otherwise, subclass needs to override this property
                to return correct number of agent. Default: 1.
        """
        return self._envs[0].num_agent

    @property
    def _num_reset_out(self) -> int:
        """
        Inner method, return the number of return value of reset.

        Returns:
            int, The number of return value of reset.
        """
        return getattr(self._envs[0], "_num_reset_out")

    @property
    def _num_step_out(self) -> int:
        """
        Inner method, return the number of return value of step.

        Returns:
            int, The number of return value of step.
        """
        return getattr(self._envs[0], "_num_step_out")

    def start(self) -> bool:
        """
        Start all the process environment.

        Returns:
            bool, Whether start processes successfully.
        """
        for env in self._envs:
            env.start()
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
        promise_list = [env.reset() for env in self._envs]
        reset_out = []
        [reset_out.extend(promise()) for promise in promise_list]
        if self._num_reset_out == 1:
            # if reset func only has one return value, do stack
            stacked_reset_out = np.array(reset_out)
        else:
            # else for each return value do the stack
            s0, *others = map(np.array, zip(*reset_out))
            stacked_reset_out = (s0, *others)
        return stacked_reset_out

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
        promise_list = []
        accum_env_num = 0
        for i in range(self._num_proc):
            worker_env_num = getattr(self._envs[i], "_num_env_per_worker")
            action_i = action[accum_env_num : worker_env_num + accum_env_num]
            if (len(action_i.shape) - len(self.action_space.shape) > 0) and (
                action_i.shape[0] == 1
            ):
                action_i = action_i.squeeze(0)
            promise_list.append(self._envs[i].step(action_i))
            accum_env_num += worker_env_num
        step_out = []
        [step_out.extend(promise()) for promise in promise_list]
        obs, rewards, dones, *others = map(np.array, zip(*step_out))
        stacked_step_out = (obs, rewards, dones, *others)
        return stacked_step_out

    def set_seed(self, seed_value: Union[int, Sequence[int]]) -> bool:
        r"""
        Set seed to control the randomness of environment.

        Args:
            seed_value (int), The value that is used to set

        Retunrs:
            Success (np.bool\_), Whether successfully set the seed.
        """
        accum_env_num = 0
        success_list = []
        for i in range(self._num_proc):
            worker_env_num = getattr(self._envs[i], "_num_env_per_worker")
            seed_list = seed_value[accum_env_num : worker_env_num + accum_env_num]
            accum_env_num += worker_env_num
            success_list.append(self._envs[i].set_seed(seed_list))
        return np.array(success_list).all()

    def close(self):
        r"""
        Close the environment to release the resource.

        Returns:
            Success(np.bool\_), Whether shutdown the process or threading successfully.
        """
        for env in self._envs:
            env.close()
        return True

    def send(self, action: Tensor, env_id: Tensor):
        r"""
        Execute the environment step asynchronously. User can obtain result by using recv.

        Args:
            action (Tensor): A tensor or array that contains the action information.
            env_id (Tensor): Which environment these actions will interact with.

        Returns:
            Success (Tensor): True if the action is successfully executed, otherwise False.
        """
        raise ValueError("SyncParallelWrapper does not support send yet")

    def recv(self):
        r"""
        Receive the result of interacting with environment.

        Returns:
            - state (Tensor), The environment state after performing the action.
            - reward (Tensor), The reward after performing the action.
            - done (Tensor), whether the simulation finishes or not.
            - env_id (Tensor), Which environments are interacted.env
            - args (Tensor, optional), Support arbitrary outputs, but user needs to ensure the
                dtype. This output is optional.
        """
        raise ValueError("SyncParallelWrapper does not support recv yet")

    def render(self) -> Union[Tensor, np.ndarray]:
        """
        Generate the image for current frame of environment.

        Returns:
            img (Union[Tensor, np.ndarray]), The image of environment at current frame.
        """
        raise ValueError("Parallel rendering does not support yet")
