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
"""
The environment base class.
"""
from typing import Callable, Iterable, Sequence, Union

# pylint: disable=R1710
import numpy as np
from mindspore import Tensor

from mindspore_rl.environment.environment import Environment
from mindspore_rl.environment.space import Space

EnvCreator = Callable[[], Environment]


class Wrapper(Environment):
    r"""
    PyFuncWrapper is a wrapper which is able to exposes a python environment as an in-graph MS environment.
    It will transfer reset, step and render function from python to mindspore ops.

    Args:
        env_creators (Union[Sequence[EnvCreator], EnvCreator]): A list of env creator or a single env creator.
        num_environment (int, optional): The number of environment. If user does not provide, the length of
            env_creators will be the number of environment. Default: None.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(
        self,
        env_creators: Union[Sequence[EnvCreator], EnvCreator],
        num_environment: int = None,
    ):
        super().__init__()
        self._batched = False
        self._num_environment = num_environment
        if isinstance(env_creators, Iterable):
            self._batched = True
            self._envs = [env_creator() for env_creator in env_creators]
        else:
            self._envs = env_creators()

    @property
    def environment(self) -> Union[Environment, Sequence[Environment]]:
        """
        Get the environment instance:

        Returns:
            Environment(Union[Sequence[Environment], Environment]: The environment instance.
        """
        return self._envs

    @property
    def batched(self) -> bool:
        """
        Whether the environment is batched.

        Returns:
            batched (bool), Whether the environment is batched. Default: False.
        """
        return self._batched or self.environment.batched

    @property
    def action_space(self) -> Space:
        """
        Get the action space of the environment.

        Returns:
            action_space(Space): The action space of environment.
        """
        if self._batched:
            low, high = self._envs[0].action_space.boundary
            return Space(
                self._envs[0].action_space.shape,
                dtype=self._envs[0].action_space.np_dtype,
                batch_shape=(self.num_environment,),
                low=low,
                high=high,
            )
        return self._envs.action_space

    @property
    def observation_space(self) -> Space:
        """
        Get the observation space of the environment.

        Returns:
            observation_space(Space): The observation space of environment.
        """
        if self._batched:
            low, high = self._envs[0].observation_space.boundary
            return Space(
                self._envs[0].observation_space.shape,
                dtype=self._envs[0].observation_space.np_dtype,
                batch_shape=(self.num_environment,),
                low=low,
                high=high,
            )
        return self._envs.observation_space

    @property
    def reward_space(self) -> Space:
        """
        Get the reward space of the environment.

        Returns:
            reward_space(Space): The reward space of environment.
        """
        if self._batched:
            low, high = self._envs[0].reward_space.boundary
            return Space(
                self._envs[0].reward_space.shape,
                dtype=self._envs[0].reward_space.np_dtype,
                batch_shape=(self.num_environment,),
                low=low,
                high=high,
            )
        return self._envs.reward_space

    @property
    def done_space(self) -> Space:
        """
        Get the done space of the environment.

        Returns:
            done_space(Space): The done space of environment.
        """
        if self._batched:
            low, high = self._envs[0].done_space.boundary
            return Space(
                self._envs[0].done_space.shape,
                dtype=self._envs[0].done_space.np_dtype,
                batch_shape=(self.num_environment,),
                low=low,
                high=high,
            )
        return self._envs.done_space

    @property
    def config(self) -> dict:
        """
        Get the config of environment.

        Returns:
            config_dict(dict): A dictionary which contains environment's info.
        """
        return self._envs[0].config if self._batched else self._envs.config

    @property
    def num_environment(self) -> int:
        r"""
        Number of environment

        Returns:
            num_env (int, optional), If the environment is not batched, it will return
                None. Otherwise, it needs to return an int value which is larger than 0. Default: None.
        """
        if self._num_environment is not None:
            num_environment = self._num_environment
        elif self._batched:
            num_environment = self._envs[0].num_environment * len(self._envs)
        else:
            num_environment = self._envs.num_environment
        return num_environment

    @property
    def num_agent(self) -> int:
        """
        Number of agents in the environment.

        Returns:
            num_agent (int), Number of agent in the current environment. If the environment is
                single agent, it will return 1. Otherwise, subclass needs to override this property
                to return correct number of agent. Default: 1.
        """
        return self._envs[0].num_agent if self._batched else self._envs.num_agent

    @property
    def _num_reset_out(self) -> int:
        """
        Inner method, return the number of return value of reset.

        Returns:
            int, The number of return value of reset.
        """
        return (
            getattr(self._envs[0], "_num_reset_out")
            if self._batched
            else getattr(self._envs, "_num_reset_out")
        )

    @property
    def _num_step_out(self) -> int:
        """
        Inner method, return the number of return value of step.

        Returns:
            int, The number of return value of step.
        """
        return (
            getattr(self._envs[0], "_num_step_out")
            if self._batched
            else getattr(self._envs, "_num_step_out")
        )

    def set_seed(self, seed_value: Union[int, Sequence[int]]) -> bool:
        r"""
        Set seed to control the randomness of environment.

        Args:
            seed_value (int), The value that is used to set

        Returns:
            Success (np.bool\_), Whether successfully set the seed.
        """
        if self._batched:
            return all(env.set_seed(seed_value) for env in self._envs)
        return self._envs.set_seed(seed_value)

    def render(self) -> Union[Tensor, np.ndarray]:
        """
        Generate the image for current frame of environment.

        Returns:
            img (Tensor), The image of environment at current frame.
        """
        if self._batched:
            raise ValueError("For multi environment, render does not support yet.")
        return self._envs.render()

    def reset(self):
        """
        Reset the environment to the initial state. It is always used at the beginning of each
        episode. It will return the value of initial state or other initial information.

        Returns:
            - state (Tensor), A Tensor which states for the initial state of environment.
            - args (Tensor, optional), Support arbitrary outputs, but user needs to ensure the
                dtype. This output is optional.
        """
        return self._reset()

    def step(self, action: Tensor):
        r"""
        Execute the environment step, which means that interact with environment once.

        Args:
            action (np.ndarray): A tensor that contains the action information.

        Returns:
            - state (Tensor), The environment state after performing the action.
            - reward (Tensor), The reward after performing the action.
            - done (Tensor), Whether the simulation finishes or not.
            - args (Tensor, optional), Support arbitrary outputs, but user needs to ensure the
                dtype. This output is optional.
        """
        return self._step(action)

    def close(self) -> bool:
        r"""
        Close the environment to release the resource.

        Returns:
            - Success(np.bool\_), Whether shutdown the process or threading successfully.
        """
        if self._batched:
            return all(env.close() for env in self._envs)
        return self._envs.close()

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
        return self._send(action, env_id)

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
        return self._recv()

    def _send(
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
        raise NotImplementedError("Method _send should be overridden by subclass.")

    def _recv(self):
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
        raise NotImplementedError("Method _recv should be overridden by subclass.")

    def _reset(self):
        """
        The inner reset function implementation. Each subclass needs to implement this function.

        Returns:
            - state (Union[np.ndarray, Tensor]), A numpy array or Tensor which states for
                the initial state of environment.
            - args (Union[np.ndarray, Tensor], optional), Support arbitrary outputs, but user needs to ensure the
                dtype. This output is optional.
        """
        raise NotImplementedError("Method _reset should be overridden by subclass.")

    def _step(self, action: Union[Tensor, np.ndarray]):
        """
        The inner step function implementation. Subclass needs to implement this function.

        Args:
            action (Union[np.ndarray, Tensor]): A numpy array or Tensor that states for the action of agent(s).

        Returns:
            - state (Union[np.ndarray, Tensor]), The environment state after performing the action.
            - reward (Union[np.ndarray, Tensor]), The reward after performing the action.
            - done (Union[np.ndarray, Tensor]), Whether the simulation finishes or not.
            - args (Union[np.ndarray, Tensor], optional), Support arbitrary outputs, but user needs to ensure the
                dtype. This output is optional.
        """
        raise NotImplementedError("Method _step should be overridden by subclass.")
