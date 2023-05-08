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
"""Python Environment"""

from typing import List, Optional, Sequence, Union

import numpy as np
from mindspore import Tensor
from mindspore import log as logger

from mindspore_rl.environment.environment import Environment
from mindspore_rl.environment.space import Space
from mindspore_rl.utils.utils import check_type, check_valid_return_value


class PythonEnvironment(Environment):
    r"""
    All the third party environment must inherit this class and implement \_reset and \_step function
    This class is used to make all the third party environment compatible with mindspore environment api.

    Args:
        action_space (Union[Space, List[Space]]): Action space of the environment. It must be MindSpore Space.
            User can use utils in space\_adapter.py to adapt third party space to MindSpore Space
        observation_space (Union[Space, List[Space]]): Observation space of the environment. It must be MindSpore Space.
        reward_space (Space, optional): Reward space of the environment. It must be MindSpore Space, if user does not
            provide reward space, a default space `Space((), np.float32)` will be provided. Default: None
        done_space (Space, optional): Done space of the environment. It must be MindSpore Space, if user does not
            provide done space, a default space `Space((), np.bool_, low=0, high=2)` will be provided. Default: None
        config (dict, optional): Configuration of the environment. It must be a dictionary. Default: None
        need_auto_reset (bool, optional): Whether the python environment needs to do auto reset for the subclass
            environment. Default: False

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(
        self,
        action_space: Union[Space, List[Space]],
        observation_space: Union[Space, List[Space]],
        reward_space: Optional[Space] = None,
        done_space: Optional[Space] = None,
        config: Optional[dict] = None,
        need_auto_reset: bool = False,
    ):
        super().__init__()

        # Set environment space
        self._action_space, _ = check_type(Space, action_space, "action_space")
        self._observation_space, _ = check_type(
            Space, observation_space, "observation_space"
        )
        self._reward_space = (
            check_type(Space, reward_space, "reward_space")[0]
            if reward_space is not None
            else Space((), dtype=np.float32)
        )
        self._done_space = (
            check_type(Space, done_space, "done_space")[0]
            if done_space is not None
            else Space((), low=0, high=2, dtype=np.bool_)
        )
        self._config = (
            check_type(dict, config, "config")[0] if config is not None else {}
        )

        self._need_auto_reset = need_auto_reset
        self._done_flag = np.ones(self._done_space.shape, self._done_space.np_dtype)
        # Pre-run environment
        reset_out = self._reset()
        action = self._action_space.sample()
        step_out = self._step(action)
        # Obtain the output number of reset and step
        self._num_env_reset_out = check_valid_return_value(reset_out, "reset")
        self._num_env_step_out = check_valid_return_value(step_out, "step")
        if (
            self._num_env_reset_out > 1 or self._num_env_step_out > 3
        ) and self._need_auto_reset:
            logger.warning(
                "There are extra output for reset or step, the auto reset may not work properly."
            )

    @property
    def action_space(self) -> Space:
        """
        Get the action space of the environment.

        Returns:
            action_space (Space), The action space of environment.
        """
        return self._action_space

    @property
    def observation_space(self) -> Space:
        """
        Get the state space of the environment.

        Returns:
            observation_space (Space), The state space of environment.
        """
        return self._observation_space

    @property
    def reward_space(self) -> Space:
        """
        Get the reward space of the environment.

        Returns:
            reward_space (Space), The reward space of environment.
        """
        return self._reward_space

    @property
    def done_space(self) -> Space:
        """
        Get the done space of the environment.

        Returns:
            done_space (Space), The done space of environment.
        """
        return self._done_space

    @property
    def config(self) -> dict:
        """
        Get the config of environment.

        Returns:
            config (dict), A dictionary which contains environment's info.
        """
        return self._config

    @property
    def _num_reset_out(self) -> int:
        """
        Inner method, return the number of return value of reset.

        Returns:
            int, The number of return value of reset.
        """
        return self._num_env_reset_out

    @property
    def _num_step_out(self) -> int:
        """
        Inner method, return the number of return value of step.

        Returns:
            int, The number of return value of step.
        """
        return self._num_env_step_out

    def should_reset(self, done: np.bool_):
        r"""
        Whether the framework need to reset the environment in the current state.

        Args:
            done (np.bool\_): Input done getting from environment.

        Returns:
            bool, States for whether the environment need to be reset.

        """
        return np.all(done) and self._need_auto_reset

    def step(self, action: np.ndarray):
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
        if self.should_reset(self._done_flag):
            reset_out = self.reset()
            init_reward = (
                np.zeros_like(self._reward_space.shape, self._reward_space.np_dtype)
                if len(self._reward_space.shape)
                else np.zeros((), self._reward_space.np_dtype)
            )
            self._done_flag = np.zeros_like(self._done_flag)
            step_out = (
                (reset_out, init_reward, self._done_flag)
                if self._num_reset_out == 1
                else (reset_out, init_reward, self._done_flag, *reset_out[1:])
            )
        else:
            step_out = self._step(action)
            if self._num_step_out < 3:
                raise ValueError(
                    f"The return number of _step must be larger and equal to 3, but got {self._num_step_out}"
                )
            next_state = step_out[0].astype(self.observation_space.np_dtype)
            reward = step_out[1].astype(self.reward_space.np_dtype)
            done = step_out[2].astype(self.done_space.np_dtype)
            squashed_flag = self._done_flag.reshape(
                -1,
            ).all()
            if not squashed_flag:
                self._done_flag = done
                step_out = (
                    (next_state, reward, done, *step_out[3:])
                    if self._num_step_out > 3
                    else (next_state, reward, done)
                )
            else:
                new_step_out = [
                    np.zeros_like(step_out[0]),
                    np.zeros_like(step_out[1]),
                    np.ones_like(step_out[2]),
                ]
                if self._num_step_out > 3:
                    for i in range(3, len(step_out)):
                        new_step_out.append(np.zeros_like(step_out[i]))
                step_out = tuple(new_step_out)

        return step_out

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
        self._done_flag = np.zeros_like(self._done_flag)
        reset_out = self._reset()
        if self._num_reset_out < 1:
            raise ValueError(
                f"The return number of _reset must be larger and equal to 1, but got {self._num_reset_out}"
            )
        state_value = reset_out if self._num_reset_out == 1 else reset_out[0]
        new_state = state_value.astype(self.observation_space.np_dtype)
        reset_out = (
            (new_state, *reset_out[1:]) if self._num_reset_out > 1 else new_state
        )
        return reset_out

    def set_seed(self, seed_value: Union[int, Sequence[int]]) -> bool:
        r"""
        Set seed to control the randomness of environment.

        Args:
            seed_value (int), The value that is used to set

        Retunrs:
            Success (np.bool\_), Whether successfully set the seed.
        """
        return self._set_seed(seed_value)

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

    def render(self) -> Union[Tensor, np.ndarray]:
        """
        Generate the image for current frame of environment.

        Returns:
            img (Union[Tensor, np.ndarray]), The image of environment at current frame.
        """
        return self._render()

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

    def _set_seed(self, seed_value: Union[int, Sequence[int]]) -> bool:
        r"""
        Set seed to control the randomness of environment.

        Args:
            seed_value (int), The value that is used to set

        Returns:
            Success (np.bool\_), Whether successfully set the seed.
        """
        raise NotImplementedError("Method _set_seed should be overridden by subclass.")

    def _render(self) -> Union[Tensor, np.ndarray]:
        """
        Generate the image for current frame of environment.

        Returns:
            img (Union[Tensor, np.ndarray]), The image of environment at current frame.
        """
        raise NotImplementedError("Method render should be overridden by subclass.")

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
        raise ValueError("Python Environment does not support send yet")

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
        raise ValueError("Python Environment does not support recv yet")
