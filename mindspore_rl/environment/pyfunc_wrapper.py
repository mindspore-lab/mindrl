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
from typing import Callable, Sequence, Union

# pylint: disable=R1710
import numpy as np
from mindspore import Tensor
from mindspore import log as logger
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P

from mindspore_rl.environment.environment import Environment
from mindspore_rl.environment.space import Space

EnvCreator = Callable[[], Environment]


class PyFuncWrapper(Environment):
    r"""
    PyFuncWrapper is a wrapper which is able to exposes a python environment as an in-graph MS environment.
    It will transfer reset, step and render function from python to mindspore ops.

    Args:
        env_instance (Environment): The environment instance.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self, env_creator: EnvCreator):
        super().__init__()
        if not callable(env_creator):
            raise TypeError(
                f"The input env_creators must be a list of callable, but got {env_creator}"
            )
        self._env = env_creator()
        env_name = self._env.__class__.__name__

        # pre-run environment
        reset_out = self._env.reset()
        if self._env.batched:
            action = []
            for _ in range(self._env.num_environment):
                action.append(self._env.action_space.sample())
            action = np.array(action)
        else:
            action = self._env.action_space.sample()
        step_out = self._env.step(action)

        # Check whether the output of reset/step func is a numpy array
        self._check_ndarray(reset_out, "reset")
        self._check_ndarray(step_out, " step")

        (
            reset_out_dtype,
            reset_out_msdtype,
            reset_output_full_shape,
            _,
            reset_batch_shape,
        ) = self._get_dtype_shape(reset_out)
        (
            step_input_dtype,
            step_input_msdtype,
            step_input_full_shape,
            _,
            step_in_batch_shape,
        ) = self._get_dtype_shape(action)
        (
            step_output_dtype,
            step_output_msdtype,
            step_output_full_shape,
            _,
            step_out_batch_shape,
        ) = self._get_dtype_shape(step_out)

        logger.info(f"Start create PyFunc of {env_name}...")
        logger.info(
            "Please check whether the dtype/shape of input/output meet the expectation."
        )
        logger.info(
            f"The output dtype of reset is [MS]{reset_out_msdtype}, [Numpy]{reset_out_dtype}"
        )
        logger.info(f"The output shape of reset is {reset_output_full_shape}")
        logger.info(f"The output batch axis of reset is {reset_batch_shape}")
        logger.info(
            f"The input dtype of step is [MS]{step_input_msdtype}, [Numpy]{step_input_dtype}"
        )
        logger.info(f"The input shape of step is {step_input_full_shape}")
        logger.info(f"The input batch axis of step is {step_in_batch_shape}")
        logger.info(
            f"The output dtype of step is [MS]{step_output_msdtype}, [Numpy]{step_output_dtype}"
        )
        logger.info(f"The output shape of step is {step_output_full_shape}")
        logger.info(f"The output batch axis of step is {step_out_batch_shape}")
        logger.info(f"Start create Space of {env_name}...")
        logger.info(f"Observation space is {self._env.observation_space}")
        logger.info(f"Action space is {self._env.action_space}")
        logger.info(f"Reward space is {self._env.reward_space}")
        logger.info(f"Done space is {self._env.done_space}")

        self.reset_ops = P.PyFunc(
            self._env.reset, [], [], reset_out_msdtype, reset_output_full_shape
        )
        self.step_ops = P.PyFunc(
            self._env.step,
            step_input_msdtype,
            step_input_full_shape,
            step_output_msdtype,
            step_output_full_shape,
        )

    @property
    def action_space(self) -> Space:
        """
        Get the action space of the environment.

        Returns:
            action_space(Space): The action space of environment.
        """
        return self._env.action_space

    @property
    def observation_space(self) -> Space:
        """
        Get the observation space of the environment.

        Returns:
            observation_space(Space): The observation space of environment.
        """
        return self._env.observation_space

    @property
    def reward_space(self) -> Space:
        """
        Get the reward space of the environment.

        Returns:
            reward_space(Space): The reward space of environment.
        """
        return self._env.reward_space

    @property
    def done_space(self) -> Space:
        """
        Get the done space of the environment.

        Returns:
            done_space(Space): The done space of environment.
        """
        return self._env.done_space

    @property
    def config(self) -> dict:
        """
        Get the config of environment.

        Returns:
            config_dict(dict): A dictionary which contains environment's info.
        """
        return self._env.config

    @property
    def num_environment(self) -> int:
        r"""
        Number of environment

        Returns:
            num_env (int, optional), If the environment is not batched, it will return
                None. Otherwise, it needs to return an int value which is larger than 0. Default: None.
        """
        return self._env.num_environment

    @property
    def num_agent(self) -> int:
        """
        Number of agents in the environment.

        Returns:
            num_agent (int), Number of agent in the current environment. If the environment is
                single agent, it will return 1. Otherwise, subclass needs to override this property
                to return correct number of agent. Default: 1.
        """
        return self._env.num_agent

    @property
    def _num_reset_out(self) -> int:
        """
        Inner method, return the number of return value of reset.

        Returns:
            int, The number of return value of reset.
        """
        return getattr(self._env, "_num_reset_out")

    @property
    def _num_step_out(self) -> int:
        """
        Inner method, return the number of return value of step.

        Returns:
            int, The number of return value of step.
        """
        return getattr(self._env, "_num_step_out")

    def set_seed(self, seed_value: Union[int, Sequence[int]]) -> bool:
        r"""
        Set seed to control the randomness of environment.

        Args:
            seed_value (int), The value that is used to set

        Returns:
            Success (np.bool\_), Whether successfully set the seed.
        """
        return self._env.set_seed(seed_value)

    def render(self) -> Union[Tensor, np.ndarray]:
        """
        Generate the image for current frame of environment.

        Returns:
            img (Tensor), The image of environment at current frame.
        """
        return self._env.render()

    def reset(self):
        """
        Reset the environment to the initial state. It is always used at the beginning of each
        episode. It will return the value of initial state or other initial information.

        Returns:
            - state (Tensor), A Tensor which states for the initial state of environment.
            - args (Tensor, optional), Support arbitrary outputs, but user needs to ensure the
                dtype. This output is optional.
        """
        if self._num_reset_out == 1:
            state = self.reset_ops()[0]
            return state
        if self._num_reset_out != 1:
            return self.reset_ops()

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
        return self.step_ops(action)

    def close(self) -> bool:
        r"""
        Close the environment to release the resource.

        Returns:
            - Success(np.bool\_), Whether shutdown the process or threading successfully.
        """
        return self._env.close()

    def send(self, action: Tensor, env_id: Tensor):
        r"""
        Execute the environment step asynchronously. User can obtain result by using recv.

        Args:
            action (Tensor): A tensor or array that contains the action information.
            env_id (Tensor): Which environment these actions will interact with.

        Returns:
            Success (Tensor): True if the action is successfully executed, otherwise False.
        """
        raise ValueError("PyFuncWrapper does not support send yet")

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
        raise ValueError("PyFuncWrapper does not support recv yet")

    def _shape_adapter(self, input_shape):
        """Treat the first axis of input shape as batch axis, if the length of shape is larger than 2"""
        batch_shape = None
        out_split_shape = input_shape
        if len(input_shape) > 1:
            batch_shape = input_shape[:1]
            out_split_shape = input_shape[1:]
        return input_shape, out_split_shape, batch_shape

    def _get_dtype_shape(self, result):
        """Generate mindspore dytpe/numpy dtype and shape for a set of input value"""
        out_np_dtype = []
        out_ms_dtype = []
        out_split_shape = []
        out_full_shape = []
        batch_shape = []
        if isinstance(result, tuple):
            for item in result:
                if isinstance(item, tuple):
                    list_dtype_shape = self._get_dtype_shape(item)
                    out_np_dtype.extend(list_dtype_shape[0])
                    out_ms_dtype.extend(list_dtype_shape[1])
                    out_split_shape.extend(list_dtype_shape[2])
                    out_full_shape.extend(list_dtype_shape[3])
                    batch_shape.extend(list_dtype_shape[4])
                else:
                    adapted_item = item
                    out_ms_dtype.append(mstype.pytype_to_dtype(adapted_item.dtype.type))
                    out_np_dtype.append(adapted_item.dtype.type)
                    (
                        item_full_shape,
                        item_split_shape,
                        adapted_batch_shape,
                    ) = self._shape_adapter(adapted_item.shape)
                    batch_shape.append(adapted_batch_shape)
                    out_split_shape.append(item_split_shape)
                    out_full_shape.append(item_full_shape)
        else:
            adapted_result = result
            out_np_dtype.append(adapted_result.dtype.type)
            out_ms_dtype.append(mstype.pytype_to_dtype(adapted_result.dtype.type))
            (
                result_full_shape,
                result_split_shape,
                adapted_batch_shape,
            ) = self._shape_adapter(adapted_result.shape)
            batch_shape.append(adapted_batch_shape)
            out_split_shape.append(result_split_shape)
            out_full_shape.append(result_full_shape)
        return out_np_dtype, out_ms_dtype, out_full_shape, out_split_shape, batch_shape

    def _check_ndarray(self, items, debug_str):
        """Check whether the input items are ndarray"""
        if isinstance(items, tuple):
            for item in items:
                self._check_ndarray(item, debug_str)
        else:
            if not isinstance(items, np.ndarray):
                raise ValueError(f"One of the output of {debug_str} is not numpy array")
        return items
