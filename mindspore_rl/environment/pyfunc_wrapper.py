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
from mindspore import ops
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P

from mindspore_rl.environment.environment import Environment
from mindspore_rl.environment.wrapper import Wrapper

EnvCreator = Callable[[], Environment]


class PyFuncWrapper(Wrapper):
    r"""
    PyFuncWrapper is a wrapper which is able to exposes a python environment as an in-graph MS environment.
    It will transfer reset, step and render function from python to mindspore ops.

    Args:
        env_creator (Environment): The environment instance.
        stateful (bool): Whether add side effect mark for pyfunc ops. Default: True.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self, env_creator: EnvCreator, stateful: bool = True):
        if not callable(env_creator):
            raise TypeError(
                f"The input env_creators must be a list of callable, but got {env_creator}"
            )
        super().__init__(env_creator)
        env_name = self.environment.__class__.__name__

        # pre-run environment
        reset_out = self.environment.reset()
        if self.environment.batched:
            action = []
            for _ in range(self.environment.num_environment):
                action.append(self.environment.action_space.sample())
            action = np.array(action)
        else:
            action = self.environment.action_space.sample()
        step_out = self.environment.step(action)

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

        # reset output dtype and shape
        self._reset_output_msdtype = reset_out_msdtype
        self._reset_output_full_shape = reset_output_full_shape
        # step input dtype and shape
        self._step_input_msdtype = step_input_msdtype
        self._step_input_full_shape = step_input_full_shape
        # step output dtype and shape
        self._step_output_msdtype = step_output_msdtype
        self._step_output_full_shape = step_output_full_shape

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
        logger.info(f"Observation space is {self.environment.observation_space}")
        logger.info(f"Action space is {self.environment.action_space}")
        logger.info(f"Reward space is {self.environment.reward_space}")
        logger.info(f"Done space is {self.environment.done_space}")

        self.reset_ops = P.PyFunc(
            self.environment.reset,
            [],
            [],
            reset_out_msdtype,
            reset_output_full_shape,
            stateful=stateful,
        )
        self.step_ops = P.PyFunc(
            self.environment.step,
            step_input_msdtype,
            step_input_full_shape,
            step_output_msdtype,
            step_output_full_shape,
            stateful=stateful,
        )
        self._input_dtype = self.environment.action_space.ms_dtype

    @property
    def reset_output_shape(self) -> Sequence[int]:
        """
        Get the output shape of reset function.

        Returns:
            reset_output_shape(tuple): The output shape of reset function.
        """
        return self._reset_output_full_shape

    @property
    def reset_output_dtype(self) -> Sequence[int]:
        """
        Get the output dtype of reset function.

        Returns:
            reset_output_dtype(str): The output dtype of reset function.
        """
        return self._reset_output_msdtype

    @property
    def step_input_shape(self) -> Sequence[int]:
        """
        Get the input shape of step function.

        Returns:
            step_input_shape(tuple): The input shape of step function.
        """
        return self._step_input_full_shape

    @property
    def step_input_dtype(self) -> Sequence[int]:
        """
        Get the input dtype of step function.

        Returns:
            step_input_dtype(str): The input dtype of step function.
        """
        return self._step_input_msdtype

    @property
    def step_output_shape(self) -> Sequence[int]:
        """
        Get the output shape of step function.

        Returns:
            step_output_shape(tuple): The output shape of step function.
        """
        return self._step_output_full_shape

    @property
    def step_output_dtype(self) -> Sequence[int]:
        """
        Get the output dtype of step function.

        Returns:
            step_output_dtype(str): The output dtype of step function.
        """
        return self._step_output_msdtype

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
        return self.reset_ops()

    def _reset(self):
        """
        Reset the environment to the initial state. It is always used at the beginning of each
        episode. It will return the value of initial state or other initial information.

        Returns:
            - state (Tensor), A Tensor which states for the initial state of environment.
            - args (Tensor, optional), Support arbitrary outputs, but user needs to ensure the
                dtype. This output is optional.
        """
        return self.environment.reset()

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
        action = ops.cast(action, self._input_dtype)
        return self.step_ops(action)

    def _step(self, action: Union[np.ndarray]):
        """
        The inner step function implementation. Subclass needs to implement this function.

        Args:
            action (Union[np.ndarray, Tensor]): A numpy array or Tensor that states for the action of agent(s).

        Returns:
            - state (np.ndarray), The environment state after performing the action.
            - reward (np.ndarray), The reward after performing the action.
            - done (np.ndarray), Whether the simulation finishes or not.
            - args (np.ndarray], optional), Support arbitrary outputs, but user needs to ensure the
                dtype. This output is optional.
        """
        return self.environment.step(action)

    def _send(self, action: Tensor, env_id: Tensor):
        r"""
        Execute the environment step asynchronously. User can obtain result by using recv.

        Args:
            action (Tensor): A tensor or array that contains the action information.
            env_id (Tensor): Which environment these actions will interact with.

        Returns:
            Success (Tensor): True if the action is successfully executed, otherwise False.
        """
        raise ValueError("PyFuncWrapper does not support send yet")

    def _recv(self):
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
