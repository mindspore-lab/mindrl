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
The environment base class.
"""
#pylint: disable=R1710
from mindspore import Tensor
from typing import Union, Sequence
import numpy as np
import mindspore.nn as nn
from mindspore import log as logger
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P
from mindspore_rl.environment.space import Space


class Environment(nn.Cell):
    r"""
    The abstract base class for environment. All the environments or wrappers need to inherit
    this base class. Moreover, subclass needs to overridden corresponding functions and properties.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self):
        super(Environment, self).__init__()

    @property
    def action_space(self) -> Space:
        """
        Get the action space of the environment.

        Returns:
            action_space (Space), The action space of environment.
        """
        raise NotImplementedError("Property action_space should be overridden by subclass.")

    @property
    def observation_space(self) -> Space:
        """
        Get the state space of the environment.

        Returns:
            observation_space (Space), The state space of environment.
        """
        raise NotImplementedError("Property observation_space should be overridden by subclass.")

    @property
    def reward_space(self) -> Space:
        """
        Get the reward space of the environment.

        Returns:
            reward_space (Space), The reward space of environment.
        """
        raise NotImplementedError("Property reward_space should be overridden by subclass.")

    @property
    def done_space(self) -> Space:
        """
        Get the done space of the environment.

        Returns:
            done_space (Space), The done space of environment.
        """
        raise NotImplementedError("Property done_space should be overridden by subclass.")

    @property
    def config(self) -> dict:
        """
        Get the config of environment.

        Returns:
            config (dict), A dictionary which contains environment's info.
        """
        raise NotImplementedError("Property config should be overridden by subclass.")

    @property
    def batched(self) -> bool:
        """
        Whether the environment is batched.

        Returns:
            batched (bool), Whether the environment is batched. Default: False.
        """
        return False

    @property
    def num_environment(self) -> int:
        """
        Number of environment

        Returns:
            num_env (int),  Number of environment.
        """
        return 1

    @property
    def num_agent(self) -> int:
        """
        Number of agents in the environment.

        Returns:
            num_agent (int), Number of agent in the current environment. If the environment is
                single agent, it will return 1. Otherwise, subclass needs to override this property
                to return correct number of agent. Default: 1.
        """
        return 1

    @property
    def _num_reset_out(self) -> int:
        """
        Inner method, return the number of return value of reset.

        Returns:
            int, The number of return value of reset.
        """
        return 1

    @property
    def _num_step_out(self) -> int:
        """
        Inner method, return the number of return value of step.

        Returns:
            int, The number of return value of step.
        """
        return 3

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
        raise NotImplementedError("Method reset should be overridden by subclass.")

    def step(self, action: Union[Tensor, np.ndarray]):
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
        raise NotImplementedError("Method step should be overridden by subclass.")

    def send(self, action: Union[Tensor, np.ndarray], env_id: Union[Tensor, np.ndarray]):
        r"""
        Execute the environment step asynchronously. User can obtain result by using recv.

        Args:
            action (Union[Tensor, np.ndarray]): A tensor or array that contains the action information.
            env_id (Union[Tensor, np.ndarray]): Which environment these actions will interact with.

        Returns:
            Success (bool): True if the action is successfully executed, otherwise False.
        """
        raise NotImplementedError("Method send should be overridden by subclass.")

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
        raise NotImplementedError("Method async_step should be overridden by subclass.")

    def set_seed(self, seed_value: Union[int, Sequence[int]]) -> bool:
        r"""
        Set seed to control the randomness of environment.

        Args:
            seed_value (int), The value that is used to set

        Returns:
            Success (np.bool\_), Whether successfully set the seed.
        """
        raise NotImplementedError("Method set_seed should be overridden by subclass.")

    def render(self) -> Union[Tensor, np.ndarray]:
        """
        Generate the image for current frame of environment.

        Returns:
            img (Union[Tensor, np.ndarray]), The image of environment at current frame.
        """
        raise NotImplementedError("Method render should be overridden by subclass.")

    def close(self) -> bool:
        r"""
        Close the environment to release the resource.

        Returns:
            Success (np.bool\_), Whether shutdown the process or threading successfully.
        """
        return True


class DeprecatedEnvironment(nn.Cell):
    r"""
    The virtual base class of the environment. Each environment subclass needs to inherit this class, and
    implements \_reset, \_get_action, \_step, \_get_min_max_action and \_get_min_max_observation in subclass.
    The base class provides the ability that automatically wrap the python function of reset and step to mindspore
    ops (PyFunc), and automatically generate the space of environmment.

    Args:
        env_name (str): The environment name of subclass. Default: None
        env (Environment): The environmen instance of subcalss. Default: None
        config (dict): The configuration of environment, it can be obtained by call the config property. Default: None
    """

    def __init__(self, env_name=None, env=None, config=None):
        super(DeprecatedEnvironment, self).__init__(auto_prefix=False)
        new_api_env = ['MultiAgentParticleEnvironment']
        if env_name in new_api_env:
            self._base_env = env
            self._config = config

            # pre-run environment
            reset_out = self._reset()
            action = self._get_action()
            step_out = self._step(action)
            min_action, max_action = self._get_min_max_action()
            min_obs, max_obs = self._get_min_max_observation()

            # Check whether the output of reset/get_action/step func is a numpy array
            self._check_ndarray(reset_out, "_reset")
            self._check_ndarray(action, "_get_action")
            self._check_ndarray(step_out, "_step")

            reset_out_dtype, reset_out_msdtype, reset_output_full_shape, reset_output_split_shape, \
                reset_batch_shape = self._get_dtype_shape(reset_out)
            step_input_dtype, step_input_msdtype, step_input_full_shape, step_input_split_shape, \
                step_in_batch_shape = self._get_dtype_shape(action)
            step_output_dtype, step_output_msdtype, step_output_full_shape, step_output_split_shape, \
                step_out_batch_shape = self._get_dtype_shape(step_out)

            self._observation_space = Space(reset_output_split_shape[0], reset_out_dtype[0],
                                            low=min_obs, high=max_obs, batch_shape=reset_batch_shape[0])
            self._action_space = Space(step_input_split_shape[0], step_input_dtype[0],
                                       low=min_action, high=max_action, batch_shape=step_in_batch_shape[0])
            self._reward_space = Space(step_output_split_shape[1],
                                       step_output_dtype[1], batch_shape=step_out_batch_shape[1])
            self._done_space = Space(step_output_split_shape[2], step_output_dtype[2],
                                     batch_shape=step_out_batch_shape[2])

            logger.info(f"Start create PyFunc of {env_name}...")
            logger.info(f"Please check whether the dtype/shape of input/output meet the expectation.")
            logger.info(f"The output dtype of reset is [MS]{reset_out_msdtype}, [Numpy]{reset_out_dtype}")
            logger.info(f"The output shape of reset is {reset_output_full_shape}")
            logger.info(f"The output batch axis of reset is {reset_batch_shape}")
            logger.info(f"The input dtype of step is [MS]{step_input_msdtype}, [Numpy]{step_input_dtype}")
            logger.info(f"The input shape of step is {step_input_full_shape}")
            logger.info(f"The input batch axis of step is {step_in_batch_shape}")
            logger.info(f"The output dtype of step is [MS]{step_output_msdtype}, [Numpy]{step_output_dtype}")
            logger.info(f"The output shape of step is {step_output_full_shape}")
            logger.info(f"The output batch axis of step is {step_out_batch_shape}")
            logger.info(f"Start create Space of {env_name}...")
            logger.info(f"Observation space is {self._observation_space}")
            logger.info(f"Action space is {self._action_space}")
            logger.info(f"Reward space is {self._reward_space}")
            logger.info(f"Done space is {self._done_space}")

            self.reset_ops = P.PyFunc(self._reset_python, [], [], reset_out_msdtype, reset_output_full_shape)
            self.step_ops = P.PyFunc(self._step_python, step_input_msdtype, step_input_full_shape,
                                     step_output_msdtype, step_output_full_shape)

            self.num_reset_out = len(reset_out_dtype)
            self.num_step_out = len(step_output_dtype)

    @property
    def action_space(self):
        """
        Get the action space of the environment.

        Returns:
            action_space(Space): The action space of environment.
        """
        return self._action_space

    @property
    def observation_space(self):
        """
        Get the observation space of the environment.

        Returns:
            observation_space(Space): The observation space of environment.
        """
        return self._observation_space

    @property
    def reward_space(self):
        """
        Get the reward space of the environment.

        Returns:
            reward_space(Space): The reward space of environment.
        """
        return self._reward_space

    @property
    def done_space(self):
        """
        Get the done space of the environment.

        Returns:
            done_space(Space): The done space of environment.
        """
        return self._done_space

    @property
    def config(self):
        """
        Get the config of environment.

        Returns:
            config_dict(dict): A dictionary which contains environment's info.
        """
        return self._config

    def reset(self):
        """
        Reset the environment to the initial state. It is always used at the beginning of each
        episode. It will return the value of initial state or other initial information.

        Returns:
            - state (Tensor), a tensor which states for the initial state of environment.
            - other (Tensor), it will flatten the other output in _reset function, except for state.

        """
        if self.num_reset_out == 1:
            state = self.reset_ops()[0]
            return state
        if self.num_reset_out != 1:
            return self.reset_ops()

    def step(self, action):
        r"""
        Execute the environment step, which means that interact with environment once.

        Args:
            action (Tensor): A tensor that contains the action information.

        Returns:
            - state (Tensor), the environment state after performing the action.
            - reward (Tensor), the reward after performing the action.
            - done (Tensor), whether the simulation finishes or not.
            - other (Tensor), it will flatten the other output in _step function, except for state, reward and done
        """
        return self.step_ops(action)

    def close(self):
        r"""
        Close the environment to release the resource.

        Returns:
            - Success(np.bool\_), Whether shutdown the process or threading successfully.
        """
        return True

    def _reset_python(self):
        """Wrap the python reset function"""
        reset_out = None
        if self.num_reset_out == 1:
            state = self._reset()
            reset_out = (state,)
        else:
            state, *others = self._reset()
            reset_out = (state, *others)
        return reset_out

    def _step_python(self, action):
        """Wrap the python step function"""
        step_out = None
        if self.num_step_out == 3:
            next_state, reward, done = self._step(action)
            step_out = (next_state, reward, done)
        else:
            next_state, reward, done, *others = self._step(action)
            step_out = (next_state, reward, done, *others)
        return step_out

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
                    item_full_shape, item_split_shape, adapted_batch_shape = self._shape_adapter(adapted_item.shape)
                    batch_shape.append(adapted_batch_shape)
                    out_split_shape.append(item_split_shape)
                    out_full_shape.append(item_full_shape)
        else:
            adapted_result = result
            out_np_dtype.append(adapted_result.dtype.type)
            out_ms_dtype.append(mstype.pytype_to_dtype(adapted_result.dtype.type))
            result_full_shape, result_split_shape, adapted_batch_shape = self._shape_adapter(adapted_result.shape)
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

    def _reset(self):
        """
        The python reset function implementation. Each subclass needs to implement the reset function of environment.

        Returns:
            - state (np.ndarray), a numpy array which states for the initial state of environment.
            - *args (np.ndarray), the other outputs

        """
        raise NotImplementedError("Method should be overridden by subclass.")

    def _step(self, actions):
        """
        The python step function implementation. Each subclass needs to implement the step function of environment.

        Args:
            action (np.ndarray): A numpy array that contains the action information.

        Returns:
            - state (np.ndarray), the environment state after performing the action.
            - reward (np.ndarray), the reward after performing the action.
            - done (np.ndarray), whether the simulation finishes or not.
            - *args (np.ndarray), the other outputs
        """
        raise NotImplementedError("Method should be overridden by subclass.")

    def _get_action(self):
        r"""
        The python get\_action function implementation. Each subclass needs to implement the step
        function of environment.

        Returns:
            - action (np.ndarray), an available action of environment.
        """
        raise NotImplementedError("Method should be overridden by subclass.")

    def _get_min_max_action(self):
        r"""
        The python get\_min\_max\_action function implementation. For each subclass, they all need
        to implement this function

        Returns:
            - min_action (np.ndarray), the minimum value of action.
            - max_action (np.ndarray), the maximum value of action.
        """
        raise NotImplementedError("Method should be overridden by subclass.")

    def _get_min_max_observation(self):
        r"""
        The python get\_min\_max_observation function implementation. For each subclass, they all need
        to implement this function

        Returns:
            - min_observation (np.ndarray), the minimum value of observation.
            - max_observation (np.ndarray), the maximum value of observation.
        """
        raise NotImplementedError("Method should be overridden by subclass.")
