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
"""DeepMind Control Suite"""

# pylint: disable=W0223
# pylint: disable=C0415
import os
from concurrent.futures import ThreadPoolExecutor

import mindspore as ms
import numpy as np
from mindspore import context
from mindspore.ops import operations as P

from mindspore_rl.environment.environment import Environment
from mindspore_rl.environment.space import Space

if context.get_context("device_target") in ["GPU"]:
    os.environ["MUJOCO_GL"] = "egl"
else:
    os.environ["MUJOCO_GL"] = "osmesa"


class DeepMindControlEnvironment(Environment):
    """
    DeepMindControlEnvironment is a wrapper which encapsulates the DeepMind Control Suite(DMC).
    It stacks for physics-based simulation and Reinforcement Learning environments, using MUJOCO
    physics.

    Args:
        params (dict): A dictionary contains all the parameters which are used in this class.

            +------------------------------+----------------------------+
            |  Configuration Parameters    |  Notices                   |
            +==============================+============================+
            |  env_name                    |  the name of game in DMC   |
            +------------------------------+----------------------------+
            |  seed                        |  seed used in Gym          |
            +------------------------------+----------------------------+
            |  camera                      |  The camera pos used in    |
            |                              |  render                    |
            +------------------------------+----------------------------+
            |  action_repeat               |  How many times an action  |
            |                              |  interacts with env        |
            +------------------------------+----------------------------+
            |  normalize_action            |  Whether needs to normalize|
            |                              |  the input action          |
            +------------------------------+----------------------------+
            |  img_size                    |  The rendered img size     |
            +------------------------------+----------------------------+
        env_id (int, optional): A integer which is used to set the seed of this environment,
            default value means the 0th environment. Default: 0.

    Examples:
        >>> env_params = {'env_name': 'walker_walk', 'img_size': (64, 64),
                          'action_repeat': 2, 'normalize_action': True, 'seed': 1,
                          'episode_limits': 1000, 'prefill_value': 5000}
        >>> environment = DeepMindControlEnvironment(env_params, 0)
        >>> print(environment)
        DeepMindControlEnvironment<>
    """

    def __init__(self, params, env_id=0):
        super().__init__()
        env_name = params["env_name"]
        camera = params.get("camera", None)
        self._size = params["img_size"]
        seed = params["seed"] + env_id * 1000
        self._action_repeat = params["action_repeat"]
        self._normalize_action = params["normalize_action"]
        domain, task = env_name.split("_", 1)
        if domain == "cup":
            domain = "ball_in_cup"
        if isinstance(domain, str):
            from dm_control import suite

            self._env = suite.load(domain, task, task_kwargs={"random": seed})
        else:
            self._env = domain()
        if camera is None:
            camera = dict(quadruped=2).get(domain, 0)
        self._camera = camera

        action_spec = self._env.action_spec()
        low = action_spec.minimum
        high = action_spec.maximum
        self._mask = np.logical_and(np.isfinite(low), np.isfinite(high))
        if self._normalize_action:
            low = np.where(self._mask, low, -1)
            high = np.where(self._mask, high, 1)
        self._action_space = Space(
            action_spec.shape,
            self._dtype_adaptor(action_spec.dtype),
            low=low,
            high=high,
        )
        self.pool = ThreadPoolExecutor(max_workers=1)
        # get img
        demo_future = self.pool.submit(self._render, self._env)
        demo = demo_future.result()
        self._observation_space = Space(demo.shape, np.float32, low=0, high=255)
        self._reward_space = Space((1,), np.float32)
        self._done_space = Space((1,), np.bool_)

        # reset op
        reset_input_type = []
        reset_input_shape = []
        reset_output_type = [
            self._observation_space.ms_dtype,
        ]
        reset_output_shape = [
            self._observation_space.shape,
        ]
        self._reset_op = P.PyFunc(
            self._reset,
            reset_input_type,
            reset_input_shape,
            reset_output_type,
            reset_output_shape,
        )

        # step op
        step_input_type = (self._action_space.ms_dtype,)
        step_input_shape = (self._action_space.shape,)
        step_output_type = (
            self.observation_space.ms_dtype,
            self._reward_space.ms_dtype,
            self._done_space.ms_dtype,
            ms.float32,
        )
        step_output_shape = (
            self._observation_space.shape,
            self._reward_space.shape,
            self._done_space.shape,
            self._done_space.shape,
        )
        self._step_op = P.PyFunc(
            self._step,
            step_input_type,
            step_input_shape,
            step_output_type,
            step_output_shape,
        )

    @property
    def action_space(self):
        """
        Get the action space of the environment.

        Returns:
            Space, The action space of environment.
        """

        return self._action_space

    @property
    def config(self):
        """
        Get the config of environment.

        Returns:
            A dictionary which contains environment's info.
        """
        return {}

    @property
    def done_space(self):
        """
        Get the done space of the environment.

        Returns:
            Space, The done space of environment.
        """
        return self._done_space

    @property
    def reward_space(self):
        """
        Get the reward space of the environment.

        Returns:
            Space, The reward space of environment.
        """
        return self._reward_space

    @property
    def observation_space(self):
        """
        Get the state space of the environment.

        Returns:
            Space, The state space of environment.
        """

        return self._observation_space

    def close(self):
        r"""
        Close the environment to release the resource.

        Returns:
            Success(np.bool\_), Whether shutdown the process or threading successfully.
        """

        self._env.close()
        self.pool.shutdown()
        return True

    def reset(self):
        """
        Reset the environment to the initial state. It is always used at the beginning of each
        episode. It will return the value of initial state.

        Returns:
            A tensor which states for the initial state of environment.

        """

        return self._reset_op()[0]

    def step(self, action):
        r"""
        Execute the environment step, which means that interact with environment once.

        Args:
            action (Tensor): A tensor that contains the action information.

        Returns:
            - state (Tensor), the environment state after performing the action.
            - reward (Tensor), the reward after performing the action.
            - done (Tensor), whether the simulation finishes or not.
            - discount (Tensor), the discount value of env.
        """

        return self._step_op(action)

    def _step(self, action):
        """Python implementation of step"""
        low, high = self.action_space.boundary
        action = (
            np.where(self._mask, (action + 1) / 2 * (high - low) + low, action)
            if self._normalize_action
            else action
        )
        done = False
        total_reward = 0
        i = 0
        # do action repeat
        while i < self._action_repeat and not done:
            time_step = self._env.step(action)
            total_reward += time_step.reward
            done = time_step.last()
            i += 1
        obs_future = self.pool.submit(self._render, self._env)
        obs = obs_future.result()
        return (
            obs,
            total_reward.astype(np.float32),
            np.array(done),
            np.array(time_step.discount, np.float32),
        )

    def _reset(self):
        """Python implementation of reset"""
        self._env.reset()
        img_future = self.pool.submit(self._render, self._env)
        img = img_future.result()
        return img

    def _render(self, env):
        """Render function"""
        rendered_img = env.physics.render(*self._size, camera_id=self._camera)
        norm_img = rendered_img.astype(np.float32) / 255.0 - 0.5
        return norm_img

    def _dtype_adaptor(self, np_dtype):
        """dtype adaptor"""
        out_dtype = np_dtype
        if np_dtype == np.float64:
            out_dtype = np.float32
        elif np_dtype == np.int64:
            out_dtype = np.int32
        return out_dtype
