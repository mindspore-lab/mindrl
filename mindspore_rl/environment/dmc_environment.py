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
# pylint: disable=W0221
import os
import queue
import threading

import numpy as np
from mindspore import context

from mindspore_rl.environment.python_environment import PythonEnvironment
from mindspore_rl.environment.space import Space
from mindspore_rl.environment.space_adapter import dmc2ms_adapter

if context.get_context("device_target") in ["GPU"]:
    os.environ["MUJOCO_GL"] = "egl"
else:
    os.environ["MUJOCO_GL"] = "osmesa"


class DeepMindControlEnvironment(PythonEnvironment):
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
            default value means the 0th environment. Default: ``0`` .

    Examples:
        >>> env_params = {'env_name': 'walker_walk', 'img_size': (64, 64),
                          'action_repeat': 2, 'normalize_action': True, 'seed': 1,
                          'episode_limits': 1000, 'prefill_value': 5000}
        >>> environment = DeepMindControlEnvironment(env_params, 0)
        >>> print(environment)
        DeepMindControlEnvironment<>
    """

    def __init__(self, params, env_id=0):
        env_name = params["env_name"]
        camera = params.get("camera", None)
        self._size = params["img_size"]
        seed = params["seed"] + env_id * 1000
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

        self._env_queue = queue.Queue(maxsize=1)
        self._img_queue = queue.Queue(maxsize=1)
        self._thread = threading.Thread(target=self._render_threading)
        self._thread.start()
        self._env_queue.put(self._env)
        demo = self._img_queue.get()
        action_space = dmc2ms_adapter(self._env.action_spec())
        observation_space = Space(demo.shape, np.float32, low=0, high=255)
        super().__init__(action_space=action_space, observation_space=observation_space)

    def close(self):
        r"""
        Close the environment to release the resource.

        Returns:
            Success(np.bool\_), Whether shutdown the process or threading successfully.
        """

        self._env.close()
        self._env_queue.put(None)
        return True

    def _step(self, action):
        """Python implementation of step"""
        time_step = self._env.step(action)
        reward = time_step.reward
        done = time_step.last()
        self._env_queue.put(self._env)
        obs = self._img_queue.get()
        obs = obs.astype(np.float32) / 255.0 - 0.5
        return (
            obs,
            np.array(reward, np.float32),
            np.array(done),
            np.array(time_step.discount, np.float32),
        )

    def _reset(self):
        """Python implementation of reset"""
        self._env.reset()
        self._env_queue.put(self._env)
        img = self._img_queue.get()
        norm_img = img.astype(np.float32) / 255.0 - 0.5
        return norm_img

    def _render_threading(self):
        """Render function"""
        while True:
            env = self._env_queue.get()
            if env is None:
                break
            rendered_img = env.physics.render(*self._size, camera_id=self._camera)
            self._img_queue.put(rendered_img)

    def _set_seed(self, seed_value: int) -> bool:
        """Inner set seed function"""
        raise ValueError(
            "DeepMindControlEnvironment does not support set_seed function, please use seed in params."
        )
