# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
"""MPEMultiEnvironment class."""
#pylint: disable=E0402
#pylint: disable=W0212
import os
import numpy as np
from gym import spaces
from mindspore_rl.environment import Environment


def _prepare_mpe_env():
    '''prepare mpe env'''
    current_path = os.path.dirname(os.path.normpath(os.path.realpath(__file__)))
    os.chdir(current_path)
    # Clone mpe environment from marlbenchmark
    os.system('git clone https://github.com/marlbenchmark/on-policy.git')
    # Copy mpe folder to current directory
    os.system('cp -r on-policy/onpolicy/envs/mpe ./')
    # Download patch from mindspore_rl
    os.system(
        'wget https://gitee.com/mindspore/reinforcement/raw/master/example/mappo/src/mpe_environment.patch\
             --no-check-certificate')
    # patch mpe folder
    os.system('patch -p0 < mpe_environment.patch')


try:
    from .mpe.MPE_env import MPEEnv

except ModuleNotFoundError:
    _prepare_mpe_env()
    from .mpe.MPE_env import MPEEnv


class MultiAgentParticleEnvironment(Environment):
    """
    This is the wrapper of Multi-Agent Particle Environment(MPE) which is modified by MAPPO author from
    (https://github.com/marlbenchmark/on-policy/tree/main/onpolicy). A simple multi-agent particle world with
    a continuous observation and discrete action space, along with some basic simulated physics.
    Used in the paper Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments.

    Args:
        params (dict): A dictionary contains all the parameters which are used in this class.

            +------------------------------+--------------------------------------------------------+
            |  Configuration Parameters    |  Notices                                               |
            +==============================+========================================================+
            |  name                        |  Name of environment in MPEEnvironment, like           |
            |                              |  simple_spread                                         |
            |------------------------------+--------------------------------------------------------|
            |  num_agent                   |  Number of agent in mpe environment                    |
            +------------------------------|--------------------------------------------------------+
            |  auto_reset                  |  When the environment is done, whether call the reset  |
            |                              |  function automatically. Default: False                |
            +------------------------------|--------------------------------------------------------+
            |  seed                        |  The seed of mpe environment. Default: None            |
            +------------------------------|--------------------------------------------------------+
        env_id (int): A integer which is used to set the seed of this environment.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> env_params = {"name": simple_spread, "num_agent": 3, "auto_reset": False}
        >>> environment = MPEMultiEnvironment(env_params, 0)
        >>> print(environment)
    """

    def __init__(self,
                 params,
                 env_id=0):
        self.params = params
        self._env_name = params['name']
        self._num_agent = params['num_agent']
        self._auto_reset = params.get('auto_reset', False)

        class AllArgs:
            def __init__(self, env_name, episode, num_agent, num_landmark):
                self.episode_length = episode
                self.num_agents = num_agent
                self.num_landmarks = num_landmark
                self.scenario_name = env_name

        all_args = AllArgs(self._env_name, 25, self._num_agent, self._num_agent)

        self._env = MPEEnv(all_args)
        seed = params.get('seed')
        if seed:
            self._env.seed(seed + env_id * 1000)
        mpe_config = {'global_observation_dim': self._env.share_observation_space[0].shape[-1],
                      'num_agent': self._num_agent,
                      "episode_limit": self._env.world_length}

        super().__init__(env_name="MultiAgentParticleEnvironment", env=self._env, config=mpe_config)

    def _step(self, actions):
        """Inner step function implementation"""
        onehot_action = np.eye(self._env.action_space[0].n)[actions].squeeze(1)
        local_obs, rewards, done, _ = self._env._step(onehot_action)
        done = np.expand_dims(np.array(done), -1)
        if (self._auto_reset and done.all()):
            local_obs = self._reset()
        return np.array(local_obs, np.float32), np.array(rewards, np.float32), done

    def _reset(self):
        """Inner reset function implementation"""
        s0 = self._env._reset()
        return np.array(s0, np.float32)

    def _get_action(self):
        r"""Inner get\_action function implementation"""
        action = []
        for space in self._env.action_space:
            action.append(space.sample())
        return np.expand_dims(np.array(action, np.int32), -1)

    def _get_min_max_action(self):
        r"""Inner get\_min\_max\_action function implementation"""
        gym_space = self._env.action_space[0]
        if isinstance(gym_space, spaces.Discrete):
            return 0, gym_space.n
        return gym_space.low, gym_space.high

    def _get_min_max_observation(self):
        r"""Inner get\_min\_max\_observation function implementation"""
        space = self._env.observation_space[0]
        return space.low, space.high
