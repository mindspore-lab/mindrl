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
Implementation of MSRL class.
"""

import inspect
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore_rl.environment.multi_environment_wrapper import MultiEnvironmentWrapper


class MSRL(nn.Cell):
    """
    The MSRL class provides the function handlers and APIs for reinforcement
    learning algorithm development.

    It exposes the following function handler to the user. The input and output of
    these function handlers are identical to the user defined functions.

    .. code-block::

        agent_act
        agent_get_action
        sample_buffer
        agent_learn
        replay_buffer_sample
        replay_buffer_insert
        replay_buffer_reset

    Args:
        alg_config(dict): provides the algorithm configuration.
        deploy_config(dict): provides the distribute configuration.

            - Top level: defines the algorithm components.

              - key: 'actor',              value: the actor configuration (dict).
              - key: 'learner',            value: the learner configuration (dict).
              - key: 'policy_and_network', value: the policy and networks used by
                actor and learner (dict).
              - key: 'collect_environment',     value: the collect environment configuration (dict).
              - key: 'eval_environment',        value: the eval environment configuration (dict).
              - key: 'replay_buffer',           value: the replay buffer configuration (dict).

            - Second level: the configuration of each algorithm component.

              - key: 'number',      value: the number of actor/learner (int).
              - key: 'type',        value: the type of the
                actor/learner/policy_and_network/environment (class name).
              - key: 'params',      value: the parameters of
                actor/learner/policy_and_network/environment (dict).
              - key: 'policies',    value: the list of policies used by the
                actor/learner (list).
              - key: 'networks',    value: the list of networks used by the
                actor/learner (list).
              - key: 'pass_environment', value: True user needs to pass the environment
                instance into actor, False otherwise (Bool).
    """

    def __init__(self, alg_config, deploy_config=None):
        super(MSRL, self).__init__()
        self.actors = []
        self.learner = None
        self.envs = []
        self.agent = []
        self.buffers = None
        self.collect_environment = None
        self.eval_environment = None
        self.num_collect_env = None
        self.num_actors = None
        self.distributed = False

        # apis
        self.agent_act = None
        self.agent_learn = None
        self.replay_buffer_sample = None
        self.replay_buffer_insert = None
        self.replay_buffer_full = None
        self.replay_buffer_reset = None

        compulsory_items = [
            'eval_environment', 'collect_environment', 'policy_and_network',
            'actor', 'learner'
        ]
        self._compulsory_items_check(alg_config, compulsory_items, 'config')

        if deploy_config is not None:
            # Need to compute the number of process per worker.
            self.proc_num = deploy_config['worker_num']
            self.distributed = True
        self.init(alg_config)

    def _compulsory_items_check(self, config, compulsory_item, position):
        for item in compulsory_item:
            if item not in config:
                raise ValueError(
                    f"The `{item}` configuration in `{position}` should be provided."
                )

    def _create_instance(self, sub_config, actor_id=None):
        """
        Create class object from the configuration file, and return the instance of 'type' in
        input sub_config.

        Args:
            sub_config (dict): configuration file of the class.
            actor_id (int): the id of the actor. Default: None.

        Returns:
            obj (object), the class instance.
        """

        class_type = sub_config['type']
        params = sub_config['params']
        if actor_id is None:
            obj = class_type(params)
        else:
            obj = class_type(params, actor_id)
        return obj

    def _create_batch_env(self, sub_config, env_num, proc_num):
        """
        Create the batch environments object from the sub_config,
        and return the instance of a batch env.

        Args:
            sub_config (dict): algorithm config of env.
            env_num (int): number of environment to be created.
            proc_num (int): the process for environment.

        Returns:
            batch_env (object), the created batch-environment object.
        """
        env_list = []
        for i in range(env_num):
            env_list.append(self._create_instance(sub_config, i))
        return MultiEnvironmentWrapper(env_list, proc_num)

    def __create_environments(self, config, num_agent=1):
        """
        Create the environments object from the configuration file, and return the instance
        of environment and evaluate environment.

        Args:
            config (dict): algorithm configuration file.

        Returns:
            - env (object), created environment object.
            - eval_env (object), created evaluate environment object.
        """

        collect_env_config = config['collect_environment']
        eval_env_config = config['eval_environment']
        self.num_collect_env = collect_env_config.get('number')
        num_eval_env = eval_env_config.get('number')
        collect_num_parallel = collect_env_config.get('num_parallel')
        eval_num_parallel = eval_env_config.get('num_parallel')

        if self.num_collect_env is None:
            self.num_collect_env = 1
        if num_eval_env is None:
            num_eval_env = 1

        if collect_num_parallel:
            if collect_num_parallel < 1:
                raise ValueError("num_parallel of collect_environment can not be non-positive")
            collect_proc_num = collect_num_parallel
        else:
            collect_proc_num = self.num_collect_env

        if eval_num_parallel:
            if eval_num_parallel < 1:
                raise ValueError("num_parallel of eval_environment can not be non-positive")
            eval_proc_num = eval_num_parallel
        else:
            eval_proc_num = num_eval_env

        compulsory_item = ['type']
        self._compulsory_items_check(collect_env_config, compulsory_item, 'collect_environment')
        self._compulsory_items_check(eval_env_config, compulsory_item, 'eval_environment')

        if not config['collect_environment'].get('params'):
            config['collect_environment']['params'] = {}
        if not config['eval_environment'].get('params'):
            config['eval_environment']['params'] = {}

        if self.num_collect_env > 1:
            collect_env = self._create_batch_env(config['collect_environment'], self.num_collect_env, collect_proc_num)
            eval_env = self._create_batch_env(config['eval_environment'], num_eval_env, eval_proc_num)
        else:
            collect_env = self._create_instance(config['collect_environment'], None)
            if num_eval_env > 1:
                collect_env = MultiEnvironmentWrapper([collect_env], collect_proc_num)
                eval_env = self._create_batch_env(config['eval_environment'], num_eval_env, eval_proc_num)
            else:
                eval_env = self._create_instance(config['eval_environment'], None)

        return collect_env, eval_env

    def __params_generate(self, config, obj, target, attribute):
        """
        Parse the input object to generate parameters, then store the parameters into
        the dictionary of configuration.

        Args:
            config (dict): the algorithm configuration.
            obj (object): the object for analysis.
            target (str): the name of the target class.
            attribute (str): the name of the attribute to parse.

        """

        for attr in inspect.getmembers(obj):
            if attr[0] in config[target][attribute]:
                config[target]['params'][attr[0]] = attr[1]

    def __create_replay_buffer(self, replay_buffer_config):
        """
        Create the replay buffer object from the configuration file, and return the instance
        of replay buffer.

        Args:
            config (dict): the configuration for the replay buffer.

        Returns:
            replay_buffer (object), created replay buffer object.
        """
        compulsory_item = ['type', 'capacity', 'data_shape', 'data_type']
        self._compulsory_items_check(replay_buffer_config, compulsory_item,
                                     'replay_buffer')

        num_replay_buffer = replay_buffer_config.get('number')
        if num_replay_buffer is None:
            num_replay_buffer = 1
        replay_buffer_type = replay_buffer_config['type']
        capacity = replay_buffer_config['capacity']
        buffer_data_shapes = replay_buffer_config['data_shape']
        buffer_data_type = replay_buffer_config['data_type']

        sample_size = replay_buffer_config.get('sample_size')
        if not sample_size:
            sample_size = 1

        if num_replay_buffer == 1:
            buffer = replay_buffer_type(sample_size, capacity,
                                        buffer_data_shapes, buffer_data_type)
        else:
            buffer = [
                replay_buffer_type(sample_size, capacity, buffer_data_shapes,
                                   buffer_data_type) for _ in range(num_replay_buffer)
            ]
            buffer = nn.CellList(buffer)
        return buffer

    def __create_policy_and_network(self, config):
        """
        Create an instance of XXXPolicy class in algorithm, it contains the networks. collect policy
        and eval policy of algorithm.

        Args:
            config (dict): A dictionary of configuration.

        Returns:
            policy_and_network (object): The instance of policy and network.
        """
        policy_and_network_config = config['policy_and_network']
        compulsory_items = ['type']
        self._compulsory_items_check(policy_and_network_config,
                                     compulsory_items, 'policy_and_network')

        params = policy_and_network_config.get('params')
        collect_env = self.collect_environment
        if isinstance(collect_env, nn.CellList):
            collect_env = collect_env[0]
        if params:
            if not params.get('state_space_dim'):
                config['policy_and_network']['params'][
                    'state_space_dim'] = collect_env.observation_space.shape[-1]
            if not params.get('action_space_dim'):
                config['policy_and_network']['params'][
                    'action_space_dim'] = collect_env.action_space.num_values
            config['policy_and_network']['params']['environment_config'] = collect_env.config

        policy_and_network = self._create_instance(policy_and_network_config)
        return policy_and_network

    def __create_actor(self, config, policy_and_network, actor_id=None):
        """
        Create an instance of actor or a list of instances of actor.

        Args:
            config (dict): A dictionary of configuration.
            policy_and_network (object): The instance of policy_and_network.
            actor_id (int): The number of the actors. Default: None.

        Returns:
            actor (object or List(object)): An instance of actor a list of instances of actor
        """
        compulsory_items = ['number', 'type', 'policies']
        self._compulsory_items_check(config['actor'], compulsory_items,
                                     'actor')

        params = config['actor'].get('params')
        if not params:
            config['actor']['params'] = {}
        config['actor']['params'][
            'collect_environment'] = self.collect_environment
        config['actor']['params'][
            'eval_environment'] = self.eval_environment

        config['actor']['params']['replay_buffer'] = self.buffers

        if config['actor'].get('policies'):
            self.__params_generate(config, policy_and_network, 'actor',
                                   'policies')
        if config['actor'].get('networks'):
            self.__params_generate(config, policy_and_network, 'actor',
                                   'networks')

        self.num_actors = config['actor']['number']
        actor = self._create_instance(config['actor'], actor_id)
        return actor

    def __create_learner(self, config, policy_and_network):
        """
        Create an instance of learner or a list of instances of learner.

        Args:
            config (dict): A dictionary of configuration.
            policy_and_network (object): The instance of policy_and_network.

        Returns:
            actor (object or List(object)): An instance of learner a list of instances of learner.
        """
        compulsory_items = ['type', 'networks']
        self._compulsory_items_check(config['learner'], compulsory_items,
                                     'learner')

        params = config['learner'].get('params')
        if not params:
            config['learner']['params'] = {}
        if config['learner'].get('networks'):
            self.__params_generate(config, policy_and_network, 'learner',
                                   'networks')

        num_learner = config['learner']['number']
        if num_learner == 1:
            learner = self._create_instance(config['learner'])
        else:
            raise ValueError(
                "Sorry, the current version only supports one learner !")
        return learner

    def init(self, config):
        """
        Initialization of MSRL object.
        The function creates all the data/objects that the algorithm requires.
        It also initializes all the function handler.

        Args:
            config (dict): algorithm configuration file.
        """
        # ---------------------- ReplayBuffer ----------------------
        replay_buffer = config.get('replay_buffer')
        if replay_buffer:
            if replay_buffer.get("multi_type_replaybuffer"):
                self.buffers = {}
                for key, item in replay_buffer.items():
                    if key != "multi_type_replaybuffer":
                        self.buffers[key] = self.__create_replay_buffer(item)
            else:
                self.buffers = self.__create_replay_buffer(replay_buffer)
                if replay_buffer.get('number') <= 1:
                    self.replay_buffer_sample = self.buffers.sample
                    self.replay_buffer_insert = self.buffers.insert
                    self.replay_buffer_full = self.buffers.full
                    self.replay_buffer_reset = self.buffers.reset

        # ---------------------- Agent ----------------------
        agent_config = config.get('agent')
        if not agent_config:
            self._compulsory_items_check(config['actor'], ['number'], 'actor')
            num_actors = config['actor']['number']
            # We consider eval_env is alwarys shared, so only create one instance whether in multi-actor or not.
            share_env = True
            if 'share_env' in config['actor']:
                share_env = config['actor']['share_env']
            # ---------------------- Environment ----------------------
            self.collect_environment, self.eval_environment = self.__create_environments(config)
            # ---------------------------------------------------------
            if self.distributed:
                self.policy_and_network = self.__create_policy_and_network(config)
                self.actors = self.__create_actor(config, self.policy_and_network)
                self.learner = self.__create_learner(config, self.policy_and_network)
                self.agent_act = self.actors.act
                self.agent_learn = self.learner.learn
            else:
                if num_actors == 1:
                    policy_and_network = self.__create_policy_and_network(config)
                    self.actors = self.__create_actor(config, policy_and_network)
                    self.learner = self.__create_learner(config, policy_and_network)
                    self.agent_act = self.actors.act
                    self.agent_learn = self.learner.learn
                    self.agent_get_action = self.actors.get_action
                elif num_actors > 1:
                    self.actors = nn.CellList()
                    if not share_env:
                        self.collect_environment = nn.CellList()
                    for i in range(num_actors):
                        if not share_env:
                            self.collect_environment.append(self.__create_environments(config)[0])
                        policy_and_network = self.__create_policy_and_network(config)
                        self.actors.append(self.__create_actor(config, policy_and_network, actor_id=i))
                    self.learner = self.__create_learner(config, policy_and_network)
                    self.agent_learn = self.learner.learn
                else:
                    raise ValueError("The number of actors should >= 1, but get ", num_actors)
        else:
            compulsory_items = ['number', 'type']
            self._compulsory_items_check(agent_config, compulsory_items, 'agent')
            agent_type = agent_config['type']
            self.num_agent = agent_config['number']
            params = agent_config.get('params')
            if not params:
                config['agent']['params'] = {}

            config['agent']['params']['num_agent'] = self.num_agent
            # ---------------------- Environment ----------------------
            self.collect_environment, self.eval_environment = self.__create_environments(
                config, self.num_agent)
            # ---------------------------------------------------------
            for i in range(self.num_agent):
                policy_and_network = self.__create_policy_and_network(config)
                self.agent.append(agent_type(self.__create_actor(config, policy_and_network),
                                             self.__create_learner(config, policy_and_network)))
            self.agent = nn.CellList(self.agent)

    def get_replay_buffer(self):
        """
        It will return the instance of replay buffer.

        Returns:
            Buffers (object), The instance of relay buffer. If the buffer is None, the return
            value will be None.
        """

        return self.buffers

    def get_replay_buffer_elements(self, transpose=False, shape=None):
        """
        It will return all the elements in the replay buffer.

        Args:
            transpose (bool): whether the output element needs to be transpose,
                if transpose is true, shape will also need to be filled. Default: False.
            shape (tuple[int]): the shape used in transpose. Default: None.

        Returns:
            elements (List[Tensor]), A set of tensor contains all the elements in the replay buffer.
        """

        transpose_op = P.Transpose()
        elements = ()
        for e in self.buffers.buffer:
            if transpose:
                e = transpose_op(e, shape)
                elements += (e,)
            else:
                elements += (e,)

        return elements
