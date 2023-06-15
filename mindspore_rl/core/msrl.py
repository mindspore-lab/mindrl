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
import copy

# pylint: disable=R1719
import inspect
from functools import partial

from mindspore import nn
from mindspore.ops import operations as P

import mindspore_rl.distribution.distribution_policies as DP
from mindspore_rl.environment._remote_env_wrapper import _RemoteEnvWrapper
from mindspore_rl.environment.batch_wrapper import BatchWrapper
from mindspore_rl.environment.multi_environment_wrapper import MultiEnvironmentWrapper
from mindspore_rl.environment.pyfunc_wrapper import PyFuncWrapper


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
              - key: 'pass_environment', value: ``True`` user needs to pass the environment
                instance into actor, ``False`` otherwise (Bool).
    """

    def __init__(self, alg_config, deploy_config=None):
        # pylint: disable=R1725
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
            "eval_environment",
            "collect_environment",
            "policy_and_network",
            "actor",
            "learner",
        ]
        self._compulsory_items_check(alg_config, compulsory_items, "config")

        self.shared_network_str = None
        self.deploy_config = deploy_config
        if deploy_config is not None:
            # Need to compute the number of process per worker.
            self.proc_num = deploy_config.get("worker_num")
            self.distributed = True
            self.shared_network_str = deploy_config.get("network")
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
            actor_id (int): the id of the actor. Default: ``None``.

        Returns:
            obj (object), the class instance.
        """

        class_type = sub_config["type"]
        params = sub_config["params"]
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

    # pylint: disable=W0613
    @staticmethod
    def create_environments(
        config,
        env_type,
        deploy_config=None,
        need_batched=False,
    ):
        r"""
        Create the environments object from the configuration file, and return the instance
        of environment and evaluate environment.

        Args:
            config (dict): algorithm configuration file.
            env_type (str): type of environment in collect\_environment and eval\_environment.
            deploy_config (dict): the configuration for deploy. Default: None.
            need_batched (bool): whether to create batched environment. Default: False.

        Returns:
            - env (object), created environment object.
            - num_env (int), the number of environment.
        """
        support_remote_env = False
        if deploy_config:
            auto_dist = deploy_config.get("auto_distribution", False)
            dp = deploy_config.get("distribution_policy", None)
            is_dist_env = dp is DP.SingleActorLearnerMultiEnvDP
            support_remote_env = auto_dist and is_dist_env
        env_config = config[env_type]
        wrappers = copy.deepcopy(env_config.get("wrappers"))
        env_split = 1
        if support_remote_env:
            config[env_type]["params"]["_RemoteEnvWrapper"] = {
                "deploy_config": deploy_config
            }
            wrappers.insert(0, _RemoteEnvWrapper)
            env_split = deploy_config.get("worker_num") - 1

        num_env = env_config.get("number")
        num_parallel = (
            0
            if env_config.get("num_parallel") is None
            else env_config.get("num_parallel")
        )
        if (num_env % env_split != 0) or (num_parallel % env_split != 0):
            raise ValueError(
                "The number of environment and num_parallel should be divisible by the worker num."
            )
        num_env = num_env // env_split
        num_parallel = num_parallel // env_split

        env_creator = partial(
            config[env_type]["type"],
            config[env_type]["params"][config[env_type]["type"].__name__],
        )
        if need_batched:
            wrappers.insert(wrappers.index(PyFuncWrapper) + 1, BatchWrapper)
        if wrappers is not None:
            for wrapper in reversed(wrappers):
                wrapper_name = wrapper.__name__
                if wrapper_name == "SyncParallelWrapper":
                    env_creator = partial(
                        wrapper, [env_creator] * num_env, num_parallel
                    )
                elif wrapper_name == "BatchWrapper":
                    env_creator = partial(wrapper, [env_creator] * num_env)
                else:
                    if config[env_type]["params"].get(wrapper_name) is not None:
                        env_creator = partial(
                            wrapper,
                            env_creator,
                            **config[env_type]["params"][wrapper_name],
                        )
                    else:
                        env_creator = partial(
                            wrapper,
                            env_creator,
                        )
        env = env_creator()
        if env_config.get("seed") is not None:
            env.set_seed(env_config.get("seed"))
        return env, env.num_environment

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
                config[target]["params"][attr[0]] = attr[1]

    def __create_replay_buffer(self, replay_buffer_config):
        """
        Create the replay buffer object from the configuration file, and return the instance
        of replay buffer.

        Args:
            config (dict): the configuration for the replay buffer.

        Returns:
            replay_buffer (object), created replay buffer object.
        """

        num_replay_buffer = replay_buffer_config.get("number", 1)
        replay_buffer_type = replay_buffer_config["type"]

        params = replay_buffer_config.get("params", None)
        if not params:
            params = {
                "sample_size": replay_buffer_config.get("sample_size", 1),
                "capacity": replay_buffer_config.get("capacity"),
            }

        params["shapes"] = tuple(replay_buffer_config.get("data_shape"))
        params["types"] = tuple(replay_buffer_config.get("data_type"))

        if num_replay_buffer == 1:
            buffer = replay_buffer_type(**params)
        else:
            buffer = [replay_buffer_type(**params) for _ in range(num_replay_buffer)]
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
        policy_and_network_config = config["policy_and_network"]
        compulsory_items = ["type"]
        self._compulsory_items_check(
            policy_and_network_config, compulsory_items, "policy_and_network"
        )

        params = policy_and_network_config.get("params")
        collect_env = self.collect_environment
        if isinstance(collect_env, nn.CellList):
            collect_env = collect_env[0]
        if params:
            if not params.get("state_space_dim"):
                config["policy_and_network"]["params"][
                    "state_space_dim"
                ] = collect_env.observation_space.shape[-1]
            if not params.get("action_space_dim"):
                config["policy_and_network"]["params"][
                    "action_space_dim"
                ] = collect_env.action_space.num_values
            config["policy_and_network"]["params"][
                "environment_config"
            ] = collect_env.config

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
        compulsory_items = ["number", "type", "policies"]
        self._compulsory_items_check(config["actor"], compulsory_items, "actor")

        params = config["actor"].get("params")
        if not params:
            config["actor"]["params"] = {}
        config["actor"]["params"]["collect_environment"] = self.collect_environment
        config["actor"]["params"]["eval_environment"] = self.eval_environment

        config["actor"]["params"]["replay_buffer"] = self.buffers

        if config["actor"].get("policies"):
            self.__params_generate(config, policy_and_network, "actor", "policies")
        if config["actor"].get("networks"):
            self.__params_generate(config, policy_and_network, "actor", "networks")

        self.num_actors = config["actor"]["number"]
        actor = self._create_instance(config["actor"], actor_id)
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
        compulsory_items = ["type", "networks"]
        self._compulsory_items_check(config["learner"], compulsory_items, "learner")

        params = config["learner"].get("params")
        if not params:
            config["learner"]["params"] = {}
        if config["learner"].get("networks"):
            self.__params_generate(config, policy_and_network, "learner", "networks")

        num_learner = config["learner"]["number"]
        if num_learner == 1:
            learner = self._create_instance(config["learner"])
        else:
            raise ValueError("Sorry, the current version only supports one learner !")
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
        replay_buffer = config.get("replay_buffer")
        if replay_buffer:
            if replay_buffer.get("multi_type_replaybuffer"):
                self.buffers = {}
                for key, item in replay_buffer.items():
                    if key != "multi_type_replaybuffer":
                        self.buffers[key] = self.__create_replay_buffer(item)
            else:
                self.buffers = self.__create_replay_buffer(replay_buffer)
                if replay_buffer.get("number") <= 1:
                    self.replay_buffer_sample = self.buffers.sample
                    self.replay_buffer_insert = self.buffers.insert
                    self.replay_buffer_full = self.buffers.full
                    self.replay_buffer_reset = self.buffers.reset

        # ---------------------- Agent ----------------------
        agent_config = config.get("agent")
        if not agent_config:
            self._compulsory_items_check(config["actor"], ["number"], "actor")
            num_actors = config["actor"]["number"]
            # We consider eval_env is alwarys shared, so only create one instance whether in multi-actor or not.
            share_env = True
            if "share_env" in config["actor"]:
                share_env = config["actor"]["share_env"]
            # ---------------------- Environment ----------------------
            self.collect_environment, self.num_collect_env = MSRL.create_environments(
                config, "collect_environment", deploy_config=self.deploy_config
            )
            need_batched = True if (self.num_collect_env > 1) else False
            self.eval_environment, _ = MSRL.create_environments(
                config,
                "eval_environment",
                need_batched=need_batched,
            )
            # ---------------------------------------------------------
            if self.distributed:
                self.policy_and_network = self.__create_policy_and_network(config)
                self.actors = self.__create_actor(config, self.policy_and_network)
                self.learner = self.__create_learner(config, self.policy_and_network)
                self.agent_act = self.actors.act
                self.agent_learn = self.learner.learn
            else:
                if num_actors == 1:
                    self.policy_and_network = self.__create_policy_and_network(config)
                    self.actors = self.__create_actor(config, self.policy_and_network)
                    self.learner = self.__create_learner(
                        config, self.policy_and_network
                    )
                    self.agent_act = self.actors.act
                    self.agent_learn = self.learner.learn
                    self.agent_get_action = self.actors.get_action
                elif num_actors > 1:
                    self.actors = nn.CellList()
                    if not share_env:
                        self.collect_environment = nn.CellList()
                    for i in range(num_actors):
                        if not share_env:
                            self.collect_environment.append(
                                MSRL.create_environments(
                                    config,
                                    "collect_environment",
                                    deploy_config=self.deploy_config,
                                )[0]
                            )
                        self.policy_and_network = self.__create_policy_and_network(
                            config
                        )
                        self.actors.append(
                            self.__create_actor(
                                config, self.policy_and_network, actor_id=i
                            )
                        )
                    self.learner = self.__create_learner(
                        config, self.policy_and_network
                    )
                    self.agent_learn = self.learner.learn
                else:
                    raise ValueError(
                        "The number of actors should >= 1, but get ", num_actors
                    )
        else:
            compulsory_items = ["number", "type"]
            self._compulsory_items_check(agent_config, compulsory_items, "agent")
            agent_type = agent_config["type"]
            self.num_agent = agent_config["number"]
            params = agent_config.get("params")
            if not params:
                config["agent"]["params"] = {}

            config["agent"]["params"]["num_agent"] = self.num_agent
            # ---------------------- Environment ----------------------
            self.collect_environment, self.num_collect_env = MSRL.create_environments(
                config, "collect_environment", deploy_config=self.deploy_config
            )
            need_batched = True if (self.num_collect_env > 1) else False
            self.eval_environment, _ = MSRL.create_environments(
                config,
                "eval_environment",
                need_batched=need_batched,
            )
            # ---------------------------------------------------------
            for i in range(self.num_agent):
                policy_and_network = self.__create_policy_and_network(config)
                self.agent.append(
                    agent_type(
                        self.__create_actor(config, policy_and_network),
                        self.__create_learner(config, policy_and_network),
                    )
                )
            self.agent = nn.CellList(self.agent)
        if self.shared_network_str:
            # pylint: disable=W0123
            self.shared_network = eval(
                "self.policy_and_network." + self.shared_network_str
            )

    def get_replay_buffer(self):
        """
        It will return the instance of replay buffer.

        Returns:
            Buffers (object), The instance of relay buffer. If the buffer is ``None``, the return
            value will be ``None``.
        """

        return self.buffers

    def get_replay_buffer_elements(self, transpose=False, shape=None):
        """
        It will return all the elements in the replay buffer.

        Args:
            transpose (bool): whether the output element needs to be transpose,
                if `transpose` is ``True``, `shape` will also need to be filled. Default: ``False``.
            shape (tuple[int]): the shape used in transpose. Default: ``None``.

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
