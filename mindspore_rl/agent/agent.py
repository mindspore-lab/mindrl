# Copyright 2021 Huawei Technologies Co., Ltd
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
Implementation of Agent base class.
"""

import mindspore.nn as nn


class Agent(nn.Cell):
    r"""
    The base class for the Agent. As the definition of agent, it is composed of actor and leanrner.
    It has basic act and learn functions for interaction with environment and update itself.

    Args:
        actors(Actor): The actor instance.
        learner(Learner): The learner instance.

    Examples:
        >>> from mindspore_rl.agent.learner import Learner
        >>> from mindspore_rl.agent.actor import Actor
        >>> from mindspore_rl.agent.agent import Agent
        >>> actors = Actor()
        >>> learner = Learner()
        >>> agent = Agent(actors, learner)
        >>> print(agent)
        Agent<
        (_actors): Actor<>
        (_learner): Learner<>
        >
    """

    def __init__(self, actors, learner):
        super(Agent, self).__init__(auto_prefix=False)
        self._actors = actors
        self._learner = learner

    def get_action(self, phase, params):
        """
        The get_action function will take an enumerate value and observation or other data which is needed during
        calculating the action. It will return a set of outputs containing actions and other data. In this
        function, agent will not interact with environment.

        Args:
            phase (enum): A enumerate value states for init, collect or eval stage.
            params (tuple(Tensor)): A tuple of tensor as input, which is used to calculate action

        Returns:
            tuple(Tensor), a tuple of tensor as output, containing actions and other data.
        """

        raise NotImplementedError("Method should be overridden by subclass.")

    def act(self, phase, params):
        """
        The act function will take an enumerate value and observation or other data which is needed during
        calculating the action. It will return a set of output which contains new observation, or other
        experience. In this function, agent will interact with environment.

        Args:
            phase (enum): A enumerate value states for init, collect or eval stage.
            params (tuple(Tensor)): A tuple of tensor as input, which is used to calculate action

        Returns:
            tuple(Tensor), a tuple of tensor as output, which states for experience
        """

        raise NotImplementedError("Method should be overridden by subclass.")

    def learn(self, experience):
        """
        The learn function will take a set of experience as input to calculate the loss and update
        the weights.

        Args:
            experience (tuple(Tensor)): A tuple of tensor states for experience

        Returns:
            tuple(Tensor), result which outputs after updating weights
        """

        raise NotImplementedError("Method should be overridden by subclass.")
