# Copyright 2022 Huawei Technologies Co., Ltd
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
"""Tic-Tac-Toe game"""
#pylint: disable=C0325
import numpy as np

import mindspore as ms
from mindspore.ops import operations as P
from mindspore_rl.environment import Environment
from mindspore_rl.environment import Space


class TicTacToeEnvironment(Environment):
    """
    Tic-Tac-Toe is a famous paper-and-pencil game (https://en.wikipedia.org/wiki/Tic-tac-toe). The rule is that two
    players draw Os or Xs in a three-by-tree grid. When three of their marks are in a Horizontal, vertical or diagonal
    row, that player will be the winner. The following figure is an example of Tic-Tac-Toe.

    +---+---+---+
    | o |   | x |
    +---+---+---+
    | x | o |   |
    +---+---+---+
    |   | x | o |
    +---+---+---+

    Args:
        params (dict): A dictionary contains all the parameters which are used in this class.
        env_id (int): A integer which is used to set the seed of this environment. Default: 0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore_rl.environment import TicTacToeEnvironment
        >>> env_params = {}
        >>> environment = TicTacToeEnvironment(env_params, 0)
        >>> print(environment)
        TicTacToeEnvironment<>
    """

    def __init__(self, params, env_id=0):
        super().__init__()

        self._board = np.zeros((3, 3), np.float32)
        self._current_player_var = 0
        self._total_num_player = 2.0
        self._avail_action = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], np.int32)
        self._max_utility = 1.0
        self.params = params
        self.env_id = env_id

        self._player_one_win = np.array([1.0, -1.0], np.float32)
        self._player_two_win = np.array([-1.0, 1.0], np.float32)
        self._draw_or_no_result = np.array([0, 0], np.float32)
        self._done = np.array([False], np.bool_)

        self._observation_space = Space((3, 3), np.float32, low=-1, high=2)
        self._action_space = Space((1,), np.int32, low=0, high=9)
        self._reward_space = Space((2,), np.float32, low=-1, high=2)
        self._done_space = Space((1,), np.bool_)

        self._reset_ops = P.PyFunc(self._reset, (), (), (self._observation_space.ms_dtype,),
                                   (self._observation_space.shape,))
        step_out_dtype = (self._observation_space.ms_dtype, self._reward_space.ms_dtype, self._done_space.ms_dtype)
        step_out_shape = (self._observation_space.shape, self._reward_space.shape, self._done_space.shape)
        self._step_ops = P.PyFunc(self._step, (self._action_space.ms_dtype,),
                                  (self._action_space.shape,), step_out_dtype, step_out_shape)
        self._save_ops = P.PyFunc(self._save, (), (), (self._observation_space.ms_dtype,),
                                  (self._observation_space.shape,))
        self._load_ops = P.PyFunc(self._load, (self._observation_space.ms_dtype,),
                                  (self._observation_space.shape,), step_out_dtype, step_out_shape)
        self._legal_action_ops = P.PyFunc(self._legal_action, (), (), (ms.int32,), ((9,),))
        self._current_player_ops = P.PyFunc(self._current_player, (), (), (ms.int32,), ((1,),))
        self._is_terminal_ops = P.PyFunc(self._is_terminal, (), (), (ms.bool_,), ((1,),))
        self._reward_ops = P.PyFunc(self._rewards, (), (), (ms.float32,), ((2,),))

    @property
    def action_space(self):
        """
        Get the action space of the environment.

        Returns:
            The action space of environment.
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
            The done space of environment.
        """
        return self._done_space

    @property
    def observation_space(self):
        """
        Get the state space of the environment.

        Returns:
            The state space of environment.
        """

        return self._observation_space

    @property
    def reward_space(self):
        """
        Get the reward space of the environment.

        Returns:
            The reward space of environment.
        """
        return self._reward_space

    def reset(self):
        """
        Reset the environment to the initial state. It is always used at the beginning of each
        episode. It will return the value of initial state.

        Returns:
            A Tensor which states for initial state.

        """
        return self._reset_ops()[0]

    def step(self, action):
        r"""
        Execute the environment step, which means that interact with environment once.

        Args:
            action (Tensor): A tensor that contains the action information.

        Returns:
            - state (Tensor), the environment state after performing the action.
            - reward (Tensor), the reward after performing the action.
            - done (Tensor), whether the simulation finishes or not.
        """
        return self._step_ops(action)

    def save(self):
        """
        Return a repilca of environment. Tic-Tac-Toe do not need a replica, thus it will return the current
        state

        Returns:
            A tensor which states for the current state.
        """
        return self._save_ops()[0]

    def load(self, state):
        """
        Load the input state. It will update the legal action, current state and done info of the game to the
        input checkpoint.

        Args:
            state (Tensor): The input checkpoint state.

        Returns:
            - state (Tensor), the state of checkpoint.
            - reward (Tensor), the reward of checkpoint.
            - done (Tensor), whether the checkpoint is terminal.
        """
        return self._load_ops(state)

    def calculate_rewards(self):
        """
        Return the rewards of current state.

        Returns:
            A tensor which states for the rewards of current state.
        """
        return self._rewards_ops()[0]

    def legal_action(self):
        """
        Return the legal action of current state.

        Returns:
            A tensor which states for the legal action.
        """
        return self._legal_action_ops()[0]

    def max_utility(self):
        """
        Return the max utility of Tic-Tac-Toe.

        Returns:
            A tensor which states for max utility.
        """
        return self._max_utility

    def total_num_player(self):
        """
        Return the total number of player

        Returns:
            int, the total number of player.
        """
        return self._total_num_player

    def current_player(self):
        """
        Return the current player of current state.

        Returns:
            A tensor which states for current player.
        """
        return self._current_player_ops()[0][0]

    def is_terminal(self):
        """
        Return whether the current state is terminal.

        Returns:
            whether the current state is terminal or not.
        """
        return self._is_terminal_ops()[0]

    def _reset(self):
        """private reset function"""
        self._board = np.zeros_like(self._board)
        return self._board

    def _step(self, action):
        """private step function"""
        action = action[0]
        if not action in self._avail_action or action == -1:
            raise ValueError("action {} is not available, please check the input of step function".format(action))
        self._avail_action[action] = -1
        row, column = np.divmod(action, 3)
        if self._current_player_var == 0:
            self._board[row][column] = 1
        else:
            self._board[row][column] = -1
        self._current_player_var = 1 - self._current_player_var
        reward = self._rewards()
        if reward[0] != 0 or self._avail_action.sum() == -9:
            self._done = np.array([True], np.bool_)
        return self._board, reward, self._done

    def _save(self):
        """private save function"""
        return self._board

    def _load(self, state):
        """private load function"""
        self._board = state
        new_avail = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], np.int32)
        for row in range(3):
            for column in range(3):
                if state[row][column] != 0:
                    new_avail[row*3+column] = -1
        out_reward = self._rewards()
        if (new_avail.sum() == -9) or (out_reward == self._player_one_win).all() \
                or (out_reward == self._player_two_win).all():
            self._done = np.array([True], np.bool_)
        else:
            self._done = np.array([False], np.bool_)
        if state.sum() == 0:
            self._current_player_var = 0
        else:
            self._current_player_var = 1
        self._avail_action = new_avail
        return self._board, out_reward, self._done

    def _legal_action(self):
        """private legal action function"""
        return self._avail_action

    def _current_player(self):
        """private current player function"""
        return np.array([self._current_player_var], np.int32)

    def _is_terminal(self):
        """private is terminal function"""
        return self._done

    def _rewards(self):
        """private rewards function"""
        if (self._board[0].sum() == 3) or (self._board[1].sum() == 3) or (self._board[2].sum() == 3):
            return self._player_one_win
        if (self._board[0].sum() == -3) or (self._board[1].sum() == -3) or (self._board[2].sum() == -3):
            return self._player_two_win
        for column in range(3):
            if (self._board[0][column] + self._board[1][column] + self._board[2][column] == 3):
                return self._player_one_win
            if (self._board[0][column] + self._board[1][column] + self._board[2][column] == -3):
                return self._player_two_win
        cross_one = self._board[0][0] + self._board[1][1] + self._board[2][2]
        cross_two = self._board[0][2] + self._board[1][1] + self._board[2][0]
        if cross_one == 3 or cross_two == 3:
            return self._player_one_win
        if cross_one == -3 or cross_two == -3:
            return self._player_two_win
        return self._draw_or_no_result
