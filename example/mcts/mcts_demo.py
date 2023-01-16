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
"""VanillaMCTS demo"""

import mindspore as ms
import mindspore.nn as nn
from mindspore import GRAPH_MODE, context
from mindspore import Tensor
import mindspore.nn.probability.distribution as msd
from mindspore.ops import operations as P

from mindspore_rl.utils.mcts import MCTS, VanillaFunc
from mindspore_rl.environment import TicTacToeEnvironment


class VanillaMCTSWithTicTacToe(nn.Cell):
    """
    A VanillaMCTS demo. In this demo, MCTS plays with a random player in the Tic-Tac-Toe.
    """

    def __init__(self, uct, device):
        super().__init__()
        self.env = TicTacToeEnvironment(None)
        vanilla_func = VanillaFunc(self.env)
        uct = (Tensor(uct, ms.float32),)
        root_player = 1.0
        self.mcts = MCTS(self.env, "{}Common".format(device), "{}Vanilla".format(device), root_player, vanilla_func,
                         device, args=uct)

        self.false = Tensor(False, ms.bool_)

        self.ones_like = P.OnesLike()
        self.categorical = msd.Categorical()

    def run(self):
        """
        The run function of this demo.
        """
        done = self.false
        while not done:
            legal_action = self.env.legal_action()
            mask = (legal_action == -1)
            invalid_action_num = (legal_action == -1).sum()
            prob = self.ones_like(legal_action).astype(ms.float32) / (len(legal_action) - invalid_action_num)
            prob[mask] = 0
            opponent_action = self.categorical.sample((), prob)
            new_state, reward, done = self.env.step(legal_action[opponent_action])
            print("player 1 acts")
            print(new_state)
            if not done:
                ckpt = self.env.save()
                action, handle = self.mcts.mcts_search()
                self.mcts.restore_tree_data(handle)
                self.env.load(ckpt)
                new_state, reward, done = self.env.step(action[0])
                print("player 2 acts")
                print(new_state)
        self.mcts.destroy(handle)
        if reward[0] == 0:
            print("Draw")
        elif reward[0] == 1:
            print("Player 1 win")
        else:
            print("Player 2 win")


if __name__ == "__main__":
    DEVICE_TARGET = "CPU"
    context.set_context(mode=GRAPH_MODE, device_target=DEVICE_TARGET)
    vanilla_mcts = VanillaMCTSWithTicTacToe(2, DEVICE_TARGET)
    vanilla_mcts.run()
