# Copyright 2022-2023 Huawei Technologies Co., Ltd
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
"""MonteCarloTreeSearch Class"""

# pylint: disable=W0235
# pylint: disable=C0209
import os

import mindspore as ms
import mindspore.nn.probability.distribution as msd
from mindspore import Parameter, Tensor, nn, ops
from mindspore.ops import CustomRegOp, DataType
from mindspore.ops import operations as P

GPU_TREE_TYPE = ["GPUCommon"]
GPU_NODE_TYPE = ["GPUVanilla"]
CPU_TREE_TYPE = ["CPUCommon"]
CPU_NODE_TYPE = ["CPUVanilla", "CPUMuzero"]


class MCTS(nn.Cell):
    """
    Monte Carlo Tree Search(MCTS) is a general search algorithm for some kinds of decision processes,
    most notably those employed in software that plays board games, such as Go, chess. It was originally
    proposed in 2006. A general MCTS has four phases:

    1. Selection - selects the next node according to the selection policy (like UCT, RAVE, AMAF and etc.).
    2. Expansion - unless the selection reached a terminal state, expansion adds a new child node to
       the last node (leaf node) that is selected in Selection phase.
    3. Simulation - performs an algorithm (random, neural net or other algorithms) to obtain the payoff.
    4. Backpropagation - propagates the payoff for all visited node.

    As the time goes by, these four phases of MCTS is evolved. AlphaGo introduced neural network to
    MCTS, which makes the MCTS more powerful.

    This class is a mindspore ops implementation MCTS. User can use provided MCTS algorithm, or develop
    their own MCTS by derived base class (MonteCarloTreeNode) in c++.

    Args:
        env (Environment): It must be the subclass of Environment.
        tree_type (str): The name of tree type.
        node_type (str): The name of node type.
        root_player (float): The root player, which should be less than the total number of player.
        customized_func (AlgorithmFunc): Some algorithm specific class. For more detail, please have a look at
            documentation of AlgorithmFunc.
        device (str): The device type ``"CPU"`` , ``"GPU"`` . ``"Ascend"`` is not support yet.
        args (Tensor): any values which will be the input of MctsCreation. Please following the table below
            to provide the input value. These value will not be reset after invoke `restore_tree_data`.
        has_init_reward (bool, optional): Whether pass the reward to each node during the node initialization.
            Default: ``False``.
        max_action (float, optional): The max number of action in environment. If the `max_action` is ``-1.0`` ,
            the step in Environment will accept the last action. Otherwise, it will accept max_action number
            of action. Default: ``-1.0`` .
        max_iteration (int, optional): The max training iteration of MCTS. Default: ``1000`` .

            +------------------------------+-----------------+-----------------------------+--------------------------+
            |  MCTS Tree Type              |  MCTS Node Type |  Configuration Parameter    |  Notices                 |
            +==============================+=================+=============================+==========================+
            |  CPUCommon                   |  CPUVanilla     |  UCT const                  |  UCT const is used to    |
            +------------------------------+-----------------+-----------------------------+  calculate UCT value in  |
            |  GPUCommon                   |  GPUVanilla     |  UCT const                  |  Selection phase         |
            |                              |                 |                             |                          |
            +------------------------------+-----------------+-----------------------------+--------------------------+

    Examples:
        >>> from mindspore import Tensor
        >>> import mindspore as ms
        >>> from mindspore_rl.environment import TicTacToeEnvironment
        >>> from mindspore_rl.utils import VanillaFunc
        >>> from mindspore_rl.utils import MCTS
        >>> env = TicTacToeEnvironment(None)
        >>> vanilla_func = VanillaFunc(env)
        >>> uct = (Tensor(uct, ms.float32),)
        >>> root_player = 0.0
        >>> mcts = MCTS(env, "CPUCommon", "CPUVanilla", root_player, vanilla_func, device, args=uct)
        >>> action, handle = mcts.mcts_search()
        >>> print(action)
    """

    def __init__(
        self,
        env,
        tree_type,
        node_type,
        root_player,
        customized_func,
        device,
        args,
        has_init_reward=False,
        max_action=-1.0,
        max_iteration=1000,
    ):
        super().__init__()
        if not isinstance(device, str) or device not in ["GPU", "CPU"]:
            raise ValueError(
                "Device {} is illegal, it must in ['GPU','CPU'].".format(device)
            )

        self._check_params(AlgorithmFunc, customized_func, "customized_func")
        self._check_params(int, max_iteration, "max_iteration")
        if max_iteration <= 0:
            raise ValueError(
                f"max_iteration must be larger than 0, but got {max_iteration}"
            )

        current_path = os.path.dirname(os.path.normpath(os.path.realpath(__file__)))
        package_root_path = current_path.rsplit("/", 2)[0]
        os.environ["MS_CUSTOM_AOT_WHITE_LIST"] = package_root_path
        so_path = current_path + "/libmcts_{}.so".format(device.lower())
        state_size = 1.0
        state_shape = env.observation_space.shape
        for shape in state_shape:
            state_size *= shape

        if device == "GPU":
            self._check_element(GPU_TREE_TYPE, tree_type, "MCTS", "tree_type")
            self._check_element(GPU_NODE_TYPE, node_type, "MCTS", "node_type")
        elif device == "CPU":
            self._check_element(CPU_TREE_TYPE, tree_type, "MCTS", "tree_type")
            self._check_element(CPU_NODE_TYPE, node_type, "MCTS", "node_type")
        else:
            raise ValueError("device does not support")
        if root_player >= env.total_num_player() or root_player < 0:
            raise ValueError(
                "root_player {} is illegal, it needs to in range [0, {})".format(
                    root_player, env.total_num_player()
                )
            )

        if node_type == "CPUMuzero":
            mcts_creation_info = (
                CustomRegOp("creation_kernel")
                .input(0, "discount")
                .input(1, "pb_c_base")
                .input(2, "pb_c_init")
                .input(3, "root_dirichlet_alpha")
                .input(4, "root_exploration_fraction")
                .output(0, "tree_handle")
                .dtype_format(
                    DataType.None_None,
                    DataType.None_None,
                    DataType.None_None,
                    DataType.None_None,
                    DataType.None_None,
                    DataType.None_None,
                )
                .attr(
                    "tree_type",
                    "required",
                    "all",
                    value=self._check_params(str, tree_type, "tree_type"),
                )
                .attr(
                    "node_type",
                    "required",
                    "all",
                    value=self._check_params(str, node_type, "node_type"),
                )
                .attr(
                    "max_utility",
                    "required",
                    "all",
                    value=self._check_params(float, env.max_utility(), "max_utility"),
                )
                .attr(
                    "state_size",
                    "required",
                    "all",
                    value=self._check_params(float, state_size, "state_size"),
                )
                .attr(
                    "player",
                    "required",
                    "all",
                    value=self._check_params(float, root_player, "root_player"),
                )
                .attr(
                    "total_num_player",
                    "required",
                    "all",
                    value=self._check_params(
                        float, env.total_num_player(), "total_num_player"
                    ),
                )
                .target(device)
                .get_op_info()
            )
        else:
            mcts_creation_info = (
                CustomRegOp("creation_kernel")
                .input(0, "uct_value")
                .output(0, "tree_handle")
                .dtype_format(DataType.None_None, DataType.None_None)
                .attr(
                    "tree_type",
                    "required",
                    "all",
                    value=self._check_params(str, tree_type, "tree_type"),
                )
                .attr(
                    "node_type",
                    "required",
                    "all",
                    value=self._check_params(str, node_type, "node_type"),
                )
                .attr(
                    "max_utility",
                    "required",
                    "all",
                    value=self._check_params(float, env.max_utility(), "max_utility"),
                )
                .attr(
                    "state_size",
                    "required",
                    "all",
                    value=self._check_params(float, state_size, "state_size"),
                )
                .attr(
                    "player",
                    "required",
                    "all",
                    value=self._check_params(float, root_player, "root_player"),
                )
                .attr(
                    "total_num_player",
                    "required",
                    "all",
                    value=self._check_params(
                        float, env.total_num_player(), "total_num_player"
                    ),
                )
                .target(device)
                .get_op_info()
            )
        mcts_creation = ops.Custom(
            "{}:MctsCreation".format(so_path),
            (1,),
            ms.int64,
            "aot",
            reg_info=mcts_creation_info,
        )
        mcts_creation.add_prim_attr("primitive_target", device)
        self.tree_handle = mcts_creation(*args)
        tree_handle_numpy = float(self.tree_handle.astype(ms.float32).asnumpy()[0])
        self.tree_handle_list = [int(tree_handle_numpy)]

        mcts_selection_info = (
            CustomRegOp("selection_kernel")
            .output(0, "visited_node")
            .output(1, "last_action")
            .dtype_format(DataType.None_None, DataType.None_None)
            .attr(
                "max_action",
                "required",
                "all",
                value=self._check_params(float, max_action, "max_action"),
            )
            .attr(
                "tree_handle",
                "required",
                "all",
                value=self._check_params(float, tree_handle_numpy, "tree_handle"),
            )
            .target(device)
            .get_op_info()
        )
        if (max_action != -1) and (max_action != len(env.legal_action())):
            raise ValueError(
                "max_action must be -1 or the largest legal action of environment, but got ",
                max_action,
            )
        if max_action != -1:
            self.mcts_selection = ops.Custom(
                "{}:MctsSelection".format(so_path),
                ((1,), (max_action,)),
                (ms.int64, ms.int32),
                "aot",
                reg_info=mcts_selection_info,
            )
        else:
            self.mcts_selection = ops.Custom(
                "{}:MctsSelection".format(so_path),
                ((1,), (1,)),
                (ms.int64, ms.int32),
                "aot",
                reg_info=mcts_selection_info,
            )
        self.mcts_selection.add_prim_attr("primitive_target", device)

        mcts_expansion_info = (
            CustomRegOp("expansion_kernel")
            .input(0, "visited_node")
            .input(1, "legal_action")
            .input(2, "prior")
            .input(3, "reward")
            .output(0, "success")
            .dtype_format(
                DataType.None_None,
                DataType.None_None,
                DataType.None_None,
                DataType.None_None,
                DataType.None_None,
            )
            .attr(
                "node_type",
                "required",
                "all",
                value=self._check_params(str, node_type, "node_type"),
            )
            .attr(
                "has_init_reward",
                "required",
                "all",
                value=self._check_params(bool, has_init_reward, "has_init_reward"),
            )
            .attr(
                "tree_handle",
                "required",
                "all",
                value=self._check_params(float, tree_handle_numpy, "tree_handle"),
            )
            .target(device)
            .get_op_info()
        )
        self.mcts_expansion = ops.Custom(
            "{}:MctsExpansion".format(so_path),
            (1,),
            (ms.bool_),
            "aot",
            reg_info=mcts_expansion_info,
        )
        self.mcts_expansion.add_prim_attr("primitive_target", device)

        mcts_backprop_info = (
            CustomRegOp("backprop_kernel")
            .input(0, "visited_node")
            .input(1, "returns")
            .output(0, "solved")
            .dtype_format(DataType.None_None, DataType.None_None, DataType.None_None)
            .attr(
                "tree_handle",
                "required",
                "all",
                value=self._check_params(float, tree_handle_numpy, "tree_handle"),
            )
            .target(device)
            .get_op_info()
        )
        self.mcts_backpropagation = ops.Custom(
            "{}:MctsBackpropagation".format(so_path),
            (1,),
            (ms.bool_),
            "aot",
            reg_info=mcts_backprop_info,
        )
        self.mcts_backpropagation.add_prim_attr("primitive_target", device)

        mcts_bestaction_info = (
            CustomRegOp("bestaction_kernel")
            .output(0, "action")
            .dtype_format(DataType.None_None)
            .attr(
                "tree_handle",
                "required",
                "all",
                value=self._check_params(float, tree_handle_numpy, "tree_handle"),
            )
            .target(device)
            .get_op_info()
        )
        self.best_action = ops.Custom(
            "{}:BestAction".format(so_path),
            (1,),
            (ms.int32),
            "aot",
            reg_info=mcts_bestaction_info,
        )
        self.best_action.add_prim_attr("primitive_target", device)

        mcts_outcome_info = (
            CustomRegOp("outcome_kernel")
            .input(0, "visited_node")
            .input(1, "reward")
            .output(0, "success")
            .dtype_format(DataType.None_None, DataType.None_None, DataType.None_None)
            .attr(
                "tree_handle",
                "required",
                "all",
                value=self._check_params(float, tree_handle_numpy, "tree_handle"),
            )
            .target(device)
            .get_op_info()
        )
        self.update_leafnode_outcome = ops.Custom(
            "{}:UpdateLeafNodeOutcome".format(so_path),
            (1,),
            (ms.bool_),
            "aot",
            reg_info=mcts_outcome_info,
        )
        self.update_leafnode_outcome.add_prim_attr("primitive_target", device)

        mcts_terminal_info = (
            CustomRegOp("terminal_kernel")
            .input(0, "visited_node")
            .input(1, "terminal")
            .output(0, "success")
            .dtype_format(DataType.None_None, DataType.None_None, DataType.None_None)
            .attr(
                "tree_handle",
                "required",
                "all",
                value=self._check_params(float, tree_handle_numpy, "tree_handle"),
            )
            .target(device)
            .get_op_info()
        )
        self.update_leafnode_terminal = ops.Custom(
            "{}:UpdateLeafNodeTerminal".format(so_path),
            (1,),
            (ms.bool_),
            "aot",
            reg_info=mcts_terminal_info,
        )
        self.update_leafnode_terminal.add_prim_attr("primitive_target", device)

        mcts_leafstate_info = (
            CustomRegOp("leafstate_kernel")
            .input(0, "visited_node")
            .input(1, "state")
            .output(0, "success")
            .dtype_format(DataType.None_None, DataType.None_None, DataType.None_None)
            .attr(
                "tree_handle",
                "required",
                "all",
                value=self._check_params(float, tree_handle_numpy, "tree_handle"),
            )
            .target(device)
            .get_op_info()
        )
        self.update_leafnode_state = ops.Custom(
            "{}:UpdateLeafNodeState".format(so_path),
            (1,),
            (ms.bool_),
            "aot",
            reg_info=mcts_leafstate_info,
        )
        self.update_leafnode_state.add_prim_attr("primitive_target", device)

        mcts_rootstate_info = (
            CustomRegOp("rootstate_kernel")
            .input(0, "state")
            .output(0, "success")
            .dtype_format(DataType.None_None, DataType.None_None)
            .attr(
                "tree_handle",
                "required",
                "all",
                value=self._check_params(float, tree_handle_numpy, "tree_handle"),
            )
            .target(device)
            .get_op_info()
        )
        self.update_root_state = ops.Custom(
            "{}:UpdateRootState".format(so_path),
            (1,),
            (ms.bool_),
            "aot",
            reg_info=mcts_rootstate_info,
        )
        self.update_root_state.add_prim_attr("primitive_target", device)

        mcts_getlast_info = (
            CustomRegOp("getlast_kernel")
            .input(0, "visited_node")
            .output(0, "state")
            .dtype_format(DataType.None_None, DataType.None_None)
            .attr(
                "tree_handle",
                "required",
                "all",
                value=self._check_params(float, tree_handle_numpy, "tree_handle"),
            )
            .target(device)
            .get_op_info()
        )
        self.get_last_state = ops.Custom(
            "{}:GetLastState".format(so_path),
            state_shape,
            (ms.float32),
            "aot",
            reg_info=mcts_getlast_info,
        )
        self.get_last_state.add_prim_attr("primitive_target", device)

        mcts_globalvar_info = (
            CustomRegOp("globalvar_kernel")
            .attr(
                "tree_handle",
                "required",
                "all",
                value=self._check_params(float, tree_handle_numpy, "tree_handle"),
            )
            .target(device)
            .get_op_info()
        )
        self.update_global_variable = ops.Custom(
            "{}:UpdateGlobalVariable".format(so_path),
            (1,),
            (ms.bool_),
            "aot",
            reg_info=mcts_globalvar_info,
        )
        self.update_global_variable.add_prim_attr("primitive_target", device)

        mcts_destroy_info = (
            CustomRegOp("destroy_kernel")
            .input(0, "handle")
            .output(0, "success")
            .dtype_format(DataType.None_None, DataType.None_None)
            .attr(
                "tree_handle",
                "required",
                "all",
                value=self._check_params(float, tree_handle_numpy, "tree_handle"),
            )
            .target(device)
            .get_op_info()
        )
        self.destroy_tree = ops.Custom(
            "{}:DestroyTree".format(so_path),
            (1,),
            (ms.bool_),
            "aot",
            reg_info=mcts_destroy_info,
        )
        self.destroy_tree.add_prim_attr("primitive_target", device)

        mcts_restore_info = (
            CustomRegOp("restore_kernel")
            .input(0, "dummy_handle")
            .output(0, "success")
            .dtype_format(DataType.None_None, DataType.None_None)
            .attr(
                "tree_handle",
                "required",
                "all",
                value=self._check_params(float, tree_handle_numpy, "tree_handle"),
            )
            .target(device)
            .get_op_info()
        )
        self.restore_tree = ops.Custom(
            "{}:RestoreTree".format(so_path),
            (1,),
            (ms.bool_),
            "aot",
            reg_info=mcts_restore_info,
        )
        self.restore_tree.add_prim_attr("primitive_target", device)

        mcts_get_value_info = (
            CustomRegOp("get_value_kernel")
            .input(0, "dummy_handle")
            .output(0, "value")
            .output(1, "norm_explore_count")
            .dtype_format(DataType.None_None, DataType.None_None, DataType.None_None)
            .attr(
                "tree_handle",
                "required",
                "all",
                value=self._check_params(float, tree_handle_numpy, "tree_handle"),
            )
            .target(device)
            .get_op_info()
        )
        self.get_root_info = ops.Custom(
            "{}:GetRootInfo".format(so_path),
            ((1,), (len(env.legal_action()),)),
            (ms.float32, ms.float32),
            "aot",
            reg_info=mcts_get_value_info,
        )
        self.get_root_info.add_prim_attr("primitive_target", device)
        self.depend = P.Depend()

        # Add side effect annotation
        self.mcts_expansion.add_prim_attr("side_effect_mem", True)
        self.mcts_backpropagation.add_prim_attr("side_effect_mem", True)
        self.update_leafnode_outcome.add_prim_attr("side_effect_mem", True)
        self.update_leafnode_terminal.add_prim_attr("side_effect_mem", True)
        self.update_leafnode_state.add_prim_attr("side_effect_mem", True)
        self.update_root_state.add_prim_attr("side_effect_mem", True)
        self.destroy_tree.add_prim_attr("side_effect_mem", True)
        self.restore_tree.add_prim_attr("side_effect_mem", True)
        self.update_global_variable.add_prim_attr("side_effect_mem", True)

        self.zero = Tensor(0, ms.int32)
        self.zero_float = Tensor(0, ms.float32)
        self.true = Tensor(True, ms.bool_)
        self.false = Tensor(False, ms.bool_)

        self.env = env
        self.tree_type = tree_type
        self.node_type = node_type
        self.max_iteration = Tensor(max_iteration, ms.int32)
        self.max_action = max_action
        self.customized_func = customized_func

    @ms.jit
    def mcts_search(self, *args):
        """
        mcts_search is the main function of MCTS. Invoke this function will return the best
        action of current state.

        Args:
            *args (Tensor): The variable which updates during each iteration. They will be restored
                            after invoking `restore_tree_data`. The input value needs to match provied
                            algorithm.

        Returns:
            - action (mindspore.int32), The action which is returned by monte carlo tree search.
            - handle (mindspore.int64), The unique handle of mcts tree.
        """

        expanded = self.false
        reward = self.zero_float
        solved = self.false
        # Create a replica of environment
        new_state = self.env.save()
        self.update_root_state(new_state)
        self.update_global_variable(*args)
        i = self.zero
        while i < self.max_iteration:
            # 1. Interact with the replica of environment, and update the latest state
            # and its reward
            visited_node, last_action = self.mcts_selection()
            last_state = self.get_last_state(visited_node)
            if expanded:
                self.env.load(last_state)
                new_state, reward, _ = self.env.step(last_action)
            else:
                new_state, reward, _ = self.env.load(last_state)
            self.update_leafnode_state(visited_node, new_state)
            # 2. Calculate the legal action and their probability of the latest state
            legal_action = self.env.legal_action()
            prior = self.customized_func.calculate_prior(new_state, legal_action)

            if not self.env.is_terminal():
                expanded = self.true
                self.mcts_expansion(visited_node, legal_action, prior, reward)
            else:
                self.update_leafnode_outcome(visited_node, reward)
                self.update_leafnode_terminal(visited_node, self.true)
            # 3. Calculate the return of the latest state, it could obtain from neural network
            #    or play randomly
            returns = self.customized_func.simulation(new_state)
            solved = self.mcts_backpropagation(visited_node, returns)
            if solved:
                break
            i += 1
        action = self.best_action()
        return action, self.tree_handle

    def restore_tree_data(self, handle):
        r"""
        restore_tree_data will restore all the data in the tree, back to the initial state.

        Args:
            handle (mindspore.int64): The unique handle of mcts tree.

        Returns:
            success (mindspore.bool\_), Whether restore is successful.
        """
        self._check_element(
            self.tree_handle_list, handle, "restore_tree_data", "handle"
        )
        return self.restore_tree(handle)

    def destroy(self, handle):
        r"""
        destroy will destroy current tree. Please call this function ONLY when
        do not use this tree any more.

        Args:
            handle (mindspore.int64): The unique handle of mcts tree.

        Returns:
            success (mindspore.bool\_), Whether destroy is successful.
        """
        self._check_element(self.tree_handle_list, handle, "destroy", "handle")
        ret = self.destroy_tree(handle)
        self.tree_handle_list.pop()
        return ret

    @ms.jit
    def _get_root_information(self, dummpy_handle):
        """Does not support yet"""
        return self.get_root_info(dummpy_handle)

    def _check_params(self, check_type, input_value, name):
        """Check params type for input"""
        if not isinstance(input_value, check_type):
            raise TypeError(
                f"Input value {name} must be {str(check_type)}, but got {type(input_value)}"
            )
        return input_value

    def _check_element(self, expected_element, input_element, func_name, arg_name):
        """Check whether input_elemnt is in expected_element"""
        if input_element not in expected_element:
            raise ValueError(
                f"The input {arg_name} of {func_name} must be in {expected_element}, but got '{input_element}'"
            )


class AlgorithmFunc(nn.Cell):
    """
    This is the base class for user to customize algorithm in MCTS. User need to
    inherit this base class and implement all the functions with SAME input and output.
    """

    def __init__(self):
        super().__init__()

    def calculate_prior(self, new_state, legal_action):
        """
        Calculate prior of the input legal actions.

        Args:
            new_state (mindspore.float32): The state of environment.
            legal_action (mindspore.int32): The legal action of environment

        Returns:
            prior (mindspore.float32), The probability (or prior) of all the input legal actions.
        """
        raise NotImplementedError("You must implement this function")

    def simulation(self, new_state):
        """
        Simulation phase in MCTS. It takes the state as input and return the rewards.

        Args:
            new_state (mindspore.float32): The state of environment.

        Returns:
            rewards (mindspore.float32), The results of simulation.
        """
        raise NotImplementedError("You must implement this function")


class VanillaFunc(AlgorithmFunc):
    """
    This is the customized algorithm for VanillaMCTS. The prior of each legal action is uniform
    distribution and it plays randomly to obtain the result of simulation.

    Args:
        env (Environment): The input environment.

    Examples:
        >>> env = TicTacToeEnvironment(None)
        >>> vanilla_func = VanillaFunc(env)
        >>> legal_action = env.legal_action()
        >>> prior = vanilla_func.calculate_prior(legal_action, legal_action)
        >>> print(prior)
    """

    def __init__(self, env):
        super().__init__()
        self.minus_one = Tensor(-1, ms.int32)
        self.zero = Tensor(0, ms.int32)
        self.ones_like = P.OnesLike()
        self.categorical = msd.Categorical()
        self.env = env

        self.false = Tensor(False, ms.bool_)

    def calculate_prior(self, new_state, legal_action):
        """
        The functionality of calculate_prior is to calculate prior of the input legal actions.

        Args:
            new_state (mindspore.float32): The state of environment.
            legal_action (mindspore.int32): The legal action of environment

        Returns:
            prior (mindspore.float32), The probability (or prior) of all the input legal actions.
        """
        invalid_action_num = (legal_action == -1).sum()
        prior = self.ones_like(legal_action).astype(ms.float32) / (
            len(legal_action) - invalid_action_num
        )
        return prior

    def simulation(self, new_state):
        """
        The functionality of simulation is to calculate reward of the input state.

        Args:
            new_state (mindspore.float32): The state of environment.

        Returns:
            rewards (mindspore.float32), The results of simulation.
        """
        _, reward, done = self.env.load(new_state)
        while not done:
            legal_action = self.env.legal_action()
            mask = legal_action == -1
            invalid_action_num = (legal_action == -1).sum()
            prob = self.ones_like(legal_action).astype(ms.float32) / (
                len(legal_action) - invalid_action_num
            )
            prob[mask] = 0
            action = self.categorical.sample((), prob)
            new_state, reward, done = self.env.step(legal_action[action])
        return reward


class _SupportToScalar(nn.Cell):
    """
    Support to scalar is used in Muzero, it will decompressed an Tensor to scalar.
    """

    def __init__(self, value_min: float, value_max: float, eps: float = 0.001):
        super().__init__()
        self.eps = eps
        self.support = nn.Range(value_min, value_max + 1)()
        self.reduce_sum = P.ReduceSum()
        self.sign = P.Sign()
        self.sqrt = P.Sqrt()
        self.absolute = P.Abs()

    def construct(self, logits):
        """Calculate the decompressed value"""
        probabilities = nn.Softmax()(logits)
        v = self.reduce_sum(probabilities * self.support, -1)

        # Inverting the value scaling (defined in https://arxiv.org/abs/1805.11593)
        decompressed_value = self.sign(v) * (
            (
                (self.sqrt(1 + 4 * self.eps * (self.absolute(v) + 1 + self.eps)) - 1)
                / (2 * self.eps)
            )
            ** 2
            - 1
        )

        return decompressed_value


class MuzeroFunc(AlgorithmFunc):
    """
    This is the customized algorithm for MuzeroCTS. The prior of each legal action and predicted value
    are calculated by neural network.
    """

    def __init__(self, net):
        super().__init__()
        self.predict_net = net
        self.decompressed_value = _SupportToScalar(-300, 300)
        self.value = Parameter(
            Tensor([0], ms.float32), requires_grad=False, name="value"
        )

        self.false = Tensor(False, ms.bool_)

    def calculate_prior(self, new_state, legal_action):
        """
        The functionality of calculate_prior is to calculate prior of the input legal actions.

        Args:
            new_state (mindspore.float32): The state of environment.
            legal_action (mindspore.int32): The legal action of environment.

        Returns:
            prior (mindspore.float32), The probability (or prior) of all the input legal actions.
        """
        policy, value = self.predict_net(new_state)
        self.value = self.decompressed_value(value)
        return policy

    def simulation(self, new_state):
        """
        The functionality of simulation is to calculate reward of the input state.

        Args:
            new_state (mindspore.float32): The state of environment.

        Returns:
            rewards (mindspore.float32), The results of simulation.
        """
        return self.value
