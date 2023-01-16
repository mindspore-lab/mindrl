"""c51 policy"""
import numpy as np
import mindspore as ms
from mindspore import Tensor
import mindspore.numpy as mnp
import mindspore.nn as nn
from mindspore_rl.policy import RandomPolicy
from mindspore_rl.policy import policy


class GreedyPolicyForValueDistribution(policy.Policy):
    """Produces a sample action base on the given greedy policy."""
    def __init__(self,
                 input_network, atoms_num, v_min, v_max, action_space_dim):
        super(GreedyPolicyForValueDistribution, self).__init__()
        self._input_network = input_network
        self.argmax = ms.ops.Argmax()
        self.mul = ms.ops.Mul()
        self.atoms_num = atoms_num
        self.v_min = v_min
        self.v_max = v_max
        self.action_dim = action_space_dim
        self.softmax = nn.Softmax()

    # pylint:disable=W0221
    def construct(self, state):
        """
        Returns the best action.

        Args:
            state (Tensor): State tensor as the input of network.

        Returns:
            action_max, the best action.
        """
        actions_distribution = self._input_network(state)
        actions_distribution = self.softmax(actions_distribution)
        actions_distribution = self.mul(actions_distribution, mnp.linspace(Tensor(self.v_min, ms.float32),
                                                                           Tensor(self.v_max, ms.float32),
                                                                           self.atoms_num))

        actions = actions_distribution.sum(2)
        action_max = self.argmax(actions)
        return action_max


class EpsilonGreedyPolicyForValueDistribution(policy.Policy):
    r"""
    Produces a sample action base on the given epsilon-greedy policy.

    Args:
        input_network (Cell): A network returns policy action.
        size (int): Shape of epsilon.
        epsi_high (float): A high epsilon for exploration betweens [0, 1].
        epsi_low (float): A low epsilon for exploration betweens [0, epsi_high].
        decay (float): A decay factor applied to epsilon.
        atoms_num:
        v_min: The lower bound of the value distribution.
        v_max: The upper bound of the value distribution.
        action_space_dim (int): Dimensions of the action space.

    Examples:
        >>> state_dim, hidden_dim, action_dim = (4, 10, 2)
        >>> input_net = FullyConnectedNet(state_dim, hidden_dim, action_dim)
        >>> policy = EpsilonGreedyPolicyForValueDistribution(input_net, 1, 0.1, 0.1, 100, action_dim)
        >>> state = Tensor(np.ones([1, state_dim]).astype(np.float32))
        >>> step =  Tensor(np.array([10,]).astype(np.float32))
        >>> output = policy(state, step)
        >>> print(output.shape)
        (1,)
    """

    def __init__(self,
                 input_network,
                 size,
                 epsi_high,
                 epsi_low,
                 decay,
                 atoms_num,
                 v_min,
                 v_max,
                 action_space_dim):
        super(EpsilonGreedyPolicyForValueDistribution, self).__init__()
        self._input_network = input_network

        self.sub = ms.ops.Sub()
        self.add = ms.ops.Add()
        self.mul = ms.ops.Mul()
        self.exp = ms.ops.Exp()
        self.slice = ms.ops.Slice()
        self.squeeze = ms.ops.Squeeze(1)
        self.less = ms.ops.Less()
        self.select = ms.ops.Select()
        self.randreal = ms.ops.UniformReal()

        self.decay_epsilon = (epsi_high != epsi_low)
        self.epsi_low = epsi_low
        self._size = size
        self._shape = (1,)
        self._elow_arr = np.ones(self._size) * epsi_low
        self._ehigh_arr = np.ones(self._size) * epsi_high
        self._steps_arr = np.ones(self._size)
        self._decay_arr = np.ones(self._size) * decay
        self._mone_arr = np.ones(self._size) * -1

        self._epsi_high = Tensor(self._ehigh_arr, ms.float32)
        self._epsi_low = Tensor(self._elow_arr, ms.float32)
        self._decay = Tensor(self._decay_arr, ms.float32)
        self._mins_one = Tensor(self._mone_arr, ms.float32)

        self.atoms_num = atoms_num
        self.v_min = v_min
        self.v_max = v_max

        self._action_space_dim = action_space_dim
        self.greedy_policy = GreedyPolicyForValueDistribution(self._input_network, self.atoms_num, self.v_min,
                                                              self.v_max, self._action_space_dim)
        self.random_policy = RandomPolicy(self._action_space_dim)

    # pylint:disable=W0221
    def construct(self, state, step):
        """
        The interface of the construct function.

        Args:
            state (Tensor): The input tensor for network.
            step (Tensor): The current step, effects the epsilon decay.

        Returns:
            The output action.
        """
        greedy_action = self.greedy_policy(state)
        random_action = self.random_policy()

        if self.decay_epsilon:
            epsi_sub = self.sub(self._epsi_high, self._epsi_low)
            epsi_exp = self.exp(
                self.mul(
                    self._mins_one,
                    step/self._decay))
            epsi_mul = self.mul(epsi_sub, epsi_exp)
            epsi = self.add(self._epsi_low, epsi_mul)
            epsi = self.slice(epsi, (0, 0), (1, 1))
            epsi = self.squeeze(epsi)
        else:
            epsi = self.epsi_low

        cond = self.less(self.randreal(self._shape), epsi)
        output_action = self.select(cond, random_action, greedy_action)
        return output_action
