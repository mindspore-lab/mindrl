"""c51 full connect layer"""

from mindspore import dtype as mstype
from mindspore import nn
from mindspore.ops import operations as P


class FullyConnectedNet(nn.Cell):
    """full connect layer with noisy option"""

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        action_dim,
        atoms_num,
        compute_type=mstype.float32,
    ):
        super().__init__()
        self.linear1 = nn.Dense(
            input_size, hidden_size, weight_init="XavierUniform", bias_init="zeros"
        ).to_float(compute_type)
        self.linear2 = nn.Dense(
            hidden_size, output_size, weight_init="XavierUniform", bias_init="zeros"
        ).to_float(compute_type)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.cast = P.Cast()
        self.action_dim = action_dim
        self.atoms_num = atoms_num

    def construct(self, x):
        """
        Returns output of Dense layer.

        Args:
            x (Tensor): Tensor as the input of network.

        Returns:
            The output of the Dense layer.
        """
        x = self.relu1(self.linear1(x))
        x = self.relu1(self.linear2(x))
        x = x.view(-1, self.action_dim, self.atoms_num)
        x = self.cast(x, mstype.float32)
        return x
