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
"""Dreamer Models"""

import mindspore as ms
import mindspore.nn.probability.distribution as msd
import numpy as np
from mindspore import Tensor, nn
from mindspore.ops import operations as P

from mindspore_rl.algorithm.sac.tanh_normal import (
    MultivariateNormalDiag,
    TanhMultivariateNormalDiag,
)
from mindspore_rl.network import FullyConnectedLayers


class ConvEncoder(nn.Cell):
    """Convolutional Encoder for input observation (images)"""

    def __init__(self, params):
        super().__init__()
        self.depth = params["depth"]
        stride = params["stride_conv_encoder"]
        self.conv1 = nn.Conv2d(
            3,
            1 * self.depth,
            4,
            stride,
            pad_mode="valid",
            has_bias=True,
            weight_init="xavier_uniform",
        )
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            1 * self.depth,
            2 * self.depth,
            4,
            stride,
            pad_mode="valid",
            has_bias=True,
            weight_init="xavier_uniform",
        )
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(
            2 * self.depth,
            4 * self.depth,
            4,
            stride,
            pad_mode="valid",
            has_bias=True,
            weight_init="xavier_uniform",
        )
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(
            4 * self.depth,
            8 * self.depth,
            4,
            stride,
            pad_mode="valid",
            has_bias=True,
            weight_init="xavier_uniform",
        )
        self.relu4 = nn.ReLU()
        self.reshape = P.Reshape()
        self.concat = P.Concat()

    def construct(self, images):
        """Forward of conv encoder"""
        x = self.reshape(images, (-1,) + tuple(images.shape[-3:]))
        x = x.transpose(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = x.transpose(0, 2, 3, 1)
        return self.reshape(x, (images.shape[:-3] + (32 * self.depth,)))


class ConvDecoder(nn.Cell):
    """Convolutional Decoder"""

    def __init__(self, params):
        super().__init__()
        self.shape = params["conv_decoder_shape"]
        self.depth = params["depth"]
        stride = params["stride_conv_decoder"]
        self.fc1 = nn.Dense(
            in_channels=238, out_channels=32 * self.depth, weight_init="xavier_uniform"
        )
        self.deconv1 = nn.Conv2dTranspose(
            in_channels=32 * self.depth,
            out_channels=4 * self.depth,
            kernel_size=5,
            stride=stride,
            pad_mode="valid",
            has_bias=True,
            weight_init="xavier_uniform",
        )
        self.relu1 = nn.ReLU()
        self.deconv2 = nn.Conv2dTranspose(
            in_channels=4 * self.depth,
            out_channels=2 * self.depth,
            kernel_size=5,
            stride=stride,
            pad_mode="valid",
            has_bias=True,
            weight_init="xavier_uniform",
        )
        self.relu2 = nn.ReLU()
        self.deconv3 = nn.Conv2dTranspose(
            in_channels=2 * self.depth,
            out_channels=1 * self.depth,
            kernel_size=6,
            stride=stride,
            pad_mode="valid",
            has_bias=True,
            weight_init="xavier_uniform",
        )
        self.relu3 = nn.ReLU()
        self.deconv4 = nn.Conv2dTranspose(
            in_channels=1 * self.depth,
            pad_mode="valid",
            out_channels=self.shape[-1],
            kernel_size=6,
            stride=stride,
            has_bias=True,
            weight_init="xavier_uniform",
        )

        self.reshape = P.Reshape()
        self.transpose = P.Transpose()

    def construct(self, feat):
        """Forward of conv decoder"""
        x = self.fc1(feat)
        x = self.reshape(x, (-1, 1, 1, 32 * self.depth))
        # Transfer from NHWC to NCHW
        x = self.transpose(x, (0, 3, 1, 2))
        x = self.deconv1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.relu2(x)
        x = self.deconv3(x)
        x = self.relu3(x)
        x = self.deconv4(x)
        # Transfer back to NHWC
        x = self.transpose(x, (0, 2, 3, 1))
        # It is used to calculate log prob of img
        return self.reshape(x, (feat.shape[:-1] + self.shape))


class DenseDecoder(nn.Cell):
    """Dense Decoder"""

    def __init__(self, dense_decoder_shape, dense_decoder_layers):
        super().__init__()
        self.shape = dense_decoder_shape
        layers_param = dense_decoder_layers
        self.fc = FullyConnectedLayers(
            fc_layer_params=layers_param,
            activation_fn=nn.ELU(),
            weight_init="xavier_uniform",
        )
        self.fc_out = nn.Dense(
            in_channels=layers_param[-1],
            out_channels=int(np.prod(self.shape)),
            weight_init="xavier_uniform",
        )

        self.reshape = P.Reshape()

    def construct(self, feat):
        x = self.fc(feat)
        x = self.fc_out(x)
        x = self.reshape(x, (feat.shape[:-1] + self.shape))
        # It is used to calculate log prob of reward
        return x


class ActionDecoder(nn.Cell):
    """Action Decoder for input features"""

    def __init__(self, params):
        super().__init__()
        self.dtype = params["dtype"]
        layers_param = params["action_decoder_layers"]
        size = params["size"]
        self.min_std = Tensor(params["min_std"], self.dtype)
        self.mean_scale = Tensor(params["mean_scale"], self.dtype)
        self.init_std = Tensor(params["init_std"], self.dtype)
        self.zero_float = Tensor(0, self.dtype)

        self.fc = FullyConnectedLayers(
            fc_layer_params=layers_param,
            activation_fn=nn.ELU(),
            weight_init="xavier_uniform",
        )
        self.fc_out = nn.Dense(
            in_channels=layers_param[-1],
            out_channels=2 * size,
            weight_init="xavier_uniform",
        )
        self.split = P.Split(-1, 2)
        self.softplus = P.Softplus()
        self.tanh = P.Tanh()
        self.normal = msd.Normal(dtype=self.dtype)
        self.transformed_dist = TanhMultivariateNormalDiag(dtype=self.dtype)

        self.log = P.Log()
        self.exp = P.Exp()
        self.select = P.Select()

    def construct(self, features, training):
        raw_init_std = self.log(self.exp(self.init_std) - 1)
        x = self.fc(features)
        x = self.fc_out(x)
        mean, std = self.split(x)
        mean = self.mean_scale * self.tanh(mean / self.mean_scale)
        std = self.softplus(std + raw_init_std) + self.min_std
        action = self.zero_float
        action = self.transformed_dist.sample((), mean, std) if training else mean
        return action


class RSSM(nn.Cell):
    """Recurrent State-Space Model"""

    def __init__(self, params):
        super().__init__()
        self.dtype = params["dtype"]
        hidden_size = params["hidden_size"]
        stoch_size = params["stoch_size"]
        deter_size = params["deter_size"]
        self.batch_size = params["batch_size"]
        # ImgStepNet
        self.fc1_img = nn.Dense(
            in_channels=36,
            out_channels=hidden_size,
            activation=nn.ELU(),
            weight_init="xavier_uniform",
        ).to_float(self.dtype)
        self.gru_img = nn.GRU(input_size=hidden_size, hidden_size=deter_size).to_float(
            self.dtype
        )
        self.fc2_img = nn.Dense(
            in_channels=deter_size,
            out_channels=hidden_size,
            activation=nn.ELU(),
            weight_init="xavier_uniform",
        ).to_float(self.dtype)
        self.fc3_img = nn.Dense(
            in_channels=hidden_size,
            out_channels=2 * stoch_size,
            weight_init="xavier_uniform",
        ).to_float(self.dtype)
        self.softplus_img = P.Softplus()

        # ObsStepNet
        self.fc1_obs = nn.Dense(
            in_channels=1232,
            out_channels=hidden_size,
            activation=nn.ELU(),
            weight_init="xavier_uniform",
        ).to_float(self.dtype)
        self.fc2_obs = nn.Dense(
            in_channels=hidden_size,
            out_channels=2 * stoch_size,
            weight_init="xavier_uniform",
        ).to_float(self.dtype)
        self.softplus_obs = P.Softplus()

        self.split = P.Split(-1, 2)
        self.zeros = P.Zeros()
        self.stack = P.Stack(axis=0)
        self.concat = P.Concat(axis=-1)
        self.expand_dims = P.ExpandDims()
        self.squeeze = P.Squeeze(axis=0)
        self.transpose = P.Transpose()
        self.multivariate_norm_diag = MultivariateNormalDiag(dtype=self.dtype)

        self.zero_int = Tensor(0, ms.int32)

    def obs_step(self, prev_stoch, prev_deter, prev_action, embed):
        """obs step, which returns the the posterior and prior info"""
        prior_mean, prior_std, prior_stoch, deter = self.img_step(
            prev_stoch, prev_deter, prev_action
        )
        x = self.concat([deter, embed])
        x = self.fc1_obs(x)
        x = self.fc2_obs(x)
        post_mean, post_std = self.split(x)
        post_std = self.softplus_obs(post_std) + 0.1
        post_stoch = self.multivariate_norm_diag.sample(
            (), post_mean, post_std, independent=1
        )

        return (
            post_mean,
            post_std,
            post_stoch,
            prior_mean,
            prior_std,
            prior_stoch,
            deter,
        )

    def img_step(self, prev_stoch, prev_deter, prev_action):
        """img step, which returns the prior info"""
        x = self.concat([prev_stoch, prev_action])
        x = self.fc1_img(x)
        x = self.expand_dims(x, 0)
        prev_deter = self.expand_dims(prev_deter, 0)
        x, deter = self.gru_img(x, prev_deter)
        x = self.squeeze(x)
        deter = self.squeeze(deter)
        x = self.fc2_img(x)
        x = self.fc3_img(x)
        mean, std = self.split(x)
        std = self.softplus_img(std) + 0.1
        stoch = self.multivariate_norm_diag.sample((), mean, std, independent=1)
        return mean, std, stoch, deter

    def observe(self, embed, action, start_stoch, start_deter):
        """observe function"""
        embed = self.transpose(embed, (1, 0, 2))
        action = self.transpose(action, (1, 0, 2))
        prev_stoch = start_stoch
        prev_deter = start_deter

        mean_post = []
        std_post = []
        stoch_post = []

        mean_prior = []
        std_prior = []
        stoch_prior = []

        deter = []

        i = 0
        while i < embed.shape[0]:
            embed_i = embed[i]
            action_i = action[i]
            (
                post_mean,
                post_std,
                prev_stoch,
                prior_mean,
                prior_std,
                prior_stoch,
                prev_deter,
            ) = self.obs_step(prev_stoch, prev_deter, action_i, embed_i)

            mean_post.append(post_mean)
            std_post.append(post_std)
            stoch_post.append(prev_stoch)
            mean_prior.append(prior_mean)
            std_prior.append(prior_std)
            stoch_prior.append(prior_stoch)
            deter.append(prev_deter)
            i += 1

        mean_post = self.stack(mean_post)
        std_post = self.stack(std_post)
        stoch_post = self.stack(stoch_post)
        mean_prior = self.stack(mean_prior)
        std_prior = self.stack(std_prior)
        stoch_prior = self.stack(stoch_prior)
        deter = self.stack(deter)

        post_mean_tensor = self.transpose(mean_post, (1, 0, 2))
        post_std_tensor = self.transpose(std_post, (1, 0, 2))
        post_stoch_tensor = self.transpose(stoch_post, (1, 0, 2))
        prior_mean_tensor = self.transpose(mean_prior, (1, 0, 2))
        prior_std_tensor = self.transpose(std_prior, (1, 0, 2))
        prior_stoch_tensor = self.transpose(stoch_prior, (1, 0, 2))
        deter_tensor = self.transpose(deter, (1, 0, 2))

        return (
            post_mean_tensor,
            post_std_tensor,
            post_stoch_tensor,
            prior_mean_tensor,
            prior_std_tensor,
            prior_stoch_tensor,
            deter_tensor,
        )
