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
"""Dreamer"""

import mindspore as ms
import mindspore.nn.probability.distribution as msd
from mindspore import Tensor, nn, ops
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P

from mindspore_rl.agent import Actor, Learner
from mindspore_rl.algorithm.dreamer.models import (
    RSSM,
    ActionDecoder,
    ConvDecoder,
    ConvEncoder,
    DenseDecoder,
)
from mindspore_rl.algorithm.sac.tanh_normal import MultivariateNormalDiag
from mindspore_rl.utils import DiscountedReturn

_grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()


@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * F.cast(reciprocal(scale), F.dtype(grad))


class DreamerPolicy:
    """Dreamer Policy"""

    class InitPolicy(nn.Cell):
        """Init policy of dreamer"""

        def __init__(self, params):
            super().__init__()
            self.params = params
            self.uniform = msd.Uniform(low=-1.0, high=1.0, dtype=ms.float16)

        def construct(self, prev_mean, prev_std, prev_stoch, prev_deter):
            return (
                self.uniform.sample((1, 6)),
                prev_mean,
                prev_std,
                prev_stoch,
                prev_deter,
            )

    class CollectPolicy(nn.Cell):
        """Collect policy of Dreamer"""

        def __init__(self, conv_encoder, rssm):
            super().__init__()
            self.conv_encoder = conv_encoder
            self.rssm = rssm

            self.concat = P.Concat(axis=-1)
            self.div = P.Div()
            self.print = P.Print()

        def construct(self, obs, prev_stoch, prev_deter, prev_action):
            embed = self.conv_encoder(obs)
            post_mean, post_std, post_stoch, _, _, _, prior_deter = self.rssm.obs_step(
                prev_stoch, prev_deter, prev_action, embed
            )
            feat = self.concat([post_stoch, prior_deter])
            return feat, post_mean, post_std, post_stoch, prior_deter

    def __init__(self, params):
        self.conv_encoder = ConvEncoder(params).to_float(ms.float16)
        self.conv_decoder = ConvDecoder(params).to_float(ms.float16)
        self.reward_decoder = DenseDecoder(
            params["reward_decoder_shape"], params["reward_decoder_layers"]
        ).to_float(ms.float16)
        self.value_decoder = DenseDecoder(
            params["value_decoder_shape"], params["value_decoder_layers"]
        ).to_float(ms.float16)
        self.action_decoder = ActionDecoder(params).to_float(ms.float16)
        self.rssm = RSSM(params)
        self.collect_policy = self.CollectPolicy(self.conv_encoder, self.rssm)
        self.init_policy = self.InitPolicy(params)


class DreamerActor(Actor):
    """Dreamer Actor Class"""

    def __init__(self, params):
        super().__init__()
        self.init_policy = params["init_policy"]
        self.collect_policy = params["collect_policy"]
        self.action_decoder = params["action_decoder"]
        self.amount = Tensor(params["expl_amount"], ms.float16)

        self.normal = msd.Normal(dtype=ms.float16)
        self.zeros_like = P.ZerosLike()

        self.true = Tensor(True, ms.bool_)
        self.false = Tensor(False, ms.bool_)
        self.zero_float = Tensor(0, ms.float32)

    def act(self, phase, params):
        obs, prev_mean, prev_std, prev_stoch, prev_deter, prev_action = params
        if phase == 1:
            action, post_mean, post_std, post_stoch, deter = self.init_policy(
                prev_mean, prev_std, prev_stoch, prev_deter
            )
        else:
            feat, post_mean, post_std, post_stoch, deter = self.collect_policy(
                obs, prev_stoch, prev_deter, prev_action
            )
            action = self.get_action(phase, feat)
        return action, post_mean, post_std, post_stoch, deter

    def get_action(self, phase, params):
        feat = params
        action = self.zero_float
        if phase == 2:
            action = self.action_decoder(feat, self.true)
            action = self._exploration(action, self.true)
        if phase == 3:
            action = self.action_decoder(feat, self.false)
            action = self._exploration(action, self.false)
        return action

    def _exploration(self, action, training):
        if training:
            action = C.clip_by_value(self.normal.sample((), action, self.amount), -1, 1)
        return action


class DreamerLearner(Learner):
    """Dreamer Learner"""

    class DreamerDynamicLossCell(nn.Cell):
        """Dreamer dynamic model loss cell (encoder, dynamics, decoder, reward)"""

        def __init__(self, params):
            super().__init__()
            self.rssm = params["rssm"]
            self.conv_encoder = params["conv_encoder"]
            self.conv_decoder = params["conv_decoder"]
            self.reward_decoder = params["reward_decoder"]
            self.free_nats = Tensor(params["free_nats"], ms.float16)
            self.kl_scale = Tensor(params["kl_scale"], ms.float16)
            self.stoch_size = params["stoch_size"]
            self.deter_size = params["deter_size"]

            self.one_float = Tensor(1, ms.float16)
            self.dtype = params["dtype"]

            self.concat = P.Concat(axis=-1)
            self.zeros = P.Zeros()
            self.normal = msd.Normal(dtype=ms.float16)
            self.reduce_mean = P.ReduceMean()
            self.reduce_sum = P.ReduceSum()
            self.maximum = P.Maximum()
            self.multivariate_norm_diag = MultivariateNormalDiag(dtype=ms.float16)

        def construct(self, obs, action, reward):
            """calculate Dreamer dynamic loss"""
            embed = self.conv_encoder(obs)
            # embed, action, start_stoch, start_deter
            start_stoch = self.zeros((obs.shape[0], self.stoch_size), ms.float16)
            start_deter = self.zeros((obs.shape[0], self.deter_size), ms.float16)
            (
                post_mean_tensor,
                post_std_tensor,
                post_stoch_tensor,
                prior_mean_tensor,
                prior_std_tensor,
                _,
                deter_tensor,
            ) = self.rssm.observe(embed, action, start_stoch, start_deter)
            feat = self.concat([post_stoch_tensor, deter_tensor])
            img_mean = self.conv_decoder(feat)
            reward_mean = self.reward_decoder(feat)
            # Separate batch shape and event shape
            independent_img_log_prob = self.reduce_sum(
                self.normal.log_prob(obs, img_mean, self.one_float), [-1, -2, -3]
            )
            img_log_porb = self.reduce_mean(independent_img_log_prob)
            reward_log_prob = self.reduce_mean(
                self.normal.log_prob(reward, reward_mean, self.one_float)
            )
            kl_div = self.multivariate_norm_diag.kl_loss(
                "Normal",
                prior_mean_tensor,
                prior_std_tensor,
                post_mean_tensor,
                post_std_tensor,
            )
            kl_div = kl_div.mean()
            loss_kl_div = self.maximum(kl_div, self.free_nats)
            dynamic_loss = self.kl_scale * loss_kl_div - (
                img_log_porb + reward_log_prob
            )
            return (
                dynamic_loss,
                ops.stop_gradient(post_stoch_tensor),
                ops.stop_gradient(deter_tensor),
            )

    class DreamerActorLossCell(nn.Cell):
        """Dreamer actor model loss cell (actor)"""

        def __init__(self, params):
            super().__init__()
            self.action_decoder = params["action_decoder"]
            self.reward_decoder = params["reward_decoder"]
            self.value_decoder = params["value_decoder"]
            self.rssm = params["rssm"]
            self.discount = Tensor(params["discount"], ms.float16)
            self.gamma = Tensor(params["gamma"], ms.float16)
            self.horizon = params["horizon"]
            self.episode_limits = int(
                params["episode_limits"] / params["action_repeat"]
            )
            self.batch_size = params["batch_size"]
            self.stoch_size = params["stoch_size"]
            self.deter_size = params["deter_size"]

            self.true = Tensor(True, ms.bool_)
            self.zero_float = Tensor(0, ms.float32)
            self.zero_int = Tensor(0, ms.int32)

            self.discounted_return = DiscountedReturn(
                params["gamma"] * params["discount"], need_bprop=True, dtype=ms.float16
            )
            self.reshape = P.Reshape()
            self.ones_like = P.OnesLike()
            self.zeros_like = P.ZerosLike()
            self.concat = P.Concat(axis=-1)
            self.concat_first = P.Concat(axis=0)
            self.stack = P.Stack(axis=0)
            self.cumprod = P.CumProd()
            self.reduce_mean = P.ReduceMean()
            self.zeros = P.Zeros()

        def construct(self, post_stoch, deter):
            """calculate Dreamer actor loss"""
            prev_stoch = self.reshape(post_stoch, (-1,) + post_stoch.shape[2:])
            prev_deter = self.reshape(deter, (-1,) + deter.shape[2:])

            stoch_tensor = self.zeros(
                (self.horizon, self.batch_size * 50, self.stoch_size), ms.float16
            )
            deter_tensor = self.zeros(
                (self.horizon, self.batch_size * 50, self.deter_size), ms.float16
            )

            # stoch_tensor = []
            # deter_tensor = []

            k = self.zero_int
            # k = 0
            while k < self.horizon:
                prev_feat = ops.stop_gradient(self.concat([prev_stoch, prev_deter]))
                prev_action = self.action_decoder(prev_feat, self.true)
                _, _, prev_stoch, prev_deter = self.rssm.img_step(
                    prev_stoch, prev_deter, prev_action
                )
                stoch_tensor[k] = prev_stoch
                deter_tensor[k] = prev_deter
                # stoch_tensor.append(prev_stoch)
                # deter_tensor.append(prev_deter)
                k += 1

            # stoch_tensor = self.stack(stoch_tensor)
            # deter_tensor = self.stack(deter_tensor)

            imagine_feat = self.concat([stoch_tensor, deter_tensor])
            reward = self.reward_decoder(imagine_feat)
            value = self.value_decoder(imagine_feat)
            pcont = self.discount * self.ones_like(reward)
            inputs = reward[:-1] + self.discount * value[1:] * (1 - self.gamma)
            last_value = value[-1]
            done = self.zeros_like(inputs).astype(ms.bool_)
            discounted_return = self.discounted_return(inputs, done, last_value)
            discount = ops.stop_gradient(
                self.cumprod(
                    self.concat_first([self.ones_like(pcont[:1]), pcont[:-2]]), 0
                )
            )
            actor_loss = -self.reduce_mean(discount * discounted_return)
            # Maximize the reward
            return (
                actor_loss,
                ops.stop_gradient(imagine_feat),
                ops.stop_gradient(discounted_return),
                ops.stop_gradient(discount),
            )

    class DreamerValueLossCell(nn.Cell):
        """Dreamer value model loss cell (value)"""

        def __init__(self, params):
            super().__init__()
            self.value_decoder = params["value_decoder"]
            self.reduce_mean = P.ReduceMean()
            self.normal = msd.Normal(dtype=ms.float16)
            self.one_float = Tensor(1, ms.float16)

        def construct(self, imagine_feat, returns, discount):
            """calculate Dreamer value loss"""
            value_pred = self.value_decoder(imagine_feat)
            returns = ops.stop_gradient(returns)
            value_log_prob = self.normal.log_prob(
                returns, value_pred[:-1], self.one_float
            )
            value_loss = -self.reduce_mean(discount * value_log_prob)
            return value_loss

    class DynamicTrainOneStepWithLossScaleCell(nn.TrainOneStepWithLossScaleCell):
        """Train one step cell for dynamic model"""

        def __init__(self, network, optimizer, scale_sense, params):
            super().__init__(network, optimizer, scale_sense)
            self.params = params

        def construct(self, *inputs):
            """Calculate loss and grads for dynamic model"""
            weights = self.weights
            loss, stoch, deter = self.network(*inputs)
            scaling_sens = self.scale_sense

            status, scaling_sens = self.start_overflow_check(loss, scaling_sens)

            scaling_sens_filled = C.ones_like(loss) * F.cast(
                scaling_sens, F.dtype(loss)
            )
            stoch_sens_filled = C.ones_like(stoch) * F.cast(
                scaling_sens, F.dtype(stoch)
            )
            deter_sens_filled = C.ones_like(deter) * F.cast(
                scaling_sens, F.dtype(deter)
            )

            grads = self.grad(self.network, weights)(
                *inputs, (scaling_sens_filled, stoch_sens_filled, deter_sens_filled)
            )
            grads = self.hyper_map(F.partial(_grad_scale, scaling_sens), grads)
            # apply grad reducer on grads
            grads = self.grad_reducer(grads)

            # get the overflow buffer
            cond = self.get_overflow_status(status, grads)
            overflow = self.process_loss_scale(cond)
            return loss, stoch, deter, overflow, grads

    class ActorTrainOneStepWithLossScaleCell(nn.TrainOneStepWithLossScaleCell):
        """Train one step cell for actor model"""

        def __init__(self, network, optimizer, scale_sense, params):
            super().__init__(network, optimizer, scale_sense)
            self.params = params

        def construct(self, *inputs):
            """Calculate loss and grads for actor model"""
            weights = self.weights
            loss, imagine_feat, discounted_return, discount = self.network(*inputs)
            scaling_sens = self.scale_sense

            status, scaling_sens = self.start_overflow_check(loss, scaling_sens)

            scaling_sens_filled = C.ones_like(loss) * F.cast(
                scaling_sens, F.dtype(loss)
            )
            imag_sens_filled = C.ones_like(imagine_feat) * F.cast(
                scaling_sens, F.dtype(imagine_feat)
            )
            dr_sens_filled = C.ones_like(discounted_return) * F.cast(
                scaling_sens, F.dtype(discounted_return)
            )
            discount_sens_filled = C.ones_like(discount) * F.cast(
                scaling_sens, F.dtype(discount)
            )

            grads = self.grad(self.network, weights)(
                *inputs,
                (
                    scaling_sens_filled,
                    imag_sens_filled,
                    dr_sens_filled,
                    discount_sens_filled,
                )
            )
            grads = self.hyper_map(F.partial(_grad_scale, scaling_sens), grads)
            # apply grad reducer on grads
            grads = self.grad_reducer(grads)

            # get the overflow buffer
            cond = self.get_overflow_status(status, grads)
            overflow = self.process_loss_scale(cond)
            return loss, imagine_feat, discounted_return, discount, overflow, grads

    class ValueTrainOneStepWithLossScaleCell(nn.TrainOneStepWithLossScaleCell):
        """Train one step cell for actor model"""

        def __init__(self, network, optimizer, scale_sense, params):
            super().__init__(network, optimizer, scale_sense)
            self.params = params

        def construct(self, *inputs):
            """Calculate loss and grads for value model"""
            weights = self.weights
            loss = self.network(*inputs)
            scaling_sens = self.scale_sense

            status, scaling_sens = self.start_overflow_check(loss, scaling_sens)

            scaling_sens_filled = C.ones_like(loss) * F.cast(
                scaling_sens, F.dtype(loss)
            )

            grads = self.grad(self.network, weights)(*inputs, (scaling_sens_filled))
            grads = self.hyper_map(F.partial(_grad_scale, scaling_sens), grads)
            # apply grad reducer on grads
            grads = self.grad_reducer(grads)

            # get the overflow buffer
            cond = self.get_overflow_status(status, grads)
            overflow = self.process_loss_scale(cond)
            return loss, overflow, grads

    def __init__(self, params):
        super().__init__()
        dynamic_loss_cell = self.DreamerDynamicLossCell(params)
        actor_loss_cell = self.DreamerActorLossCell(params)
        value_loss_cell = self.DreamerValueLossCell(params)
        dynamic_params = (
            params["conv_encoder"].trainable_params()
            + params["rssm"].trainable_params()
            + params["conv_decoder"].trainable_params()
            + params["reward_decoder"].trainable_params()
        )
        optimizer_dynamic = nn.Adam(
            dynamic_params, learning_rate=params["dynamic_lr"], eps=1e-7
        )
        actor_params = params["action_decoder"].trainable_params()
        optimizer_actor = nn.Adam(
            actor_params, learning_rate=params["actor_lr"], eps=1e-7
        )
        value_params = params["value_decoder"].trainable_params()
        optimizer_value = nn.Adam(
            value_params, learning_rate=params["value_lr"], eps=1e-7
        )
        dynamic_loss_scale_update_cell = nn.DynamicLossScaleUpdateCell(
            loss_scale_value=(2**15), scale_factor=2.0, scale_window=2000
        )
        actor_loss_scale_update_cell = nn.DynamicLossScaleUpdateCell(
            loss_scale_value=(2**15), scale_factor=2.0, scale_window=2000
        )
        value_loss_scale_update_cell = nn.DynamicLossScaleUpdateCell(
            loss_scale_value=(2**15), scale_factor=2.0, scale_window=2000
        )
        self.train_dynamic = self.DynamicTrainOneStepWithLossScaleCell(
            dynamic_loss_cell, optimizer_dynamic, dynamic_loss_scale_update_cell, params
        )
        self.train_actor = self.ActorTrainOneStepWithLossScaleCell(
            actor_loss_cell, optimizer_actor, actor_loss_scale_update_cell, params
        )
        self.train_value = self.ValueTrainOneStepWithLossScaleCell(
            value_loss_cell, optimizer_value, value_loss_scale_update_cell, params
        )
        self.l2_loss = P.L2Loss()
        self.hyper_map = C.HyperMap()
        self.square = P.Square()
        self.stack = P.Stack()
        self.sqrt = P.Sqrt()
        self.maximum = P.Maximum()
        self.two_float = Tensor(2, ms.float32)
        self.clip_norm = params.get("clip", 100.0)

    def learn(self, experience):
        """The learn function of dreamer"""
        obs, action, reward, discount = experience

        # Calculate Grads
        (
            dynamic_loss,
            stoch,
            deter,
            dynamic_overflow,
            dynamic_grads,
        ) = self.train_dynamic(obs, action, reward.squeeze(-1))
        (
            actor_loss,
            imagine_feat,
            discounted_return,
            discount,
            actor_overflow,
            actor_grads,
        ) = self.train_actor(stoch, deter)
        value_loss, value_overflow, value_grads = self.train_value(
            imagine_feat, discounted_return, discount
        )

        # Update Net
        if not dynamic_overflow:
            value_loss = F.depend(
                value_loss,
                self.train_dynamic.optimizer(
                    ops.clip_by_global_norm(dynamic_grads, self.clip_norm)
                ),
            )
        if not actor_overflow:
            value_loss = F.depend(
                value_loss,
                self.train_actor.optimizer(
                    ops.clip_by_global_norm(actor_grads, self.clip_norm)
                ),
            )
        if not value_overflow:
            value_loss = F.depend(
                value_loss,
                self.train_value.optimizer(
                    ops.clip_by_global_norm(value_grads, self.clip_norm)
                ),
            )
        total_loss = dynamic_loss + actor_loss + value_loss

        return total_loss
