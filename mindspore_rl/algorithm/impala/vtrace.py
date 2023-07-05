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

"""Functions to compute V-trace off-policy actor critic targets.

    For details and theory see:

    "IMPALA: Scalable Distributed Deep-RL with
    Importance Weighted Actor-Learner Architectures"
    by Espeholt, Soyer, Munos et al.

    See https://arxiv.org/abs/1802.01561 for the full paper.
"""

import mindspore as ms
from mindspore import Tensor, nn, ops


def log_probs_logits(logits, actions):
    """Compute log-probs from policy logits and actions"""
    # [t,b,c]
    logits = ops.transpose(logits, (1, 2, 0))
    actions = ops.transpose(actions, (1, 0))
    cross = nn.CrossEntropyLoss(reduction="none")
    return -ops.transpose(cross(logits, actions), (1, 0))


def get_importance_weight(behavior_policy_logits, target_policy_logits, actions):
    """Calculate log_rho according to the paper https://arxiv.org/abs/1802.01561"""
    target_log_probs = log_probs_logits(target_policy_logits, actions)
    behavior_log_probs = log_probs_logits(behavior_policy_logits, actions)
    log_rhos = target_log_probs - behavior_log_probs
    return log_rhos


def get_vs_and_advantages(
    log_rhos,
    discounts,
    rewards,
    values,
    bootstrap_value,
    clip_rho_threshold,
    clip_pg_threshold,
    clip_cs_threshold,
    masks,
):
    r"""V-trace for softmax policies.

    Calculates V-trace actor critic targets for softmax polices

    Computing v_s recursively using:

    # Math: v_s = V(x_s)+\delta_sV+\gamma c_s(v_{s+1}-V(x_{s+1}))
    # Math: \delta_sV = \rho(r_t+\gamma V(x_{t+1})-V(x_t))

    Computing Policy Gradient by
    # Math: \rho_s\cdot (r_s+\gamma v_{s+1}-V(x_s))\cdot \nabla \log \pi(a_s|x_s)
    """

    if clip_rho_threshold is not None:
        clip_rho_threshold = Tensor(clip_rho_threshold, dtype=ms.float32)
    if clip_pg_threshold is not None:
        clip_pg_threshold = Tensor(clip_pg_threshold, dtype=ms.float32)
    if clip_cs_threshold is not None:
        clip_cs_threshold = Tensor(clip_cs_threshold, dtype=ms.float32)

    rhos = ops.exp(log_rhos)
    rhos = ops.minimum(rhos, clip_rho_threshold)
    cs = ops.minimum(rhos, clip_cs_threshold)
    pg_rhos = ops.minimum(rhos, clip_pg_threshold)

    values_1 = ops.concat([values[1:], ops.expand_dims(bootstrap_value, 0)], axis=0)

    deltas = rhos * (rewards + discounts * values_1 - values)

    vs_minus_v_xs = [ops.zeros_like(bootstrap_value)]
    i = len(discounts) - 1
    while i >= 0:
        discount_t, c_t, delta_t = discounts[i], cs[i], deltas[i]
        vs_minus_v_xs.append(delta_t + discount_t * c_t * vs_minus_v_xs[-1] * masks[i])
        i -= 1

    vs_minus_v_xs = ops.stack(vs_minus_v_xs[1:])
    # Reverse the results back to original order.
    vs_minus_v_xs = ops.flip(vs_minus_v_xs, dims=[0])
    # Add V(x_s) to get v_s.
    vs = vs_minus_v_xs + values

    # Advantage

    target_v_1 = ops.concat([vs[1:], ops.expand_dims(bootstrap_value, 0)], axis=0)

    pg_advantage = pg_rhos * (rewards + discounts * target_v_1 - values)

    pg_advantage = ops.stop_gradient(pg_advantage)
    return ops.stop_gradient(vs), ops.stop_gradient(pg_advantage)
