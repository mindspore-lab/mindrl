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
"""Advantage Actor Critic"""

import argparse

import mindspore
import mindspore.nn as nn
import mindspore.nn.probability.distribution as msd
import mindspore.ops as ops
import numpy as np
from mindspore import Tensor, context, ms_function
from mindspore.common.parameter import ParameterTuple
from mindspore.ops import composite as C
from mindspore.ops import operations as P

from mindspore_rl.agent import trainer
from mindspore_rl.agent.actor import Actor
from mindspore_rl.agent.learner import Learner
from mindspore_rl.agent.trainer import Trainer
from mindspore_rl.core import Session
from mindspore_rl.environment import GymEnvironment
from mindspore_rl.utils import BatchRead, BatchWrite, DiscountedReturn, TensorArray

SEED = 42
np.random.seed(SEED)


class A2CPolicyAndNetwork:
    """A2CPolicyAndNetwork"""

    class ActorCriticNet(nn.Cell):
        """ActorCriticNet"""

        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.common = nn.Dense(input_size, hidden_size, weight_init="XavierUniform")
            self.actor = nn.Dense(hidden_size, output_size, weight_init="XavierUniform")
            self.critic = nn.Dense(hidden_size, 1, weight_init="XavierUniform")
            self.relu = nn.LeakyReLU()

        def construct(self, x):
            x = self.common(x)
            x = self.relu(x)
            return self.actor(x), self.critic(x)

    def __init__(self, params):
        self.a2c_net = self.ActorCriticNet(
            params["state_space_dim"], params["hidden_size"], params["action_space_dim"]
        )

        self.a2c_net_learn = self.ActorCriticNet(
            params["state_space_dim"], params["hidden_size"], params["action_space_dim"]
        )
        self.a2c_net_copy = ParameterTuple(self.a2c_net_learn.trainable_params())


# pylint: disable=W0223
class A2CActor(Actor):
    """A2C Actor"""

    class Loss(nn.Cell):
        """Actor-Critic loss"""

        def __init__(self, a2c_net):
            super().__init__(auto_prefix=False)
            self.a2c_net = a2c_net
            self.reduce_sum = ops.ReduceSum(keep_dims=False)
            self.log = ops.Log()
            self.gather = ops.GatherD()
            self.softmax = ops.Softmax()
            self.smoothl1_loss = nn.SmoothL1Loss(beta=1.0, reduction="sum")

        def construct(self, states, actions, returns):
            """Calculate actor loss and critic loss"""
            action_logits_ts, values = self.a2c_net(states)
            action_probs_t = self.softmax(action_logits_ts)
            action_probs = self.gather(action_probs_t, 1, actions)
            advantage = returns - values
            action_log_probs = self.log(action_probs)
            adv_mul_prob = action_log_probs * advantage
            actor_loss = -self.reduce_sum(adv_mul_prob)
            critic_loss = self.smoothl1_loss(values, returns)
            return critic_loss + actor_loss

    def __init__(self, params=None):
        super(A2CActor, self).__init__()
        self._params_config = params
        self.a2c_net = params["a2c_net"]
        self.local_param = self.a2c_net.trainable_params()
        self._environment = params["collect_environment"]
        loop_size = 200
        self.loop_size = Tensor(loop_size, mindspore.int64)
        self.c_dist = msd.Categorical(dtype=mindspore.float32, seed=SEED)
        self.expand_dims = P.ExpandDims()
        self.reshape = P.Reshape()
        self.cast = P.Cast()
        self.softmax = ops.Softmax()
        # added manual need automate
        self.pull = BatchRead()
        self.depend = ops.Depend()
        self.zero = Tensor(0, mindspore.int64)
        self.done = Tensor(True, mindspore.bool_)
        self.states = TensorArray(
            mindspore.float32, (4,), dynamic_size=False, size=loop_size
        )
        self.actions = TensorArray(
            mindspore.int32, (1,), dynamic_size=False, size=loop_size
        )
        self.rewards = TensorArray(
            mindspore.float32, (1,), dynamic_size=False, size=loop_size
        )
        self.masks = Tensor(np.zeros([loop_size, 1], dtype=np.bool_), mindspore.bool_)
        self.mask_done = Tensor([1], mindspore.bool_)
        self.print = P.Print()
        self.discount_return = DiscountedReturn(gamma=self._params_config["gamma"])
        self.loss_net = self.Loss(self.a2c_net)
        self.grad = C.GradOperation(get_by_list=True, sens_param=False)
        self.local_weight = ParameterTuple(self.local_param)
        self.moments = nn.Moments(keep_dims=False)
        self.sqrt = ops.Sqrt()
        self.zero_float = Tensor([0.0], mindspore.float32)
        self.epsilon = Tensor(1.1920929e-07, mindspore.float32)

    # pylint: disable=W0221
    def act(self, phase, actor_id=0, weight_copy=None):
        """Store returns into TensorArrays from env"""
        if phase == 2:
            params = self._environment[actor_id].reset()
            update = self.pull(self.local_weight, weight_copy)

            t = self.zero
            done_status = self.zero
            done_num = self.zero
            masks = self.masks
            while t < self.loop_size:
                self.states.write(t, params)
                ts0 = self.expand_dims(params, 0)
                action_logits, _ = self.a2c_net(ts0)
                action_logits = self.depend(action_logits, update)
                action_probs_t = self.softmax(action_logits)
                action = self.reshape(
                    self.c_dist.sample((1,), probs=action_probs_t), (1,)
                )
                action = self.cast(action, mindspore.int32)
                self.actions.write(t, action)
                new_state, reward, done = self._environment[actor_id].step(action)
                self.rewards.write(t, reward)
                params = new_state
                if done == self.done:
                    if done_status == self.zero:
                        done_status += 1
                        done_num = t
                    masks[t] = self.mask_done
                    self._environment[actor_id].reset()
                t += 1
            rewards = self.rewards.stack()
            states = self.states.stack()
            actions = self.actions.stack()
            self.rewards.clear()
            self.states.clear()
            self.actions.clear()
            # compute local loss and grads
            returns = self.discount_return(rewards, masks, self.zero_float)
            adv_mean, adv_var = self.moments(returns)
            normalized_returns = (returns - adv_mean) / (
                self.sqrt(adv_var) + self.epsilon
            )
            loss = self.loss_net(states, actions, normalized_returns)
            grads = self.grad(self.loss_net, self.local_weight)(
                *(states, actions, normalized_returns)
            )
            return done_num, grads, loss
        self.print("Phase is incorrect")
        return 0


class A2CLearner(Learner):
    """A2C Learner"""

    def __init__(self, params):
        super(A2CLearner, self).__init__()
        self._params_config = params
        self.a2c_net = params["a2c_net"]
        self.global_weight = self.a2c_net.trainable_params()
        self.global_params = ParameterTuple(self.global_weight)
        self.optimizer = nn.Adam(self.global_weight, learning_rate=params["lr"])

    # pylint: disable=W0221
    def learn(self, grads):
        """update"""
        success = self.optimizer(grads)
        return success


class A2CTrainer(Trainer):
    """trainer"""

    def __init__(self, msrl):
        super(A2CTrainer, self).__init__(msrl)
        self.reduce_sum = ops.ReduceSum()
        self.actor_nums = msrl.num_actors
        self.learner_rank = self.actor_nums
        self.weight_copy = msrl.learner.global_weight
        shapes = []
        for i in self.weight_copy:
            shapes.append(i.shape)
        self.zero = mindspore.Tensor(0, mindspore.float32)

        # For actors.
        # Create a shared send op, each actor will send the grads to the same target.
        # Create receive op for each actor which will receive the updated weights from learner.
        self.send_actor = MuxSend(
            dest_rank=self.learner_rank, group=NCCL_WORLD_COMM_GROUP
        )
        self.recv_actor = MuxReceive(
            shape=shapes, dtype=mindspore.float32, group=NCCL_WORLD_COMM_GROUP
        )

        # For learner.
        # There is only one learner for A3C, the recv op will receive the grads form actors.
        # The learner will update the specific actor, depending on which actor the gradient comes from.
        self.recv_learner = MuxReceive(
            shape=shapes, dtype=mindspore.float32, group=NCCL_WORLD_COMM_GROUP
        )
        self.send_learner = MuxSend(dest_rank=-1, group=NCCL_WORLD_COMM_GROUP)

        self.depend = ops.Depend
        self.update = BatchWrite()

    # pylint: disable=W0613
    def train(self, episodes, callbacks=None, ckpt_path=None):
        """Train A2C"""
        # For episode in learner, it is triggered by each actor, so there is a acotor_num-fold expansion
        if rank_id == self.learner_rank:
            episodes *= self.actor_nums
        for i in range(episodes):
            one_step = self.train_one_episode()
            if rank_id != self.learner_rank:
                print(
                    f"Train in actor {rank_id}, episode {i}, rewards {one_step[0].asnumpy()}, "
                    f"loss {one_step[2].asnumpy()}"
                )

    @mindspore.jit
    def train_one_episode(self):
        # actors
        _, grads, _ = self.msrl.agent_act(trainer.COLLECT, self.weight_copy)
        result = self.msrl.agent_learn(grads)
        return result


parser = argparse.ArgumentParser(description="MindSpore Reinforcement A3C")
parser.add_argument("--episode", type=int, default=1000, help="total episode numbers.")
parser.add_argument(
    "--device_target",
    type=str,
    default="Auto",
    choices=["CPU", "GPU", "Auto"],
    help="Choose a device to run the a3c example(Default: Auto).",
)
parser.add_argument("--actor_num", type=int, default=3, help="actor number")
parser.add_argument(
    "--env_yaml",
    type=str,
    default="../env_yaml/CartPole-v0.yaml",
    help="Choose an environment yaml to update the a3c example(Default: CartPole-v0.yaml).",
)
parser.add_argument(
    "--algo_yaml",
    type=str,
    default=None,
    help="Choose an algo yaml to update the a3c example(Default: None).",
)
options, _ = parser.parse_known_args()

collect_env_params = {
    "name": "CartPole-v0",
    "seed": 42,
}
eval_env_params = {"name": "CartPole-v0"}
policy_params = {
    "hidden_size": 128,
    "gamma": 0.99,
}
learner_params = {
    "lr": 0.01,
}
algorithm_config = {
    "actor": {
        "number": 1,
        "type": A2CActor,
        "params": policy_params,
        "policies": [],
        "networks": ["a2c_net"],
    },
    "learner": {
        "number": 1,
        "type": A2CLearner,
        "params": learner_params,
        "networks": ["a2c_net"],
    },
    "policy_and_network": {"type": A2CPolicyAndNetwork, "params": policy_params},
    "collect_environment": {
        "number": 1,
        "type": GymEnvironment,
        "params": collect_env_params,
    },
    "eval_environment": {
        "number": 1,
        "type": GymEnvironment,
        "params": collect_env_params,
    },
}

deploy_config = {
    "auto_distribution": True,
    "distribution_policy": "AsyncMultiActorSingleLearnerDP",
    "worker_num": 2,
    "config": {},
}


class A2CSession(Session):
    """A2C session"""

    def __init__(self, env_yaml=None, algo_yaml=None):
        update_config(config, env_yaml, algo_yaml)
        super().__init__(config.algorithm_config)


def train(episode=options.episode):
    if options.device_target != "Auto":
        context.set_context(device_target=options.device_target)
    context.set_context(mode=context.GRAPH_MODE)
    try:
        algorithm_config["actor"]["number"] = options.actor_num
    except KeyError:
        print("Key doesn't exist")
    ac_session = Session(algorithm_config, deploy_config)
    ac_session.run(class_type=A2CTrainer, episode=episode)


if __name__ == "__main__":
    train()
