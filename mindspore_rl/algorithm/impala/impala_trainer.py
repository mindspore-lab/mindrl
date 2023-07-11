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
"""IMPALA Trainer"""
import os

import mindspore as ms
from mindspore import Tensor, context, ops
from mindspore.communication.management import (
    NCCL_WORLD_COMM_GROUP,
    get_group_size,
    get_rank,
    init,
)
from mindspore.ops.operations._rl_inner_ops import MuxReceive, MuxSend
from mindspore.parallel._ps_context import _is_role_sched

from mindspore_rl.agent import trainer
from mindspore_rl.agent.trainer import Trainer
from mindspore_rl.utils import BatchWrite

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
init("nccl")
rank_id = get_rank()
rank_size = get_group_size()


class IMPALATrainer(Trainer):
    """IMPALATrainer"""

    def __init__(self, msrl, params):
        super().__init__(msrl)
        self.actor_nums = msrl.actors.__len__()
        self.learner_rank = self.actor_nums
        self.weight_copy = msrl.learner.global_weight
        self.batch_size = params["batch_size"]
        self.time_scope = params["loop_size"]
        self.state_space_dim = params["state_space_dim"]
        self.action_space_dim = params["action_space_dim"]
        self.num_evaluate_episode = params["num_evaluate_episode"]
        self.less = ops.Less()
        self.false = Tensor((False,), ms.bool_)
        self.true = Tensor((True,), ms.bool_)
        self.zero = ms.Tensor(0, ms.float32)

        actor_shape = [
            (self.time_scope, self.batch_size, self.state_space_dim),
            (self.time_scope, self.batch_size),
            (self.time_scope, self.batch_size),
            (self.time_scope, self.batch_size, self.action_space_dim),
            (self.time_scope, self.batch_size),
        ]

        params_shape = []
        for i in self.weight_copy:
            params_shape.append(i.shape)

        # For actors.
        # Create a shared send op, each actor will send a trajectory tuple (x_t,a_t,r_t,u(a_t|x_t)) to learner.
        # Create receive op for each actor which will receive the updated weights from learner.
        self.send_actor = MuxSend(
            dest_rank=self.learner_rank, group=NCCL_WORLD_COMM_GROUP
        )
        self.recv_actor = MuxReceive(
            shape=params_shape, dtype=ms.float32, group=NCCL_WORLD_COMM_GROUP
        )

        # For learner.
        # Receiver will receive trajectory tuple from actors (x_t,a_t,r_t,u(a_t|x_t)).
        # The learner will update the specific actor, depending on which actor the trajectory comes from.
        self.send_learner = MuxSend(dest_rank=-1, group=NCCL_WORLD_COMM_GROUP)
        self.recv_learner = MuxReceive(
            shape=actor_shape, dtype=ms.float32, group=NCCL_WORLD_COMM_GROUP
        )

        self.depend = ops.Depend
        self.update = BatchWrite()
        self.print = ops.Print()

    # pylint: disable=W0613
    def train(self, episodes, callbacks=None, ckpt_path=None):
        """Train IMPALA"""

        if rank_id == self.learner_rank:
            episodes *= self.actor_nums
        for i in range(episodes):
            result = self.train_one_episode()
            if _is_role_sched():
                # pylint: disable=W0212
                os._exit(0)
            if rank_id == self.learner_rank:
                print(
                    f"Train from one actor, episode {int(i/self.actor_nums)}, loss {result}",
                    flush=True,
                )
            elif i % 10 == 0:
                print(f"Evaluating in actor {rank_id}", flush=True)
                avg = self.evaluate()
                print(f"evaluate in actor {rank_id}, avg_reward {avg}", flush=True)

    @ms.jit
    def train_one_episode(self):
        """Train one episode"""

        result = 0
        if rank_id == self.learner_rank:
            result = self.msrl.agent_learn(self.recv_learner())
            self.send_learner(self.msrl.learner.global_params)
        else:
            (states, actions, rewards, policys, masks) = self.msrl.actors[rank_id].act(
                trainer.COLLECT, actor_id=rank_id, weight_copy=self.weight_copy
            )

            actions = actions.astype(ms.float32)
            masks = masks.astype(ms.float32)
            self.send_actor((states, actions, rewards, policys, masks))
            weight = self.recv_actor()
            self.update(self.weight_copy, weight)

        return result

    @ms.jit
    def evaluate(self):
        if rank_id != self.learner_rank:
            total_reward = self.zero
            eval_iter = self.zero
            while self.less(eval_iter, self.num_evaluate_episode):
                episode_reward = self.zero
                state = self.msrl.eval_environment.reset()
                done = self.false
                while not done:
                    state, r, done = self.msrl.actors[rank_id].act(
                        trainer.EVAL, actor_id=id, eval_state=state
                    )
                    episode_reward += r
                total_reward += episode_reward
                eval_iter += 1
            avg_reward = total_reward / self.num_evaluate_episode
            return avg_reward
        return 0

    def trainable_variables(self):
        """Trainable variables for saving."""
        return self.msrl.learner.global_params
