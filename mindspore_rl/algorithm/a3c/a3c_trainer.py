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
"""A3C Trainer"""
from mindspore_rl.agent.trainer import Trainer
from mindspore_rl.agent import trainer
from mindspore_rl.utils import  BatchWrite
import mindspore
from mindspore.ops.operations._rl_inner_ops import MuxSend, MuxReceive
from mindspore.communication.management import init, NCCL_WORLD_COMM_GROUP, get_rank, get_group_size
from mindspore.ops import operations as ops
from mindspore import context

# Dynamic networking is only supported in GRAPH_MODE, but the default mode in MS is PYNATIVE_MODE.
context.set_context(mode=context.GRAPH_MODE)
init()
rank_id = get_rank()
rank_size = get_group_size()


class A3CTrainer(Trainer):
    '''A3CTrainer'''
    def __init__(self, msrl):
        super(A3CTrainer, self).__init__(msrl)
        self.reduce_sum = ops.ReduceSum()
        self.actor_nums = msrl.actors.__len__()
        self.learner_rank = self.actor_nums
        self.weight_copy = msrl.learner.global_weight
        shapes = []
        for i in self.weight_copy:
            shapes.append(i.shape)
        self.zero = mindspore.Tensor(0, mindspore.float32)

        # For actors.
        # Create a shared send op, each actor will send the grads to the same target.
        # Create receive op for each actor which will receive the updated weights from learner.
        self.send_actor = MuxSend(dest_rank=self.learner_rank, group=NCCL_WORLD_COMM_GROUP)
        self.recv_actor = MuxReceive(shape=shapes, dtype=mindspore.float32, group=NCCL_WORLD_COMM_GROUP)

        # For learner.
        # There is only one learner for A3C, the recv op will receive the grads form actors.
        # The learner will update the specific actor, depending on which actor the gradient comes from.
        self.recv_learner = MuxReceive(shape=shapes, dtype=mindspore.float32, group=NCCL_WORLD_COMM_GROUP)
        self.send_learner = MuxSend(dest_rank=-1, group=NCCL_WORLD_COMM_GROUP)

        self.depend = ops.Depend
        self.update = BatchWrite()

    #pylint: disable=W0613
    def train(self, episodes, callbacks=None, ckpt_path=None):
        '''Train A3C'''
        # For episode in learner, it is triggered by each actor, so there is a acotor_num-fold expansion
        if rank_id == self.learner_rank:
            episodes *= self.actor_nums
        for i in range(episodes):
            one_step = self.train_one_episode()
            if rank_id != self.learner_rank:
                print(f"Train in actor {rank_id}, episode {i}, rewards {one_step[0].asnumpy()}, "
                      f"loss {one_step[2].asnumpy()}")

    @mindspore.jit
    def train_one_episode(self):
        '''Train one episode'''
        # actors
        result = (self.zero, self.zero, self.zero)  # rewards, grads, loss
        if rank_id == self.learner_rank:# learner
            grads = self.recv_learner()
            self.msrl.agent_learn(grads)
            self.send_learner(self.msrl.learner.global_params)
        elif rank_id != self.learner_rank:
            result = self.msrl.actors[rank_id].act(trainer.COLLECT, actor_id=rank_id,
                                                   weight_copy=self.weight_copy)
            self.send_actor(result[1])
            weight = self.recv_actor()
            self.update(self.weight_copy, weight)
        return result

    def evaluate(self):
        '''Default evaluate'''
        return

    def trainable_variables(self):
        '''Default trainable variables'''
        return
