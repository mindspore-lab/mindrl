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

"""
Generate fragments with python ast.
"""
import ast
import copy
import importlib
import inspect
import os
import shutil

import astor


class GenerateFragment:
    """Class for generate fargments."""

    def __init__(self, algo_name, policy, worker_num, config):
        """
        algo_name : the algo name, used to find the strandard trainer class defined in MSRL.
        policy : distribute policy
        worker_num : worker_num in config.py in `mindspore_rl/algorithm/xx` dir.
        """
        self.src_file = self.get_trainer_by_algo_name(algo_name)
        self.template = self.get_template_by_policy(policy)
        self.policy = policy
        self.worker_num = worker_num
        self.config = config
        self.boundary = self.policy.boundary

    @classmethod
    def get_template_by_policy(cls, policy) -> str:
        """Find the template.tp in the disribution policy dir."""
        file_dir = os.path.dirname(os.path.abspath(inspect.getmodule(policy).__file__))
        src_template = file_dir + "/template.tp"
        template = file_dir + "/template" + str(os.getpid()) + ".py"
        shutil.copy(src_template, template)
        return template

    @classmethod
    def get_trainer_by_algo_name(cls, algo_name) -> str:
        """Find the trainer class for algo."""
        cur = os.path.dirname(__file__)
        trainer_file = (
            cur + "/../../algorithm/" + algo_name + "/" + algo_name + "_trainer.py"
        )
        return trainer_file

    @classmethod
    def generate_ast_from_py(cls, source_py) -> ast.Module:
        """generate ast form source python code"""
        with open(source_py, "r", encoding="utf-8") as fsource:
            source = fsource.read()
        ast_source = ast.parse(source)
        return ast_source

    @classmethod
    # pylint: disable=W0102
    def _find_args_in_node(cls, node_list, args):
        """Inner function, used to find the input and output args, fill the dict."""
        for i in node_list:
            if isinstance(i, (ast.Assign, ast.AugAssign)) and isinstance(
                i.value, ast.Call
            ):
                func_name = i.value.func.attr
                if func_name in args:
                    if isinstance(i, ast.AugAssign):
                        target = i.target
                    else:
                        target = i.targets[0]
                    # xx = msrl.get_buffer() or x1, x2 = msrl.get_buffer()
                    if isinstance(target, ast.Tuple):
                        args[func_name]["output"] = list(
                            map(lambda n: n.id, target.elts)
                        )
                    else:
                        args[func_name]["output"] = target.id
                        # state = env.reset() or loss = msrl.agent_learn(s, r) or loss = msrl.agent_learn(exp)
                    if i.value.args == []:
                        args[func_name]["input"] = None
                    elif isinstance(i.value.args[0], (ast.List, ast.Tuple)):
                        args[func_name]["input"] = list(
                            map(lambda n: n.id, i.value.args[0].elts)
                        )
                    else:
                        args[func_name]["input"] = i.value.args[0].id
            if isinstance(i, ast.Return):
                func_name = "return"
                args[func_name]["output"] = i.value
        return args

    # pylint: disable=W0102
    def find_standard_function_args(
        self,
        ast_source,
        func={
            "get_replay_buffer_elements": {},
            "replay_buffer_sample": {},
            "agent_learn": {},
            "step": {},
            "reset": {},
            "get_action": {},
            "return": {},
        },
    ):
        """
        Search the code to find the standard functions defined in `func` keys.
        Used to get the input and output args which defined by the user.
        """
        # pylint: disable=R1702
        for nodes in ast.iter_child_nodes(ast_source):
            for node in ast.iter_child_nodes(nodes):
                if (
                    isinstance(node, ast.FunctionDef)
                    and node.name == "train_one_episode"
                ):
                    node_list = []
                    for i in node.body:
                        if isinstance(i, ast.While):
                            for n in i.body:
                                node_list.append(n)
                        else:
                            node_list.append(i)
                    args = self._find_args_in_node(node_list, func)
        return args

    @classmethod
    def _find_func(cls, ast_body, func):
        """Find the AugAssign node."""
        for i, node in enumerate(ast_body):
            if isinstance(node, ast.AugAssign) and isinstance(node.value, ast.Call):
                if node.value.func.attr in func:
                    return i, node
        return None

    # pylint: disable=W0102
    def replace_augassign(self, ast_body, learn_func=["agent_learn"]):
        """replace the AugAssign by Assign."""
        if not isinstance(ast_body, list):
            node = self._find_func([ast_body], learn_func)
        else:
            node = self._find_func(ast_body, learn_func)
        if node:
            new_node = ast.Assign(
                lineno=node[1].lineno,
                col_offset=node[1].col_offset,
                targets=[node[1].target],
                value=node[1].value,
            )
            if isinstance(ast_body, list):
                ast_body[node[0]] = new_node
            else:
                ast_body = new_node
        return ast_body

    @classmethod
    # pylint: disable=W0102
    def find_frag(cls, ast_source, func=["agent_learn"]):
        """Find the boundary in func `train_one_episode`, we split this function by the keyword in `func`."""
        actor_part = []
        learner_part = []
        cur_func = ""
        boundary = False
        # pylint: disable=R1702
        for nodes in ast.iter_child_nodes(ast_source):
            for node in ast.iter_child_nodes(nodes):
                if (
                    isinstance(node, ast.FunctionDef)
                    and node.name == "train_one_episode"
                ):
                    for n_body in node.body:
                        insert_body = n_body
                        if (
                            isinstance(n_body, (ast.AugAssign, ast.Assign))
                            and isinstance(n_body.value, ast.Call)
                            and n_body.value.func.attr in func
                        ):
                            print("Find boundary function ", n_body.value.func.attr)
                            boundary = True
                            cur_func = n_body.value.func.attr
                        elif isinstance(n_body, ast.While):
                            for n_n_body in n_body.body:
                                if (
                                    isinstance(n_n_body, (ast.AugAssign, ast.Assign))
                                    and isinstance(n_n_body.value, ast.Call)
                                    and n_n_body.value.func.attr in func
                                ):
                                    print(
                                        "Find boundary function ",
                                        n_n_body.value.func.attr,
                                    )
                                    insert_body = n_n_body
                                    boundary = True
                                    cur_func = n_n_body.value.func.attr
                        if boundary:
                            learner_part.append(insert_body)
                        else:
                            actor_part.append(insert_body)
                        if cur_func == "step" and boundary:
                            n_body.body.remove(insert_body)
                            actor_part.append(n_body)
                            boundary = False
                        elif cur_func == "reset" and boundary:
                            # node.body.remove(n_body)
                            boundary = False
        return actor_part, learner_part

    # pylint: disable=R1702
    def insert_to_target(self, ast_target, fragment_type, frag):
        """Insert fragment in to traget ast code, replace the pass part in template ast."""
        for nodes in ast.iter_child_nodes(ast_target):
            if isinstance(nodes, ast.ClassDef) and nodes.name == fragment_type:
                for i in ast.iter_child_nodes(nodes):
                    if isinstance(i, ast.FunctionDef) and i.name == "kernel":
                        for j in ast.iter_child_nodes(i):
                            if isinstance(j, ast.Pass):
                                i.body.pop(0)
                                if isinstance(frag, list):
                                    frag = (
                                        frag[:-1]
                                        if isinstance(frag[-1], ast.Return)
                                        else frag
                                    )
                                    frag = (
                                        frag[1:]
                                        if self.boundary == "actor-learner"
                                        else frag
                                    )
                                    i.body[-1:-1] = iter(frag)
                                else:
                                    i.body.insert(0, frag)
                    if isinstance(i, ast.FunctionDef) and i.name == "gather_state":
                        for j in ast.iter_child_nodes(i):
                            if isinstance(j, ast.Pass):
                                i.body.pop(0)
                                i.body.insert(0, frag[0])
        ast.fix_missing_locations(ast_target)

    def split_trainer_to_fragment(self, ast_target, ast_source) -> ast.Module:
        """Read ast source of Trainer, split Trainer into Actor and Learner."""
        if self.boundary == "actor-learner":
            func = ["step", "reset"]
        else:
            func = ["agent_learn"]
        actor_frag, learner_frag = self.find_frag(copy.deepcopy(ast_source), func=func)
        if self.boundary == "actor-learner":
            actor_frag, learner_frag = learner_frag, actor_frag
        learner_frag = self.replace_augassign(learner_frag)
        self.add_common_kernel(ast_target)
        self.insert_to_target(ast_target, "Actor", actor_frag)
        self.insert_to_target(ast_target, "Learner", learner_frag)
        return ast_target

    def build_muxsend(
        self,
        frag_type,
        learn_args,
    ):
        """Build the communication nodes for MuxSend/Receive."""
        top_op_nodes = []
        bottom_op_nodes = []
        ast_args = []
        single_out = False
        for arg in learn_args:
            ast_args.append(ast.Name(id=arg, ctx=ast.Load()))
        if len(ast_args) == 1:
            # True:  grad = msrl.agent_act(state)
            # False: states, rewards, actions = msrl.agent_act(state)
            single_out = True
        if frag_type == "Actor":
            # 1 insert cast
            weight_name = "weight"
            for arg in learn_args:
                cast_node = ast.Assign(
                    targets=[ast.Name(id=arg, ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Name(id="self.cast", ctx=ast.Load()),
                        args=[
                            ast.Name(id=arg, ctx=ast.Load()),
                            ast.Name(id="mindspore.float32", ctx=ast.Load()),
                        ],
                        keywords=[],
                    ),
                )
                if not single_out:
                    bottom_op_nodes.append(cast_node)
            # 2 add send
            send_arg = ast.Tuple(elts=ast_args, ctx=ast.Load())
            if single_out:
                send_arg = ast.Name(id=ast_args[0], ctx=ast.Load())
            send_op = ast.Expr(
                value=ast.Call(
                    func=ast.Name(id="self.send", ctx=ast.Load()),
                    args=[send_arg],
                    keywords=[],
                )
            )
            bottom_op_nodes.append(send_op)
            # 3 add receive
            receive_node = ast.Assign(
                targets=[ast.Name(id=weight_name, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id="self.receive", ctx=ast.Load()),
                    args=[],
                    keywords=[],
                ),
            )
            bottom_op_nodes.append(receive_node)
            # 4 add update weight
            update_node = ast.Expr(
                value=ast.Call(
                    func=ast.Name(id="self.update", ctx=ast.Load()),
                    args=[
                        ast.Name(id="self.weight", ctx=ast.Load()),
                        ast.Name(id=weight_name, ctx=ast.Load()),
                    ],
                    keywords=[],
                )
            )
            bottom_op_nodes.append(update_node)
        elif frag_type == "Learner":
            # 1 recive exp
            recv_arg = ast.Tuple(elts=ast_args, ctx=ast.Load())
            if single_out:
                recv_arg = ast.Name(id=ast_args[0], ctx=ast.Load())
            receive_node = ast.Assign(
                targets=[recv_arg],
                value=ast.Call(
                    func=ast.Name(id="self.receive", ctx=ast.Load()),
                    args=[],
                    keywords=[],
                ),
            )
            top_op_nodes.append(receive_node)
            # 2 cast back
            for i, arg in enumerate(learn_args):
                cast_node = ast.Assign(
                    targets=[ast.Name(id=arg, ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Name(id="self.cast", ctx=ast.Load()),
                        args=[
                            ast.Name(id=arg, ctx=ast.Load()),
                            ast.Name(
                                id=str(self.config.get("TYPE")[i]), ctx=ast.Load()
                            ),
                        ],
                        keywords=[],
                    ),
                )
                if not single_out:
                    top_op_nodes.append(cast_node)
            # 3 send weight
            send_op = ast.Expr(
                value=ast.Call(
                    func=ast.Name(id="self.send", ctx=ast.Load()),
                    args=[ast.Name(id="self.weight", ctx=ast.Load())],
                    keywords=[],
                )
            )
            bottom_op_nodes.append(send_op)
        else:
            raise RuntimeError("Fragment {frag_type} is not supported yet.")
        return top_op_nodes, bottom_op_nodes

    @classmethod
    def build_allgather(
        cls,
        frag_type,
        user_defined_buffer_name,
        framework_buffer_name,
        framework_weight_name,
        learn_args,
        learn_return,
    ):
        """
        Build the communication nodes for AllGather.
        For each Fragment type: `Learner` and `Actor`, the behaveior is different in AllGather.
        For the usage of `Actor`, we AllGather the weight of each rank,
        and get the first dim of it(the rank 0 replicate learner).
        the assign the weight into local policy net.These code will add in the top of the Actor part,
        we call it top_op_nodes. Then we will expand first dims for the buffer created by the each actor
        and choose the dims except the first(the rank 0),and AllGather the buffer at last,
        we call these code bottom_op_nodes.
        For the usage of `Learner`, we AllGather all the buffer and strip the first dim as the top_op_nodes.
        Then AllGather the policy net weight after update as the bottom_op_node.
        ######
        Actor:
           weight  -->   local policy
           generate actions and setp in environment to generate buffer.
           broadcast buffer

        Learner:
           allgather buffer
           feed the buffer into agent_learn and update the policy net
           broadcast the weight
        ######

        """
        top_op_nodes = []
        bottom_op_nodes = []
        buffer_name = []
        # User may use a tuple or split variables to obtain buffer.
        # buffer = msrl.get_replay_buffer()
        # s, r, a = msrl.get_replay_buffer()
        if len(user_defined_buffer_name) == 1:
            for i in range(len(framework_buffer_name)):
                buffer_name.append(user_defined_buffer_name[0])
        else:
            buffer_name = user_defined_buffer_name
        weight_head = "self._weight_data"
        net_weight_head = "self.weight"
        buffer_head = "self._buffer_data"
        if frag_type == "Actor":
            # generate top node
            for i, b in enumerate(buffer_name):
                if len(user_defined_buffer_name) == 1:
                    arg = ast.Subscript(
                        value=ast.Name(buffer_name[0], ctx=ast.Load()),
                        slice=ast.Index(value=ast.Num(n=i)),
                        ctx=ast.Load(),
                    )
                else:
                    arg = ast.Name(id=b)
                tmp_buffer_all = "all_buffer_" + str(i)
                src_buffer = buffer_head + "_" + str(i)
                tmp_src_buffer = src_buffer.strip("self._")
                expand_node = ast.Assign(
                    targets=[ast.Name(id=tmp_src_buffer, ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Name(id="self.expanddims", ctx=ast.Load()),
                        args=[ast.Name(id=arg), ast.Num(0)],
                        keywords=[],
                    ),
                )
                assign_node = ast.Assign(
                    targets=[ast.Name(id=src_buffer, ctx=ast.Store())],
                    value=ast.Name(id=tmp_src_buffer),
                    ctx=ast.Load(),
                )
                allgather_node = ast.Assign(
                    targets=[ast.Name(id=tmp_buffer_all, ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Name(id="self.allgather", ctx=ast.Load()),
                        args=[ast.Name(id=src_buffer)],
                        keywords=[],
                    ),
                )
                depend_node = ast.Assign(
                    targets=[ast.Name(id=tmp_buffer_all, ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Name(id="self.depend", ctx=ast.Load()),
                        args=[
                            ast.Name(id=tmp_buffer_all),
                            ast.Name(id="all_buffer_" + str(i - 1)),
                        ],
                        keywords=[],
                    ),
                )
                bottom_op_nodes.append(expand_node)
                bottom_op_nodes.append(assign_node)
                bottom_op_nodes.append(allgather_node)
                if i == 0:
                    depend_node = ast.Assign(
                        targets=[ast.Name(id=tmp_buffer_all, ctx=ast.Store())],
                        value=ast.Call(
                            func=ast.Name(id="self.depend", ctx=ast.Load()),
                            args=[
                                ast.Name(id=tmp_buffer_all),
                                ast.Name(id=tmp_src_buffer),
                            ],
                            keywords=[],
                        ),
                    )
                bottom_op_nodes.append(depend_node)
            # generate bottom node
            for i in range(len(framework_weight_name)):
                src_weight = weight_head + "_" + str(i)
                tmp_data = "all_weight_data_" + str(i)
                dst_weight = net_weight_head + "_" + str(i)
                allgather_node = ast.Assign(
                    targets=[ast.Name(id=tmp_data, ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Name(id="self.allgather", ctx=ast.Load()),
                        args=[ast.Name(id=src_weight)],
                        keywords=[],
                    ),
                )
                assign_node = ast.Assign(
                    targets=[ast.Name(id=dst_weight.strip("self."), ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Name(id="self.assign", ctx=ast.Load()),
                        args=[ast.Name(id=dst_weight), ast.Name(id=tmp_data + "[0,:]")],
                        keywords=[],
                    ),
                )
                depend_node = ast.Assign(
                    targets=[ast.Name(id=tmp_data, ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Name(id="self.depend", ctx=ast.Load()),
                        args=[
                            ast.Name(id=tmp_data),
                            ast.Name(
                                id=net_weight_head.strip("self.") + "_" + str(i - 1)
                            ),
                        ],
                        keywords=[],
                    ),
                )
                if i == 0:
                    depend_node = ast.Assign(
                        targets=[ast.Name(id=tmp_data, ctx=ast.Store())],
                        value=ast.Call(
                            func=ast.Name(id="self.depend", ctx=ast.Load()),
                            args=[
                                ast.Name(id=tmp_data),
                                ast.Name(id="all_buffer_" + str(len(buffer_name) - 1)),
                            ],
                            keywords=[],
                        ),
                    )
                bottom_op_nodes.append(allgather_node)
                bottom_op_nodes.append(depend_node)
                bottom_op_nodes.append(assign_node)

        if frag_type == "Learner":
            exp_node = []
            # generate top node
            for i, b in enumerate(buffer_name):
                if len(learn_args) == 1:
                    learn_arg = learn_args[0] + str(i)
                    exp_node.append(ast.Name(id=learn_arg, ctx=ast.Load()))
                else:
                    learn_arg = learn_args[i]
                tmp_buffer_all = "all_buffer_" + str(i)
                src_buffer = buffer_head + "_" + str(i)
                allgather_node = ast.Assign(
                    targets=[ast.Name(id=tmp_buffer_all, ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Name(id="self.allgather", ctx=ast.Load()),
                        args=[ast.Name(id=src_buffer)],
                        keywords=[],
                    ),
                )
                assign_node = ast.Assign(
                    targets=[ast.Name(id=learn_arg, ctx=ast.Store())],
                    value=ast.Subscript(
                        value=ast.Name(id=tmp_buffer_all, ctx=ast.Load()),
                        slice=ast.Slice(lower=ast.Num(1), upper=None, step=None),
                        ctx=ast.Load(),
                    ),
                )
                depend_node = ast.Assign(
                    targets=[ast.Name(id=tmp_buffer_all, ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Name(id="self.depend", ctx=ast.Load()),
                        args=[
                            ast.Name(id=tmp_buffer_all),
                            ast.Name(id="all_buffer_" + str(i - 1)),
                        ],
                        keywords=[],
                    ),
                )
                reshape_node = ast.Assign(
                    targets=[ast.Name(id=learn_arg, ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Name(id=learn_arg + ".reshape", ctx=ast.Load()),
                        args=[
                            ast.Num(n=-1),
                            ast.Name(id=learn_arg + ".shape[-2]"),
                            ast.Name(id=learn_arg + ".shape[-1]"),
                        ],
                        keywords=[],
                    ),
                )

                top_op_nodes.append(allgather_node)
                top_op_nodes.append(assign_node)
                top_op_nodes.append(reshape_node)
                if i > 0:
                    top_op_nodes.append(depend_node)
            if len(learn_args) == 1:
                assign_node = ast.Assign(
                    targets=[ast.Name(id=learn_arg, ctx=ast.Store())],
                    value=ast.Tuple(elts=exp_node),
                )
                top_op_nodes.append(assign_node)
            # generate bottom node
            depend_node = ast.Assign(
                targets=[ast.Name(id=learn_return[0], ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id="self.depend", ctx=ast.Load()),
                    args=[
                        ast.Name(id=learn_return[0]),
                        ast.Name(id="all_buffer_" + str(len(buffer_name) - 1)),
                    ],
                    keywords=[],
                ),
            )
            bottom_op_nodes.append(depend_node)
            print_node = ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="self", ctx=ast.Load()),
                        attr="print",
                        ctx=ast.Load(),
                    ),
                    args=[
                        ast.Str(s=learn_return[0]),
                        ast.Name(id=learn_return[0], ctx=ast.Load()),
                    ],
                    keywords=[],
                )
            )
            bottom_op_nodes.append(print_node)

            for i in range(len(framework_weight_name)):
                src_weight = net_weight_head + "_" + str(i)
                dst_weight = weight_head + "_" + str(i)
                tmp = "all_weight_data_" + str(i)
                assign_node = ast.Assign(
                    targets=[ast.Name(id=dst_weight, ctx=ast.Store())],
                    value=ast.Name(id=src_weight),
                    ctx=ast.Load(),
                )
                allgather_node = ast.Assign(
                    targets=[ast.Name(id=tmp, ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Name(id="self.allgather", ctx=ast.Load()),
                        args=[ast.Name(id=dst_weight)],
                        keywords=[],
                    ),
                )
                depend_node = ast.Assign(
                    targets=[ast.Name(id=tmp, ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Name(id="self.depend", ctx=ast.Load()),
                        args=[
                            ast.Name(id=tmp),
                            ast.Name(id="all_weight_data_" + str(i - 1)),
                        ],
                        keywords=[],
                    ),
                )
                bottom_op_nodes.append(assign_node)
                bottom_op_nodes.append(allgather_node)
                if i > 0:
                    bottom_op_nodes.append(depend_node)
                else:
                    depend_node = ast.Assign(
                        targets=[ast.Name(id=tmp, ctx=ast.Store())],
                        value=ast.Call(
                            func=ast.Name(id="self.depend", ctx=ast.Load()),
                            args=[ast.Name(id=tmp), ast.Name(id=dst_weight)],
                            keywords=[],
                        ),
                    )
                    bottom_op_nodes.append(depend_node)

        return top_op_nodes, bottom_op_nodes

    def build_broadcast(self, frag_type, function_args):
        """
        Build the communication nodes for Broadcast.
        For each Fragment type: `Learner` and `Actor`, the behaveior is different in Broadcast.
        For the usage of `Actor`, firstly we Allgather the states form each `Actor` to the `Learner`,
        then Broadcast the actions from the `Learner` to each `Actor`, each `Actor` will cut its own action.

        For the usage of `Learner`, we AllGather all the states and strip the first dim as the top_op_nodes.
        Then Broadcast the actions to each `Actor`, AllGather the experiences from each `Actor` and then update
        the network.
        ######
        Actor:
           state  -->  Learner
           get action form Learner
           experience --> Learner

        Learner:
           allgather state
           broadcast action --> Actor
           allgather experience
        ######
        """
        top_op_nodes = []
        bottom_op_nodes = []
        extras = []
        experience = function_args.get("step").get("output")
        experience = list(filter(lambda x: x != "_", experience))
        step_input = function_args.get("step").get("input")
        state = function_args.get("get_action").get("input")
        if frag_type == "Actor":
            allgather_node = ast.Assign(
                targets=[ast.Name(id=state, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id="self.allgather", ctx=ast.Load()),
                    args=[ast.Name(id=state)],
                    keywords=[],
                ),
            )
            extras.append(allgather_node)
            # Top: 1 insert broadcast, all actions from learner
            broadcast_node = ast.Assign(
                targets=[ast.Name(id=step_input, ctx=ast.Store())],
                value=ast.Subscript(
                    value=ast.Call(
                        func=ast.Name(id="self.broadcast", ctx=ast.Load()),
                        args=[
                            ast.Tuple(
                                elts=[
                                    ast.Name(
                                        id="self.action_placeholder", ctx=ast.Load()
                                    ),
                                ],
                                ctx=ast.Load(),
                            ),
                        ],
                        keywords=[],
                    ),
                    slice=ast.Index(
                        value=ast.Num(0),
                        ctx=ast.Load(),
                    ),
                    ctx=ast.Load(),
                ),
            )
            # Top: 2 slice action
            slice_node = ast.Assign(
                targets=[ast.Name(id=step_input, ctx=ast.Store())],
                value=ast.Subscript(
                    value=ast.Name(id=step_input, ctx=ast.Load()),
                    slice=ast.Slice(
                        lower=ast.Name(id="self.index_start"),
                        upper=ast.Name(id="self.index_end"),
                        step=None,
                    ),
                    ctx=ast.Load(),
                ),
            )
            top_op_nodes.append(broadcast_node)
            top_op_nodes.append(slice_node)
            # Bottom: 1 insert allgather for experience
            for i, exp in enumerate(experience):
                allgather_node = ast.Assign(
                    targets=[ast.Name(id=exp, ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Name(id="self.allgather", ctx=ast.Load()),
                        args=[ast.Name(id=exp)],
                        keywords=[],
                    ),
                )
                depend_node = ast.Assign(
                    targets=[ast.Name(id=exp, ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Name(id="self.depend", ctx=ast.Load()),
                        args=[
                            ast.Name(id=exp),
                            ast.Name(id=(step_input if i == 0 else experience[i - 1])),
                        ],
                        keywords=[],
                    ),
                )
                bottom_op_nodes.append(allgather_node)
                bottom_op_nodes.append(depend_node)

        elif frag_type == "Learner":
            # Top: 1 allgather states
            allgather_node = ast.Assign(
                targets=[ast.Name(id=state, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id="self.allgather", ctx=ast.Load()),
                    args=[ast.Name(id="self.placeholder_0")],
                    keywords=[],
                ),
            )
            # Top： 2 split the holder in learner
            split = ast.Assign(
                targets=[ast.Name(id=state, ctx=ast.Store())],
                value=ast.Subscript(
                    value=ast.Name(id=state, ctx=ast.Load()),
                    slice=ast.Slice(
                        lower=ast.Name(id="self.index_start"),
                        upper=None,
                        step=None,
                    ),
                    ctx=ast.Load(),
                ),
            )
            top_op_nodes.append(allgather_node)
            top_op_nodes.append(split)
            # Bottom: broadcast action and gather experience. (need record position)
            action = ast.Assign(
                targets=[ast.Name(id=step_input, ctx=ast.Store())],
                value=ast.Subscript(
                    value=ast.Call(
                        func=ast.Name(id="self.broadcast", ctx=ast.Load()),
                        args=[
                            ast.Tuple(
                                elts=[
                                    ast.Name(id=step_input, ctx=ast.Load()),
                                ],
                                ctx=ast.Load(),
                            ),
                        ],
                        keywords=[],
                    ),
                    slice=ast.Index(
                        value=ast.Num(n=0),
                        ctx=ast.Load(),
                    ),
                    ctx=ast.Load(),
                ),
            )
            bottom_op_nodes.append(action)
            for i, exp in enumerate(experience):
                allgather_node = ast.Assign(
                    targets=[ast.Name(id=exp, ctx=ast.Store())],
                    value=ast.Subscript(
                        value=ast.Call(
                            func=ast.Name(id="self.allgather", ctx=ast.Load()),
                            args=[
                                ast.Name(
                                    id="self." + "placeholder_" + str(i), ctx=ast.Load()
                                )
                            ],
                            keywords=[],
                        ),
                        slice=ast.Slice(
                            lower=ast.Name(id="self.index_start", ctx=ast.Load()),
                            upper=None,
                            step=None,
                        ),
                        ctx=ast.Load(),
                    ),
                )
                depend_node = ast.Assign(
                    targets=[ast.Name(id=exp, ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Name(id="self.depend", ctx=ast.Load()),
                        args=[
                            ast.Name(id=exp),
                            ast.Name(id=(step_input if i == 0 else experience[i - 1])),
                        ],
                        keywords=[],
                    ),
                )
                bottom_op_nodes.append(allgather_node)
                bottom_op_nodes.append(depend_node)
        return top_op_nodes, bottom_op_nodes, extras

    def gen_commuicate_statement(
        self,
        frag_type,
        communication_op,
        function_args,
        framework_buffer_name,
        framework_weight_name,
    ):
        """Genearte communicate statement by the input communicate ops"""
        user_defined_buffer_name = function_args.get("get_replay_buffer_elements").get(
            "output"
        )
        learn_args = function_args.get("agent_learn").get("output")
        learn_return = function_args.get("agent_learn").get("input")
        extras = []
        if "AllGather" in communication_op:
            top_state, bottom_state = self.build_allgather(
                frag_type,
                user_defined_buffer_name,
                framework_buffer_name,
                framework_weight_name,
                learn_args,
                learn_return,
            )
        if "MuxSend" in communication_op:
            top_state, bottom_state = self.build_muxsend(
                frag_type,
                learn_args,
            )
        if "Broadcast" in communication_op:
            top_state, bottom_state, extras = self.build_broadcast(
                frag_type,
                function_args,
            )

        return top_state, bottom_state, extras

    def insert_states(
        self, ast_target, frag_type, top, bottom, extras, top_pos=0, bottom_pos=-1
    ) -> ast.Module:
        """Insert the states part into target ast code."""
        for node in ast.iter_child_nodes(ast_target):
            if isinstance(node, ast.ClassDef) and node.name == frag_type:
                for i in ast.iter_child_nodes(node):
                    if isinstance(i, ast.FunctionDef) and i.name == "kernel":
                        if self.boundary == "actor-learner":
                            for index, n_body in enumerate(i.body):
                                if isinstance(n_body, ast.Assign) and isinstance(
                                    n_body.value, ast.Call
                                ):
                                    func_name = n_body.value.func.attr
                                    if func_name == "get_action":
                                        bottom_pos = index
                                elif isinstance(n_body, ast.While):
                                    for ind, n in enumerate(n_body.body):
                                        if (
                                            isinstance(n, ast.Assign)
                                            and isinstance(n.value, ast.Call)
                                            and n.value.func.attr == "get_action"
                                        ):
                                            i.body[index].body[
                                                ind + 1 : ind + 1
                                            ] = iter(bottom)
                                            bottom = []
                                            break
                        i.body[bottom_pos:bottom_pos] = iter(bottom)
                        i.body[top_pos:top_pos] = iter(top)
                    if (
                        frag_type == "Actor"
                        and isinstance(i, ast.FunctionDef)
                        and i.name == "gather_state"
                    ):
                        i.body[bottom_pos:bottom_pos] = iter(extras)

        return ast_target

    @classmethod
    def get_fragment_types(cls, parameter_list) -> list:
        """Get fragments names in parameter_list."""
        fragment_list = list(parameter_list.keys())
        return fragment_list

    def init_communication(self, ast_target, frag_type, communication_op):
        """
        Init communication, add comunication opertor that not include in common ops.
        If framework_buffer_name is none, we choose learn_args to communicate between actor and learner.
        """
        if self.config.get("DATA") is None:
            raise ValueError(
                "If the DATA not provided in DP, it must be provided in depoly config."
            )
        node = []
        if frag_type == "Learner" and "MuxReceive" in communication_op:
            muxsend = ast.Assign(
                targets=[
                    ast.Attribute(
                        value=ast.Name(id="self", ctx=ast.Load()),
                        attr="receive",
                        ctx=ast.Store(),
                    )
                ],
                value=ast.Call(
                    func=ast.Name(id="MuxReceive", ctx=ast.Load()),
                    args=[],
                    keywords=[
                        ast.keyword(
                            arg="shape",
                            value=ast.Tuple(
                                elts=list(
                                    map(
                                        lambda s: ast.List(
                                            elts=list(map(lambda e: ast.Num(n=e), s))
                                        ),
                                        self.config.get("DATA"),
                                    )
                                ),
                                ctx=ast.Load(),
                            ),
                        ),
                        ast.keyword(
                            arg="dtype",
                            value=ast.Name(id="mindspore.float32", ctx=ast.Load()),
                        ),
                        ast.keyword(
                            arg="group",
                            value=ast.Name(id="NCCL_WORLD_COMM_GROUP", ctx=ast.Load()),
                        ),
                    ],
                ),
            )
            node.append(muxsend)
        for n in ast.iter_child_nodes(ast_target):
            if isinstance(n, ast.ClassDef) and n.name in ["Learner"]:
                for i in ast.iter_child_nodes(n):
                    if isinstance(i, ast.FunctionDef) and i.name == "__init__":
                        i.body[-1:-1] = iter(node)
        return ast_target

    @classmethod
    def replace_return(cls, ast_target, frag_type, new_return):
        """Replace template return with the input new_return."""
        if isinstance(new_return, list):
            return_node = ast.Return(value=ast.Name(id=new_return[0], ctx=ast.Load()))
        else:
            return_node = ast.Return(value=new_return)
        for nodes in ast.iter_child_nodes(ast_target):
            if isinstance(nodes, ast.ClassDef) and nodes.name == frag_type:
                for i in ast.iter_child_nodes(nodes):
                    if isinstance(i, ast.FunctionDef) and i.name == "kernel":
                        for j in ast.iter_child_nodes(i):
                            if isinstance(j, ast.Return):
                                i.body[-1] = return_node
        return ast_target

    def insert_communication_states(
        self, ast_target, ast_source, parameter_list, policy
    ) -> ast.Module:
        """Insert communication states into traget ast code."""
        fragment_type = self.get_fragment_types(parameter_list)
        function_args = self.find_standard_function_args(ast_source)
        learn_return = function_args.get("agent_learn").get("output")
        origin_return = function_args.get("return").get("output")
        framework_buffer_name = policy.communication_data.get("Data")
        framework_weight_name = policy.communication_data.get("Weight")
        for frag_type in fragment_type:
            communication_op = parameter_list[frag_type]["operations"]
            top, bottom, extras = self.gen_commuicate_statement(
                frag_type,
                communication_op,
                function_args,
                framework_buffer_name,
                framework_weight_name,
            )
            if framework_buffer_name is None:
                ast_target = self.init_communication(
                    ast_target, frag_type, communication_op
                )
            ast_target = self.insert_states(ast_target, frag_type, top, bottom, extras)
        if self.boundary == "actor-learner":
            new_return = origin_return
        else:
            new_return = learn_return
        ast_target = self.replace_return(ast_target, "Learner", new_return)
        return ast_target

    @classmethod
    def save_fragment(cls, ast_target) -> str:
        """Save fragment in each process."""
        pid = os.getpid()
        f_name = "Fragments" + str(pid)
        flags = os.O_WRONLY | os.O_CREAT
        with os.fdopen(os.open(f_name + ".py", flags), "w") as f:
            f.write(astor.to_source(ast_target))
        return f_name

    @classmethod
    def clean_files(cls, files):
        """Clean the template files."""
        if isinstance(files, list):
            for file in files:
                os.remove(file)
        else:
            os.remove(files)

    @classmethod
    def add_common_kernel(cls, ast_target):
        """add common kernels in trainer.__init__"""
        kernel_names = {
            "Assign": {},
            "ExpandDims": {},
            "Depend": {},
            "Less": {},
            "Print": {},
        }
        kernel_list = []
        for key, var in kernel_names.items():
            keyword = []
            kernel = str(key)
            if var:
                keyword = [
                    ast.keyword(
                        arg=list(var.keys())[0],
                        value=ast.Name(id=list(var.values())[0], ctx=ast.Load()),
                    )
                ]
            node = ast.Assign(
                targets=[
                    ast.Attribute(
                        value=ast.Name(id="self", ctx=ast.Load()),
                        attr=kernel.lower(),
                        ctx=ast.Store(),
                    )
                ],
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="P", ctx=ast.Load()),
                        attr=kernel,
                        ctx=ast.Load(),
                    ),
                    args=[],
                    keywords=keyword,
                ),
            )
            kernel_list.append(node)
        for n in ast.iter_child_nodes(ast_target):
            if isinstance(n, ast.ClassDef) and n.name in ["Actor", "Learner"]:
                for i in ast.iter_child_nodes(n):
                    if isinstance(i, ast.FunctionDef) and i.name == "__init__":
                        i.body[-1:-1] = iter(kernel_list)

    def interface_parser(self) -> list:
        """Parsing the interface in distribution self.policy"""
        interface_parameters = []
        interfaces = copy.copy(self.policy.interface)
        if self.policy.fuse:
            for fused in self.policy.fuse:
                for key in fused:
                    for f_type in fused[key]:
                        interfaces.pop(f_type)
                    new_interface = {key: self.policy.interface[fused[key][-1]]}
                    interface_parameters.append(new_interface)
            for f_type in interfaces:
                new_interface = {f_type: interfaces[f_type]}
                interface_parameters.append(new_interface)
        else:
            interface_parameters = interfaces
        return interface_parameters

    def create_fragment(self, frag_file=None) -> list:
        """Main function to create fragment."""
        # 1 read source code and transform to ast module. `train.py` and `template.py`
        ast_source = self.generate_ast_from_py(self.src_file)
        ast_target = self.generate_ast_from_py(self.template)
        # 2 Split source func `train_one_episode` in Class Trainer and insert to ast_target.
        ast_target = self.split_trainer_to_fragment(ast_target, ast_source)
        # 3 Get parameter list from distribution policy adn insert communication kernels to ast_target.
        parameter_list = self.interface_parser()
        ast_target = self.insert_communication_states(
            ast_target, ast_source, parameter_list, self.policy
        )
        # 4 Unparse the ast module to python code, and create the module of Actor and Learner.
        if frag_file is None:
            frag_name = self.save_fragment(ast_target)
        else:
            frag_name = frag_file.strip(".py")
            print(f"Import fragment from file: {frag_name}.")
        fragment_module = importlib.import_module(frag_name)
        actor = getattr(fragment_module, "Actor")
        learner = getattr(fragment_module, "Learner")
        # 5 Create the fragments list across to the policy topology.
        fragment_list = []
        if list(self.policy.topology.keys())[0] == "Actor":
            for _ in range(self.worker_num - 1):
                fragment_list.append(actor)
                fragment_list.append(learner)
        else:
            fragment_list.append(learner)
            for _ in range(self.worker_num - 1):
                fragment_list.append(actor)
        print("Fragments generated: ", fragment_list)
        self.clean_files([self.template])
        return fragment_list
