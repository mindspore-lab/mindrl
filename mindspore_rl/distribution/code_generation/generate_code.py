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
Generate fragments.
"""

import ast
import os
import stat
import sys
import importlib


#pylint: disable=E1123
def _transfor_class_name(target, fragment_names):
    '''Convert class name'''
    found = 0
    with open(target, 'r', encoding='utf-8') as template:
        lines = template.readlines()
        for name in fragment_names:
            cnt = 0
            for line_no, _ in enumerate(lines):
                line = lines[line_no]
                if '$fragment_name$' in line:
                    new_line = line.replace('$fragment_name$', name)
                    lines[line_no] = new_line
                    cnt += 1
                    found += cnt
                    if cnt == 2:
                        break

    if found != 4:
        tmp_file = "template_tmp.py"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        modes = stat.S_IWUSR | stat.S_IRUSR
        with os.fdopen(os.open(tmp_file, flags, modes), 'w') as fout:
            fout.writelines(lines)
        with open(tmp_file, 'r', encoding="utf-8") as target_out:
            target_out = target_out.read()
            ast_target = ast.parse(target_out, type_comments=True)
    else:
        print("error, the template does not contain $fragment_name$ place holder.")
    return ast_target


def _copy_statement(ast_source):
    statements = []
    for i in ast.iter_child_nodes(ast_source):
        if not isinstance(i, ast.Expr) and not isinstance(i, ast.arguments):
            statements.insert(0, i)
    return statements


def _copy_constructor(ast_target, ast_source, fragment_name):
    '''copy constructor from src to target'''
    statements = _copy_statement(ast_source)
    cnt = 0
    for n in ast.iter_child_nodes(ast_target):
        if isinstance(n, ast.ClassDef) and n.name == fragment_name:
            for i in ast.iter_child_nodes(n):
                if isinstance(i, ast.FunctionDef) and i.name == '__init__':
                    for _ in ast.iter_child_nodes(i):
                        cnt = cnt + 1
                    cnt -= 1
                    for statement in statements:
                        i.body.insert(cnt, statement)


def _build_receiver(f_type):
    '''build receive statement'''
    op_nodes = []

    op_list = ['Receive', 'Send']
    for op in op_list:
        if op == 'Receive' and f_type == 'Actor':
            irecv_node = ast.Assign(targets=[ast.Name(id='success', ctx=ast.Store())],\
                                    value=ast.Call(func=ast.Name(id='self.update', ctx=ast.Load()),\
                                    args=[ast.Name(id='self.weight'), ast.Name('weight')], keywords=[]))
            op_nodes.append(irecv_node)
            irecv_node = ast.Assign(targets=[ast.Name(id='weight', ctx=ast.Store())],\
                                    value=ast.Call(func=ast.Name(id='self.recv_actor', ctx=ast.Load()),\
                                    args=[], keywords=[]))
            op_nodes.append(irecv_node)
        if op == 'Receive' and f_type == 'Learner':
            irecv_node = ast.Assign(targets=[ast.Name(id='grads', ctx=ast.Store())],\
                                    value=ast.Call(func=ast.Name(id='self.recv_learner', ctx=ast.Load()),\
                                    args=[], keywords=[]))
            op_nodes.append(irecv_node)
    return op_nodes


def _build_sender(f_type):
    '''build send statement'''
    op_nodes = []
    op_list = ['Receive', 'Send']

    for op in op_list:
        if op == 'Send' and f_type == 'Actor':
            isend_node = ast.Assign(targets=[ast.Name(id='send_res', ctx=ast.Store())],\
                                    value=ast.Call(func=ast.Name(id='self.send_actor',\
                                    ctx=ast.Load()), args=[ast.Name(id='grads')], keywords=[]))
            op_nodes.append(isend_node)
        if op == 'Send' and f_type == 'Learner':
            isend_node = ast.Assign(targets=[ast.Name(id='send_res', ctx=ast.Store())],\
                                value=ast.Call(func=ast.Name(id='self.send_learner', ctx=ast.Load()),\
                                args=[ast.Name(id='self.msrl.learner.global_params')], keywords=[]))
            op_nodes.append(isend_node)
    return op_nodes


def _create_tuple(data):
    '''create tuple AST'''
    data_list = []
    for i in data:
        if isinstance(i, int):
            data_list.append(ast.Constant(value=i))
        else:
            data_list.append(ast.Name(id=i))
    tuple_arg = ast.Tuple(elts=data_list, ctx=ast.Load())
    return tuple_arg


def _build_allgather_send(f_type):
    '''build allgather on sender side'''
    op_nodes = []
    if f_type == 'Actor':
        data_list = ['self.state_list', 'self.reward_list', 'self.action_list',
                     'self.next_state_list', 'self.miu_list', 'self.sigma_list']
        tmp_data_list = ['state_list', 'reward_list', 'action_list',
                         'next_state_list', 'miu_list', 'sigma_list']
        for i, _ in enumerate(data_list):
            data = data_list[i]
            tmp_data = tmp_data_list[i]
            if i > 0:
                dep_data = tmp_data_list[i-1]
                depend_node = ast.Assign(targets=[ast.Name(id=data, ctx=ast.Store())],\
                                         value=ast.Call(func=ast.Name(id='self.depend',\
                                         ctx=ast.Load()), args=[ast.Name(id=data), ast.Name(id=dep_data)],\
                                         keywords=[]))
                op_nodes.append(depend_node)

            expand_node = ast.Assign(targets=[ast.Name(id=tmp_data, ctx=ast.Store())],\
                                     value=ast.Call(func=ast.Name(id='self.expand_dims',\
                                     ctx=ast.Load()), args=[ast.Name(id=data), ast.Num(0)], keywords=[]))
            op_nodes.append(expand_node)

            allgather_node = ast.Assign(targets=[ast.Name(id=tmp_data, ctx=ast.Store())],\
                                        value=ast.Call(func=ast.Name(id='self.allgather',\
                                        ctx=ast.Load()), args=[ast.Name(id=tmp_data)], keywords=[]))
            op_nodes.append(allgather_node)

    if f_type == 'Learner':
        depend_node = ast.Assign(targets=[ast.Name(id='actor_net_param', ctx=ast.Store())],\
                                 value=ast.Call(func=ast.Name(id='self.depend', ctx=ast.Load()),\
                                 args=[ast.Name(id='self.actor_net_param'),\
                                 ast.Name(id='self.training_loss')], keywords=[]))
        op_nodes.append(depend_node)

        data_list = ['self.actor_net_param[0]', 'self.actor_net_param[1]',
                     'self.actor_net_param[2]', 'self.actor_net_param[3]',
                     'self.actor_net_param[4]', 'self.actor_net_param[5]',
                     'self.actor_net_param[6]']
        data_list2 = ['self.network_weigt_0', 'self.network_weigt_1',
                      'self.network_weigt_2', 'self.network_weigt_3',
                      'self.network_weigt_4', 'self.network_weigt_5',
                      'self.network_weigt_6']
        tmp_list = ['nw0', 'nw1', 'nw2', 'nw3', 'nw4', 'nw5', 'nw6']

        for i, _ in enumerate(data_list):
            data = data_list[i]
            data2 = data_list2[i]
            if i > 0:
                dep_data = tmp_list[i-1]
                depend_node = ast.Assign(targets=[ast.Name(id=tmp_list[i], ctx=ast.Store())],\
                                         value=ast.Call(func=ast.Name(id='self.depend',\
                                         ctx=ast.Load()), args=[ast.Name(id=data2), ast.Name(id=dep_data)],\
                                         keywords=[]))
                op_nodes.append(depend_node)

            assign_node = ast.Expr(value=ast.Call(func=ast.Name(id='self.assign', ctx=ast.Load()),\
                                  args=[ast.Name(id=data2), ast.Name(id=data)], keywords=[]))
            op_nodes.append(assign_node)

            if i == 0:
                send_data = data_list2[i]
            else:
                send_data = tmp_list[i]
            allgather_node = ast.Assign(targets=[ast.Name(id=tmp_list[i], ctx=ast.Store())],\
                                                 value=ast.Call(func=ast.Name(id='self.allgather',\
                                                 ctx=ast.Load()), args=[ast.Name(id=send_data)], keywords=[]))
            op_nodes.append(allgather_node)
    return op_nodes


def _build_allgather_receive(f_type):
    '''build allgather on receiver side'''
    op_nodes = []
    if f_type == 'Learner':
        data_list = ['self.state_list', 'self.reward_list', 'self.action_list',
                     'self.next_state_list', 'self.miu_list', 'self.sigma_list']
        tmp_data_list = ['state_list', 'reward_list', 'action_list',
                         'next_state_list', 'miu_list', 'sigma_list']
        pos_list = [(1, 0, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0)]
        shape_list = [('self.num_actor', 'self.num_collect_env', 'self.duration', 17),
                      ('self.num_actor', 'self.num_collect_env', 'self.duration', 1),
                      ('self.num_actor', 'self.num_collect_env', 'self.duration', 6),
                      ('self.num_actor', 'self.num_collect_env', 'self.duration', 17),
                      ('self.num_actor', 'self.num_collect_env', 'self.duration', 6),
                      ('self.num_actor', 'self.num_collect_env', 'self.duration', 6),
                     ]

        for i, _ in enumerate(data_list):
            data = data_list[i]
            tmp_data = tmp_data_list[i]


            allgather_node = ast.Assign(targets=[ast.Name(id=tmp_data, ctx=ast.Store())],\
                                        value=ast.Call(func=ast.Name(id='self.allgather',\
                                        ctx=ast.Load()), args=[ast.Name(id=data)], keywords=[]))
            op_nodes.append(allgather_node)

            if i < len(data_list)-1:
                dep_data = tmp_data_list[i]
                depend_node = ast.Assign(targets=[ast.Name(id=tmp_data_list[i+1], ctx=ast.Store())],\
                                         value=ast.Call(func=ast.Name(id='self.depend',\
                                         ctx=ast.Load()), args=[ast.Name(id=data_list[i+1]),\
                                         ast.Name(id=dep_data)], keywords=[]))
                op_nodes.append(depend_node)

            pos_tuple = _create_tuple(pos_list[i])
            shape_tuple = _create_tuple(shape_list[i])
            slice_node = ast.Assign(targets=[ast.Name(id=tmp_data, ctx=ast.Store())],\
                                    value=ast.Call(func=ast.Name(id='self.slice',\
                                    ctx=ast.Load()), args=[ast.Name(id=tmp_data),\
                                    pos_tuple, shape_tuple], keywords=[]))
            op_nodes.append(slice_node)

    if f_type == 'Actor':
        data_list = ['self.actor_net_param[0]', 'self.actor_net_param[1]',
                     'self.actor_net_param[2]', 'self.actor_net_param[3]',
                     'self.actor_net_param[4]', 'self.actor_net_param[5]',
                     'self.actor_net_param[6]']
        data_list2 = ['self.network_weigt_0', 'self.network_weigt_1',
                      'self.network_weigt_2', 'self.network_weigt_3',
                      'self.network_weigt_4', 'self.network_weigt_5',
                      'self.network_weigt_6']
        tmp_list = ['nw0', 'nw1', 'nw2', 'nw3', 'nw4', 'nw5', 'nw6']
        pos_list = [(0,), (0, 0), (0,), (0, 0), (0,), (0, 0), (0,)]
        shape_list = [(6,), (200, 17), (200,), (100, 200), (100,), (6, 100), (6,)]

        for i, _ in enumerate(data_list):
            data = data_list[i]
            pos_tuple = _create_tuple(pos_list[i])
            shape_tuple = _create_tuple(shape_list[i])

            if i == 0:
                send_data = data
            else:
                send_data = tmp_list[i]
            allgather_node = ast.Assign(targets=[ast.Name(id=tmp_list[i], ctx=ast.Store())],\
                                        value=ast.Call(func=ast.Name(id='self.allgather',\
                                        ctx=ast.Load()), args=[ast.Name(id=send_data)], keywords=[]))
            op_nodes.append(allgather_node)

            if i < len(data_list)-1:
                data2 = data_list2[i+1]
                dep_data = tmp_list[i]
                depend_node = ast.Assign(targets=[ast.Name(id=tmp_list[i+1], ctx=ast.Store())],\
                                         value=ast.Call(func=ast.Name(id='self.depend',\
                                         ctx=ast.Load()), args=[ast.Name(id=data2),\
                                         ast.Name(id=dep_data)], keywords=[]))
                op_nodes.append(depend_node)

            slice_node = ast.Assign(targets=[ast.Name(id=tmp_list[i], ctx=ast.Store())],\
                                        value=ast.Call(func=ast.Name(id='self.slice',\
                                        ctx=ast.Load()), args=[ast.Name(id=tmp_list[i]),\
                                    pos_tuple, shape_tuple], keywords=[]))
            op_nodes.append(slice_node)
            assign_node = ast.Expr(value=ast.Call(func=ast.Name(id='self.assign',\
                                   ctx=ast.Load()), args=[ast.Name(id=data),\
                                   ast.Name(id=tmp_list[i])], keywords=[]))
            op_nodes.append(assign_node)

    return op_nodes


# pylint: disable=R1710
def _parse_func_name(d, c):
    '''get function name from ast.Call'''
    def _parse_chain(d, c, p):
        if isinstance(d, ast.Name):
            return [d.id]+p
        if isinstance(d, ast.Call):
            for i in d.args:
                _parse_func_name(i, c)
            return _parse_chain(d.func, c, p)
        if isinstance(d, ast.Attribute):
            return _parse_chain(d.value, c, [d.attr]+p)
    if isinstance(d, (ast.Call, ast.Attribute)):
        p = []
        c.append('.'.join(_parse_chain(d, c, p)))
    else:
        for i in getattr(d, '_fields', []):
            t = getattr(d, i)
            if isinstance(t, list):
                for j in t:
                    _parse_func_name(j, c)
            else:
                _parse_func_name(t, c)


def _is_learner(ast_source, f):
    '''check fragment type'''
    flag = f
    if isinstance(ast_source, ast.Assign):
        for nn in ast.iter_child_nodes(ast_source):
            if isinstance(nn, ast.Call):
                names = []
                _parse_func_name(nn, names)
                for name in names:
                    if name == 'self.msrl.agent_learn':
                        flag = True
    return flag


def _select_statement(ast_src, f_type, statement_list):
    '''iterate the source '''
    flag = False
    for j in ast.iter_child_nodes(ast_src):
        flag = _is_learner(j, flag)
        if not flag and f_type == 'Actor':
            if not isinstance(j, ast.Return) and \
               not isinstance(j, ast.Name) and \
               not isinstance(j, ast.arguments):
                statement_list[f_type].insert(0, j)
        elif flag and f_type == 'Learner':
            if not isinstance(j, ast.Return) and \
               not isinstance(j, ast.Name) and \
               not isinstance(j, ast.arguments):
                statement_list[f_type].insert(0, j)
    return statement_list


def _build_statement_list(f_type, ast_source, position, ast_target, statement_list):
    'build statement list for to insert'
    source_name = list(position.keys())[0]
    for n in ast.iter_child_nodes(ast_source):
        if isinstance(n, ast.ClassDef) and source_name in n.name:
            for i in ast.iter_child_nodes(n):
                if isinstance(i, ast.FunctionDef) and i.name == '__init__':
                    _copy_constructor(ast_target, i, f_type)
                if isinstance(i, ast.FunctionDef) and i.name == position[source_name]:
                    _select_statement(i, f_type, statement_list)
    return statement_list


def _insert_statement(f_type, ast_target, statement_list, parameters):
    '''insert statement'''
    for j in ast.iter_child_nodes(ast_target):
        if isinstance(j, ast.Pass):
            ast_target.body.pop(0)
            if 'Send' in parameters[f_type]['operations']:
                statements = _build_sender(f_type)
                for statement in statements:
                    ast_target.body.insert(0, statement)
            if 'AllGather' in parameters[f_type]['operations']:
                statements = _build_allgather_send(f_type)
                for idx in range(len(statements)-1, -1, -1):
                    statement = statements[idx]
                    ast_target.body.insert(0, statement)
            for idx in range(len(statement_list[f_type])):
                statement = statement_list[f_type][idx]
                ast_target.body.insert(0, statement)
            if 'Receive' in parameters[f_type]['operations']:
                statements = _build_receiver(f_type)
                for statement in statements:
                    ast_target.body.insert(0, statement)
            if 'AllGather' in parameters[f_type]['operations']:
                statements = _build_allgather_receive(f_type)
                for idx in range(len(statements)-1, -1, -1):
                    statement = statements[idx]
                    ast_target.body.insert(0, statement)


def _insert_to_target(f_type, ast_target, statement_list, parameters):
    '''insert statements into target AST'''
    for n in ast.iter_child_nodes(ast_target):
        if isinstance(n, ast.ClassDef) and n.name == f_type:
            for i in ast.iter_child_nodes(n):
                if isinstance(i, ast.FunctionDef) and i.name == 'kernel':
                    _insert_statement(f_type, i, statement_list, parameters)
    ast.fix_missing_locations(ast_target)


def _insert_code(ast_source, ast_target, fragment_type, position, parameters):
    '''insert code from source algorithlm'''
    statement_list = {}
    for f_type in fragment_type:
        statement_list[f_type] = []
    for f_type in fragment_type:
        _build_statement_list(f_type, ast_source, position, ast_target, statement_list)
        _insert_to_target(f_type, ast_target, statement_list, parameters)

    return ast_target


def _generate_ast(algorithm):
    with open(algorithm, 'r', encoding='utf-8') as fsource:
        source = fsource.read()
    ast_source = ast.parse(source, type_comments=True)
    return ast_source


def _get_fragment_types(parameter_list):
    fragment_types = list(parameter_list.keys())
    return fragment_types


def generate_fragment(algorithm, parameter_list, template, algorithm_config, position, policy):
    '''fragment generation'''
    path = os.path.dirname(os.path.abspath(template))
    fragment_type = _get_fragment_types(parameter_list)
    ast_target = _transfor_class_name(template, fragment_type)
    ast_source = _generate_ast(algorithm)

    ast_target = _insert_code(ast_source, ast_target, fragment_type, position, parameter_list)

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    modes = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open("Fragments.py", flags, modes), 'w') as f:
        f.write(ast.unparse(ast_target))

    sys.path.insert(0, path)
    fragment_module = importlib.import_module("Fragments")
    fragment_list = []

    actor = getattr(fragment_module, 'Actor')
    learner = getattr(fragment_module, 'Learner')
    if list(policy.topology.keys())[0] == 'Actor':
        for _ in range(algorithm_config['actor']['number']):
            fragment_list.append(actor)
        fragment_list.append(learner)
    else:
        fragment_list.append(learner)
        for _ in range(algorithm_config['actor']['number']):
            fragment_list.append(actor)

    return fragment_list
