# Copyright 2022-2023 Huawei Technologies Co., Ltd
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
Utils.
"""
import os
from importlib import import_module
import yaml


def _update_dict(dest, src) -> None:
    """update config dict"""
    if src is not None:
        for key in src:
            if key in dest.keys() and isinstance(dest.get(key), dict):
                if isinstance(src.get(key), dict):
                    for v in src.get(key):
                        if isinstance(src.get(key).get(v), dict) and v in dest.get(key) and \
                                isinstance(dest.get(key).get(v), dict):
                            _update_dict(dest[key], src[key])
                        elif isinstance(dest.get(key).get(v), dict):
                            dest[key][v].update(src.get(key).get(v))
                        else:
                            dest[key][v] = src.get(key).get(v)
                else:
                    dest[key].update(src[key])
            else:
                dest[key] = src[key]


def update_config(config, env_yaml, algo_yaml) -> None:
    r'''
    Update the config by the provided yamls.

    Args:
        config (dict): the config to be update.
        env_yaml (str): the environment yaml file.
        algo_yaml (str): the algorithm yaml file.
    '''
    if env_yaml:
        if os.path.exists(env_yaml):
            with open(env_yaml) as f:
                data = yaml.safe_load(f)
                config.collect_env_params['name'] = data.get('env')
                config.eval_env_params['name'] = data.get('env')
                _update_dict(config.collect_env_params, data.get('collect_env_params'))
                _update_dict(config.eval_env_params, data.get('eval_env_params'))
                if data.get('env_class') and data.get('env_type'):
                    try:
                        env_class = import_module(data.get('env_class'))
                        env_type = getattr(env_class, data.get('env_type'))
                        config.algorithm_config['collect_environment']['type'] = env_type
                        config.algorithm_config['eval_environment']['type'] = env_type
                    except:
                        raise ValueError(f"Import {data.get('env_class')} failed")
        else:
            print(f"File {env_yaml} is not exists.")
            return
    if algo_yaml:
        if os.path.exists(algo_yaml):
            with open(algo_yaml) as f:
                data = yaml.safe_load(f)
                if data.get('algorithm_config'):
                    _update_dict(config.algorithm_config, data.get('algorithm_config'))
                if data.get('policy_params'):
                    _update_dict(config.policy_params, data.get('policy_params'))
                if data.get('trainer_params'):
                    _update_dict(config.trainer_params, data.get('trainer_params'))
                if data.get('learner_params'):
                    _update_dict(config.learner_params, data.get('learner_params'))
                    if data.get('learner_class') and data.get('learner_type'):
                        try:
                            learner_class = import_module(data.get('learner_class'))
                            learner_type = getattr(learner_class, data.get('learner_type'))
                            config.algorithm_config['learner']['type'] = learner_type
                        except:
                            raise ValueError(f"Import {data.get('learner_class')} failed")
        else:
            print(f"File {algo_yaml} is not exiddsts.")
            return
