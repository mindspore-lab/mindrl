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
Generate fragment
"""
from .generate_code import GenerateFragment


def fragment_generation(algo_name, worker_num, policy, msrl, frag_file=None) -> list:
    """
    Function of generate fragments.
    algo_name: the name of the standard algorithm, used to find the trainer
    worker_num: it will determinate to the fragment num.
    policy: the distribution policy.
    msrl: used to init the distribution policy.
    frag_file: skip auto-generate, and create fragment list from frag_file.
    """
    policy = policy(msrl)
    gen_fragment = GenerateFragment(algo_name, policy, worker_num)
    fragments = gen_fragment.create_fragment(frag_file)
    return fragments
