# Copyright 2021 Huawei Technologies Co., Ltd
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
Components for MindSpore Reinforcement Learning Framework.
"""

import time

from mindspore_rl.core import MSRL, Session, UniformReplayBuffer
from mindspore_rl.distribution import fragment_generation
from mindspore_rl.version import __version__

__all__ = ["MSRL", "Session", "UniformReplayBuffer", "fragment_generation"]
__all__.extend(__version__)


def _mindspore_version_check():
    """
    Do the MindSpore version check for MindSpore Reinforcement. If the
    MindSpore can not be imported, it will raise ImportError. If its
    version is not compatibale with current MindSpore Reinforcement verision,
    it will print a warning.

    Raise:
        ImportError: If the MindSpore can not be imported.
    """

    try:
        import mindspore as ms
        from mindspore import log as logger
    except (ImportError, ModuleNotFoundError):
        print(
            "Can not find MindSpore in current environment. Please install "
            "MindSpore before using MindSpore Reinforcement, by following "
            "the instruction at https://www.mindspore.cn/install"
        )
        raise

    ms_msrl_version_match = {
        "0.1": "1.5",
        "0.2": "1.6",
        "0.3": "1.7",
        "0.5": "1.8",
        "0.6": "2.0",
        "0.7": "2.1",
        "0.8": "2.2",
    }

    ms_version = ms.__version__[:3]
    required_mindspore_verision = ms_msrl_version_match.get(__version__[:3])

    if ms_version != required_mindspore_verision:
        logger.warning(
            "Current version of MindSpore is not compatible with MindSpore Reinforcement. "
            "Some functions might not work or even raise error. Please install MindSpore "
            "version == {}.0 For more details about dependency setting, please check "
            "the instructions at MindSpore official website https://www.mindspore.cn/install "
            "or check the README.md at "
            "https://gitee.com/mindspore/reinforcement".format(
                required_mindspore_verision
            )
        )
        warning_countdown = 3
        for i in range(warning_countdown, 0, -1):
            logger.warning(f"Please pay attention to the above warning, countdown: {i}")
            time.sleep(1)


_mindspore_version_check()
