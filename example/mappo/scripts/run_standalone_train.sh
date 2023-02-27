#!/bin/bash
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

script_self=$(readlink -f "$0")
self_path=$(dirname "${script_self}")
PRECISION="fp32"
if [ $# == 0 ]; then
  DEVICE="Auto"
  EPISODE=1000
elif [ $# == 1 ]; then
  EPISODE=$1
  DEVICE="Auto"
elif [ $# == 2 ]; then
  EPISODE=$1
  DEVICE=$2
elif [ $# == 3 ]; then
  EPISODE=$1
  DEVICE=$2
  PRECISION=$3
else
  echo "Usage: bash run_standalone_train.sh [CKPT_PATH](optional) [DEVICE_TARGET](optional) [PRECISION](optional)."
  echo "Example: bash run_standalone_train.sh"
fi

python -s ${self_path}/../train.py --episode $EPISODE --device_target=$DEVICE \
--precision_mode=$PRECISION > mappo_train_log.txt 2>&1 &
