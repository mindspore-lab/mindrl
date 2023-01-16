#!/bin/bash
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

script_self=$(readlink -f "$0")
self_path=$(dirname "${script_self}")
if [ $# == 0 ]; then
  EPISODE=650
  DEVICE="Auto"
elif [ $# == 1 ]; then
  EPISODE=$1
  DEVICE="Auto"
elif [ $# == 2 ]; then
  EPISODE=$1
  DEVICE=$2
else
  echo "Usage: bash run_standalone_train.sh [EPISODE](optional) [DEVICE_TARGET](optional)."
  echo "Example: bash run_standalone_train.sh"
fi
export OMP_NUM_THREADS=10
python -s ${self_path}/../train.py --device_target=$DEVICE --episode=$EPISODE > td3_train_log.txt 2>&1 &
