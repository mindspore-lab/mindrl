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
slef_path=$(dirname "${script_self}")
if [ $# == 0 ]; then
  ONNX='./onnx/policy_net/policy_net_1000.onnx'
elif [ $# == 1 ]; then
  ONNX=$1
else
  echo "Usage: bash run_infer_onnx.sh [ONNX_PATH]."
  echo "Example: bash run_infer_onnx ./onnx/policy_net/policy_net_1000.onnx"
fi
export OMP_NUM_THREADS=10
python -s ${slef_path}/../infer_dqn_onnx.py --onnx_file=$ONNX > dqn_infer_onnx_log.txt 2>&1 &
