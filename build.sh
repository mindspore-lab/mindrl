#!/bin/bash
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

export ENABLE_GPU="off"
export DEBUG_MODE="off"
while getopts 'e:d:e' opt
do
    OPTARG=$(echo ${OPTARG} | tr '[A-Z]' '[a-z]')
    case "${opt}" in
    d)
        DEBUG_MODE="on" ;;
    e)
        export DEVICE=$OPTARG ;;
    *)
        echo "Unknown opt ${opt}!"
        echo "Usage:"
        echo "bash build.sh [-d on|off] [-e gpu|cpu]"
        echo ""
        echo "Options:"
        echo "    -d Debug mode"
        echo "    -e Use cpu or gpu"
        exit 1
    esac
done
if [[ "X$DEVICE" == "Xgpu" ]]; then
    export ENABLE_GPU="on"
fi

if [[ "X$ENABLE_GPU" = "Xon" ]]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_GPU=ON"
fi

echo "---------------- Reinforcement: build start ----------------"
BASEPATH=$(cd "$(dirname $0)"; pwd)
BUILD_PATH="${BASEPATH}/build"
CMAKE_ARGS="${CMAKE_ARGS} -DCMAKE_INSTALL_PREFIX=${BUILD_PATH} -DDEBUG_MODE=$DEBUG_MODE"
mkdir ${BUILD_PATH}
cd ${BUILD_PATH}
cmake ${CMAKE_ARGS} ${BASEPATH}
make && make install

if [ $? -ne "0" ]; then
  echo "Cmake failed, please check Error"
  exit 1
fi

cd ${BASEPATH}
python3 setup.py bdist_wheel -d ${BASEPATH}/output

if [ ! -d "${BASEPATH}/output" ]; then
    echo "The directory ${BASEPATH}/output dose not exist."
    exit 1
fi

cd ${BASEPATH}/output || exit
for package in mindspore_rl*whl
do
    [[ -e "${package}" ]] || break
    sha256sum ${package} > ${package}.sha256
done
cd ${BASEPATH} || exit
echo "---------------- Reinforcement: build end   ----------------"
