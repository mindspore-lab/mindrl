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

export MS_SCHED_HOST=127.0.0.1
export MS_SCHED_PORT=8088
if [ $# == 0 ]; then
  MS_WORKER_NUM=2
  EPISODE=500
elif [ $# == 1 ]; then
  MS_WORKER_NUM=$1
  EPISODE=500
elif [ $# == 2 ]; then
  MS_WORKER_NUM=$1
  EPISODE=$2
else
  echo "Usage: bash run_distribute.sh [WORKER_NUM] [EPISODE](optional)."
  echo "Example: bash run_distribute.sh"
fi
export MS_WORKER_NUM=${MS_WORKER_NUM}
# Launch 1 scheduler
export MS_ROLE=MS_SCHED
python train.py --episode ${EPISODE} --enable_distribute True --worker_num ${MS_WORKER_NUM} > scheduler.txt 2>&1 &
sched_pid=${!}
echo "Scheduler $sched_pid start sucess!"

# Launch MS_WORKER_NUM workers, including learner and actors
export MS_ROLE=MS_WORKER
for ((i=0;i<${MS_WORKER_NUM};i++));
do
    python train.py --episode ${EPISODE} --enable_distribute True --worker_num ${MS_WORKER_NUM} > worker_$i.txt 2>&1 &
    echo "Worker ${i} start success with pid ${!}"
done
exit 0
