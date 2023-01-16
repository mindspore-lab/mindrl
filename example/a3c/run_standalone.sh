#!/bin/bash
Usage(){
    echo "Failed!"
    echo "Usage: bash run_standalone.sh [worker_nums]"
    echo "In single node, actor_nums equals to worker_nums - 1."
}

if [ $# != 1 ];then
    Usage
    exit 1
fi

bash src/start_standalone.sh train.py $1
