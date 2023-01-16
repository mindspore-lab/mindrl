#!/bin/bash
Usage(){
    echo "Usage: Start a scheduler:"
    echo "bash run_distribute.sh [scheduler] [worker_num] [ip] [port]"
    echo "Start workers:"
    echo "bash run_distribute.sh [worker] [worker_num] [worker_num_in_this_node] [ip] [port]"
    echo "In multi nodes, worker_nums should be the total number of workers(all the nodes);"
    echo "worker_num_in_this_node is defined by users, it means how many workers in this node."
    echo "The sum of these worker_num_in_this_node in each node should equal to worker_numbers."
    echo "Attention: modify the MS_SCHED_HOST to the real ip in src/start_distribute.sh"
}

if [ $# == 4 ];then
    if [ $1 == scheduler ];then
        bash src/start_distribute.sh train.py $1 $2 $3 $4
    else
        Usage
        exit 1
    fi
elif [ $# == 5 ];then
    if [ $1 == worker ];then
        bash src/start_distribute.sh train.py $1 $2 $3 $4 $5
    else
        Usage
        exit 1
    fi
else
    Usage
    exit 1
fi


## Start a scheduler.
# bash run_distribute.sh scheduler 4 0.0.0.1 6379
## Start 2 workers in total 4 workers.
#bash run_distribute worker 4 2 0.0.0.1 6379
## Then run in the another node with 2 workers.
#bash run_distribute worker 4 2 0.0.0.1 6379
