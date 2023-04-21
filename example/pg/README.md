# Policy Grandient Algorithm (PG)

## Related Paper

1. Williams R J. [Simple statistical gradient-following algorithms for connectionist reinforcement learning](https://link.springer.com/content/pdf/10.1007/BF00992696.pdf)

## Game that this algorithm used

PG use  an open source reinforcement learning environment library called  [Gym](https://github.com/openai/gym) which is developed by OpenAI.

The game solved in PG from Gym is [**CartPole-v0**](https://www.gymlibrary.dev/environments/classic_control/cart_pole/). "A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over."[1](https://www.gymlibrary.dev/environments/classic_control/cart_pole/)

## How to run PG

Before running PG, you should first install [MindSpore](https://www.mindspore.cn/install) and MindSpore-Reinforcement. Besides, you should also install following dependencies. Please follow the instruction on the official website.

- MindSpore >= 2.0.0

- numpy >= 1.17.0
- matplotlib >=3.1.3
- [gym](https://github.com/openai/gym) >= 0.18.3

After installation, you can directly use the following command to run the PG algorithm.

For comprehensive performance considerations on CPU, it is recommended to set `OMP_NUM_THREADS` and configure a unified configuration to 1/4 of the number of physical cores, such as export `OMP_NUM_THREADS=32`(edit in `run_standalone_train.sh`).

### Train

```shell
> cd example/pg/scripts
> bash run_standalone_train.sh [EPISODE] [DEVICE_TARGET] [PRECISION_MODE]
```

You will obtain outputs which is similar with the things below in `example/pg/scripts/pg_train_log.txt`.

```shell
Episode 520 has 200 steps, cost time: 158.550 ms, per step time: 0.793 ms
-----------------------------------------
Evaluate for episode 520 total rewards is 200.000
-----------------------------------------
Episode 530 has 200 steps, cost time: 156.738 ms, per step time: 0.784 ms
-----------------------------------------
Evaluate for episode 530 total rewards is 200.000
-----------------------------------------
Episode 540 has 200 steps, cost time: 161.918 ms, per step time: 0.810 ms
-----------------------------------------
Evaluate for episode 540 total rewards is 200.000
-----------------------------------------
Episode 550 has 200 steps, cost time: 161.121 ms, per step time: 0.806 ms
-----------------------------------------
Evaluate for episode 550 total rewards is 200.000
-----------------------------------------
Episode 560 has 200 steps, cost time: 159.035 ms, per step time: 0.795 ms
-----------------------------------------
Evaluate for episode 560 total rewards is 200.000
-----------------------------------------
Episode 570 has 200 steps, cost time: 134.095 ms, per step time: 0.670 ms
-----------------------------------------
Evaluate for episode 570 total rewards is 200.000
-----------------------------------------
Episode 580 has 200 steps, cost time: 166.513 ms, per step time: 0.833 ms
-----------------------------------------
Evaluate for episode 580 total rewards is 200.000
-----------------------------------------
Episode 590 has 200 steps, cost time: 131.775 ms, per step time: 0.659 ms
-----------------------------------------
Evaluate for episode 590 total rewards is 200.000
-----------------------------------------
```

### Eval

```shell
> cd example/pg/scripts
> bash run_standalone_eval.sh [CKPT_FILE_PATH] [DEVICE_TARGET](optional)
```

You will obtain outputs which is similar with the things below in `example/pg/scripts/pg_eval_log.txt`.

```shell
-----------------------------------------
Evaluate result is 200.000, checkpoint file in /path/ckpt/actor_net/actor_net_550.ckpt
-----------------------------------------
```

## Supported Platform

PG algorithm supports GPU, CPU and Ascend platform.
