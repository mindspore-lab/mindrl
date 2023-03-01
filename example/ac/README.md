# Actor-Critic Algorithm (AC)

## Related Paper

1. Konda, Vijay R., and John N. Tsitsiklis. "[Actor-critic algorithm](https://proceedings.neurips.cc/paper/1999/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)"

## Game that this algorithm used

AC use  an open source reinforcement learning environment library called  [Gym](https://github.com/openai/gym) which is developed by OpenAI.

The game solved in AC from Gym is [**CartPole-v0**]https://www.gymlibrary.dev/environments/classic_control/cart_pole/). "A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over."[1](https://www.gymlibrary.dev/environments/classic_control/cart_pole/)

## How to run AC

Before running AC, you should first install [MindSpore](https://www.mindspore.cn/install) and MindSpore-Reinforcement. Besides, you should also install following dependencies. Please follow the instruction on the official website.

- MindSpore >= 1.6.0

- numpy >= 1.17.0
- matplotlib >=3.1.3
- [gym](https://github.com/openai/gym) >= 0.18.3

After installation, you can directly use the following command to run the AC algorithm.

For comprehensive performance considerations on CPU, it is recommended to set `OMP_NUM_THREADS` and configure a unified configuration to 1/4 of the number of physical cores, such as export `OMP_NUM_THREADS=32`(edit in `run_standalone_train.sh`).

### Train

```shell
> cd example/ac/scripts
> bash run_standalone_train.sh [EPISODE] [DEVICE_TARGET](optional)
```

You will obtain outputs which is similar with the things below in `log.txt`.

```shell
Episode 0, loss is 386.797, rewards is 20.0
Episode 1, loss is 386.477, rewards is 25.0
Episode 2, loss is 385.673, rewards is 11.0
Episode 3, loss is 386.896, rewards is 17.0
Episode 4, loss is 385.612, rewards is 28.0
Episode 5, loss is 386.764, rewards is 43.0
Episode 6, loss is 386.637, rewards is 32.0
Episode 7, loss is 388.327, rewards is 12.0
Episode 8, loss is 385.753, rewards is 39.0
Episode 9, loss is 386.731, rewards is 17.0
------------------------------------
Evaluate for episode 10 total rewards is 9.600
------------------------------------
```

### Eval

```shell
> cd example/ac/scripts
> bash run_standalone_eval.sh [CKPT_FILE_PATH] [DEVICE_TARGET](optional)
```

You will obtain outputs which is similar with the things below in `log.txt`.

```shell
-----------------------------------------
Evaluate result is 170.300, checkpoint file in /path/ckpt/actor_net/actor_net_950.ckpt
-----------------------------------------
```

## Supported Platform

AC algorithm supports GPU, CPU and Ascend platform.
