# Advantage Actor-Critic Algorithm (A2C)

## Related Paper

1. Konda, Vijay R., and John N. Tsitsiklis. "[Actor-critic algorithm](https://proceedings.neurips.cc/paper/1999/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)"

## Game that this algorithm used

A2C use  an open source reinforcement learning environment library called  [Gym](https://github.com/openai/gym) which is developed by OpenAI.

The game solved in A2C from Gym is [**CartPole-v0**](https://www.gymlibrary.dev/environments/classic_control/cart_pole/). "A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over."[1](https://www.gymlibrary.dev/environments/classic_control/cart_pole/)

## How to run A2C

Before running A2C, you should first install [MindSpore](https://www.mindspore.cn/install) and MindSpore-Reinforcement. Besides, you should also install following dependencies. Please follow the instruction on the official website.

- MindSpore >= 1.6.0

- numpy >= 1.17.0
- matplotlib >=3.1.3
- [gym](https://github.com/openai/gym) >= 0.18.3
- tqdm >= 4.46.0

After installation, you can directly use the following command to run the A2C algorithm.

For comprehensive performance considerations on CPU, it is recommended to set `OMP_NUM_THREADS` and configure a unified configuration to 1/4 of the number of physical cores, such as export `OMP_NUM_THREADS=32`(edit in `run_standalone_train_and_eval.sh`).

### Train

```shell
> cd example/a2c/scripts
> bash run_standalone_train_and_eval.sh [DEVICE_TARGET](optional)
```

You will obtain outputs which is similar with the things below in `a2c_log.txt`.

```shell
Solved at episode 353: average reward: 195.74.
Episode 353:  4%|██▏                                       | 353/10000 [08:43<03:38,  1.48s/it, episode_reward=200.0, loss=27.060312, running_reward=196]
training end
```

## Supported Platform

A2C algorithm supports GPU and CPU platform.
