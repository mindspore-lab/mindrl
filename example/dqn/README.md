# Deep Q-Learning (DQN)

## Related Paper

1. Mnih, Volodymyr, et al. [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602.pdf)

2. Mnih, Volodymyr, *et al.* [Human-level control through deep reinforcement learning. *Nature* **518,** 529–533 (2015).](https://www.nature.com/articles/nature14236)

## Game that this algorithm used

DQN uses  an open source reinforcement learning environment library called  [Gym](https://github.com/openai/gym) which is developed by OpenAI.

The game solved in DQN is [**CartPole-v0**](https://gym.openai.com/envs/CartPole-v0/). "A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over."[1](https://gym.openai.com/envs/CartPole-v0/)

## How to run DQN

Before running DQN, you should first install [MindSpore](https://www.mindspore.cn/install) and MindSpore-Reinforcement. Besides, you should also install following dependencies. Please follow the instruction on the official website.

- MindSpore >= 1.6.0

- numpy >= 1.17.0
- matplotlib >=3.1.3
- [gym](https://github.com/openai/gym) >= 0.18.3

After installation, you can directly use the following command to run the DQN algorithm.

For comprehensive performance considerations on CPU, it is recommended to set `OMP_NUM_THREADS` and configure a unified configuration to 1/4 of the number of physical cores, such as export `OMP_NUM_THREADS=32`(edit in `run_standalone_train.sh`).

### Train

```shell
> cd example/dqn/scripts
> bash run_standalone_train.sh [EPISODE] [DEVICE_TARGET](optional)
```

You will obtain outputs which is similar with the things below in `dqn_train_log.txt`.

```shell
Episode 0： loss is 0.223, rewards is 10.0.
Episode 1： loss is 0.186, rewards is 38.0..
Episode 2： loss is 0.152, rewards is 23.0.
Episode 3： loss is 0.118, rewards is 9.0.
Episode 4： loss is 0.1, rewards is 10.0.
Episode 5： loss is 0.146, rewards is 12.0.
Episode 6： loss is 0.062, rewards is 10.0.
Episode 7： loss is 0.144, rewards is 10.0.
Episode 8： loss is 0.086, rewards is 9.0.
Episode 9： loss is 0.125, rewards is 9.0.
Episode 10： loss is 0.143, rewards is 9.0.
-----------------------------------------
Evaluate for episode 10 total rewards is 9.300
-----------------------------------------
```

### Eval

```shell
> cd example/dqn/scripts
> bash run_standalone_eval.sh [CKPT_FILE_PATH] [DEVICE_TARGET](optional)
```

You will obtain outputs which is similar with the things below in `dqn_eval_log.txt`.

```shell
Load file /path/ckpt/policy_net/policy_net_600.ckpt
-----------------------------------------
Evaluate result is 199.300, checkpoint file in /path/ckpt/policy_net/policy_net_600.ckpt
-----------------------------------------
```

## Supported Platform

DQN algorithm supports GPU, CPU and Ascend platform
