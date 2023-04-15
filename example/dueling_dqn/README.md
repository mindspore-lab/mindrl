# Dueling Deep Q-Learning (Dueling DQN)

## Related Paper

1. Hado van Hasselt, et al. [Deep Reinforcement Learning with Dueling Q-learning](https://arxiv.org/abs/1509.06461)

## Game that this algorithm used

Dueling DQN uses  an open source reinforcement learning environment library called  [Gym](https://github.com/openai/gym) which is developed by OpenAI.

The game solved in Dueling DQN is [**CartPole-v0**](https://gym.openai.com/envs/CartPole-v0/). "A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over."[1](https://gym.openai.com/envs/CartPole-v0/)

## How to run Dueling DQN

Before running Dueling DQN, you should first install [MindSpore](https://www.mindspore.cn/install) and MindRL. Besides, you should also install following dependencies. Please follow the instruction on the official website.

- MindSpore >= 1.6.0

- numpy >= 1.17.0
- matplotlib >=3.1.3
- [gym](https://github.com/openai/gym) >= 0.18.3

After installation, you can directly use the following command to run the Dueling DQN algorithm.

### Train

```shell
> cd example/dueling_dqn/scripts
> bash run_standalone_train.sh [EPISODE] [DEVICE_TARGET](optional)
```

You will obtain outputs which is similar with the things below in `ddqn_train_log.txt`.

```shell
Episode 1 has 10.0 steps, cost time: 18.480 ms, per step time: 1.848 ms
Episode 1: loss is 0.632, rewards is 10.0
Episode 2 has 9.0 steps, cost time: 19.179 ms, per step time: 2.131 ms
Episode 2: loss is 0.379, rewards is 9.0
Episode 3 has 10.0 steps, cost time: 20.021 ms, per step time: 2.002 ms
Episode 3: loss is 0.338, rewards is 10.0
Episode 4 has 8.0 steps, cost time: 16.123 ms, per step time: 2.015 ms
Episode 4: loss is 0.311, rewards is 8.0
Episode 5 has 10.0 steps, cost time: 18.964 ms, per step time: 1.896 ms
Episode 5: loss is 0.208, rewards is 10.0
Episode 6 has 12.0 steps, cost time: 23.792 ms, per step time: 1.983 ms
Episode 6: loss is 0.175, rewards is 12.0
Episode 7 has 11.0 steps, cost time: 21.279 ms, per step time: 1.934 ms
Episode 7: loss is 0.134, rewards is 11.0
Episode 8 has 10.0 steps, cost time: 19.681 ms, per step time: 1.968 ms
Episode 8: loss is 0.167, rewards is 10.0
Episode 9 has 13.0 steps, cost time: 25.184 ms, per step time: 1.937 ms
Episode 9: loss is 0.148, rewards is 13.0
Episode 10 has 11.0 steps, cost time: 19.831 ms, per step time: 1.803 ms
Episode 10: loss is 0.062, rewards is 11.0
-----------------------------------------
Evaluate for episode 10 total rewards is 10.500
-----------------------------------------
```

### Eval

```shell
> cd example/dueling_dqn/scripts
> bash run_standalone_eval.sh [CKPT_FILE_PATH] [DEVICE_TARGET](optional)
```

You will obtain outputs which is similar with the things below in `ddqn_eval_log.txt`.

```shell
Load file /path/ckpt/policy_net/policy_net_600.ckpt
-----------------------------------------
Evaluate result is 199.300, checkpoint file in /path/ckpt/policy_net/policy_net_600.ckpt
-----------------------------------------
```

## Supported Platform

Dueling DQN algorithm supports GPU, CPU and Ascend platform
