# Categorical 51-atom Agent Algorithm (C51)

## Related Paper

1. Marc G. Bellemare, Will Dabney, Rémi Munos, [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887)

## Game that this algorithm used

C51 use  an open source reinforcement learning environment library called [Gym](https://github.com/openai/gym) which is developed by OpenAI. Compared with the traditional DQN algorithm, the desired Q is a numerical value, in the series of value distribution reinforcement learning algorithms, the target is changed from a numerical value to a distribution. This change allows you to learn more than just a numerical value, but the complete value distribution.

The game solved in C51 from Gym is [**CartPole-v0**](https://www.gymlibrary.dev/environments/classic_control/cart_pole/). "A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over."

<img src="../../docs/images/cartpole.gif" alt="cartpole" style="zoom: 67%;" />

## How to run C51

Before running C51, you should first install [MindSpore](https://www.mindspore.cn/install) and MindSpore-Reinforcement. Besides, you should also install following dependencies. Please follow the instruction on the official website.

- MindSpore >= 1.9.0
- Reinforcement >= 0.6
- numpy >= 1.17.0
- [gym](https://github.com/openai/gym) <= 0.21

After installation, you can directly use the following command to run the C51 algorithm.

### Train

```shell
> cd example/c51/scripts
> bash run_standalone_train.sh [CKPT_PATH] [DEVICE_TARGET](optional)
```

You will obtain outputs which is similar with the things below in`example/c51/scripts/c51_train_log.txt`

```shell
Episode 301: loss is 0.111, rewards is 200.0
Episode 302: loss is 0.03, rewards is 200.0
Episode 303: loss is 0.114, rewards is 200.0
Episode 304: loss is 0.078, rewards is 200.0
Episode 305: loss is 0.016, rewards is 200.0
Episode 306: loss is 0.166, rewards is 191.0
Episode 307: loss is 0.155, rewards is 200.0
Episode 308: loss is 0.094, rewards is 200.0
Episode 309: loss is 0.111, rewards is 199.0
Episode 310: loss is 0.035, rewards is 200.0
-----------------------------------------
Evaluate for episode 310 total rewards is 199.600
-----------------------------------------
```

### Eval

```shell
> cd example/c51/scripts
> bash run_standalone_eval.sh [CKPT_PATH] [DEVICE_TARGET](optional)
```

You will obtain outputs which is similar with the things below in `example/c51/scripts/c51_eval_log.txt`.

```shell
Load file  /ckpt/policy_net/policy_net_300.ckpt
-----------------------------------------
Evaluate result is 200.000, checkpoint file in /ckpt/policy_net/policy_net_300.ckpt
-----------------------------------------
eval end
```

## Supported Platform

C51 algorithm supports CPU platform.
