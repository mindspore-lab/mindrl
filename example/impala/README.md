# Importance Weighted Actor-Learner Architecture (IMPALA)

[查看中文](./README_CN.md)

## Related Paper

1. Espeholt L, Soyer H, Munos R, et al. [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](https://arxiv.org/abs/1802.01561)

IMPALA: Importance Weighted Actor-Learner Architectures, whose model is similar to A3C. The main difference is that the actor no longer performs the gradient calculation, but only the environmental data collection. The collected trajectory is passed to the learner for learning, and the latest parameters are updated from the learner. The V-trace method is also introduced, which allows the model to accept more policy-lag, achieve off-line learning, and have greater throughput.

![IMPALA](../../docs/images/IMPALA_arch.png)

## Game that this algorithm used

IMPALA use  an open source reinforcement learning environment library called  [Gym](https://github.com/openai/gym) which is developed by OpenAI.

The game solved in IMPALA from Gym is [**CartPole-v0**](https://www.gymlibrary.dev/environments/classic_control/cart_pole/). "A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over."[1](https://www.gymlibrary.dev/environments/classic_control/cart_pole/)

![A3C](../../docs/images/cartpole.gif)

## How to run IMPALA

Before running IMPALA, you should first install [MindSpore](https://www.mindspore.cn/install) and MindSpore-Reinforcement. Besides, you should also install following dependencies. Please follow the instruction on the official website.

- MindSpore > 2.0.0
- Reinforcement > 0.6
- numpy >= 1.17.0
- [gym](https://github.com/openai/gym) >= 0.18.3

After installation, you can directly use the following command to run the IMPALA algorithm.

### Train

### Standalone Training

Take the single node with 4 GPUs (1 learner + 3 actors) as an example:

```shell
> cd example/impala
> bash start_standalone.sh train.py 4
```

You will find four txts `worker_0.txt` to `worker_3.txt` under `example/impala`, representing the output of one learner and three actors respectively, and the output is similar to the following.


learner:
```
Train from one actor, episode 698, loss 444.2373
Train from one actor, episode 698, loss 516.88477
Train from one actor, episode 698, loss 400.71414
Train from one actor, episode 699, loss 191.12558
Train from one actor, episode 699, loss 1321.3834
Train from one actor, episode 699, loss 564.8677
Train from one actor, episode 700, loss 660.90216
Train from one actor, episode 700, loss 557.4285
Train from one actor, episode 700, loss 435.9295
Train from one actor, episode 701, loss 478.4693
```

actor:
```
Evaluating in actor 1
evaluate in actor 1, avg_reward 127.0
Evaluating in actor 1
evaluate in actor 1, avg_reward 123.0
Evaluating in actor 1
evaluate in actor 1, avg_reward 178.0
Evaluating in actor 1
evaluate in actor 1, avg_reward 161.0
Evaluating in actor 1
evaluate in actor 1, avg_reward 200.0
Evaluating in actor 1
evaluate in actor 1, avg_reward 200.0
Evaluating in actor 1
evaluate in actor 1, avg_reward 200.0
```

## Supported Platform

IMPALA algorithm supports GPU platform.
