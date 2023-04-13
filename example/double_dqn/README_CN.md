# Double Deep Q-Learning (Double DQN)

## 相关论文

1. Hado van Hasselt, et al. [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)

## 使用的游戏

Double DQN算法使用了OpenAI开发的一个强化学习环境库[Gym](https://github.com/openai/gym) ，来作为算法的游戏环境。

在Double DQN算法中，解决了倒立摆([**CartPole-v0**](https://github.com/openai/gym/wiki/CartPole-v0)) 游戏。“一根杆子通过一个关节连接在一个小车上，这个小车可以沿着没有摩擦的轨道上左右移动。系统会对这个小车施加+1或者-1的力。这个游戏的目标是防止这根杆子从车上倒下。“[1](https://github.com/openai/gym/wiki/CartPole-v0)

## 如何运行Double DQN

在运行Double DQN前，首先需要安装[MindSpore](https://www.mindspore.cn/install) 和MindRL。除此之外，还需要安装以下依赖。请根据官网的教程安装。

- MindSpore >= 1.6.0

- numpy >= 1.17.0
- matplotlib >=3.1.3
- [gym](https://github.com/openai/gym) >= 0.18.3

安装成功之后，可以直接通过输入如下指令来运行Double DQN。


### 训练

```shell
> cd example/double_dqn/scripts
> bash run_standalone_train.sh [EPISODE] [DEVICE_TARGET](可选)
```

你会在`example/double_dqn/scripts/ddqn_train_log.txt`中获得和下面内容相似的输出

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

### 推理

```shell
> cd example/double_dqn/scripts
> bash run_standalone_eval.sh [CKPT_PATH] [DEVICE_TARGET](可选)
```

你会在`example/double_dqn/scripts/ddqn_eval_log.txt`中获得和下面内容相似的输出

```shell
Load file /path/ckpt/policy_net/policy_net_600.ckpt
-----------------------------------------
Evaluate result is 199.300, checkpoint file in /path/ckpt/policy_net/policy_net_600.ckpt
-----------------------------------------
```

## 支持平台

Double DQN算法支持GPU，CPU和Ascend。
