# Categorical 51-atom Agent Algorithm (C51)

## 相关论文

1. Marc G. Bellemare, Will Dabney, Rémi Munos, [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887)

## 使用的游戏

C51使用了OpenAI开发的一个强化学习环境库叫做[Gym](https://github.com/openai/gym)，来作为算法的游戏环境。相比于传统 DQN 算法希望学习的 Q 是一个数值，在值分布强化学习系列算法中，目标则由数值变为一个分布。这种改变可以学到除了数值以外的更多信息，即完整的分布信息。

在C51中，解决了倒立摆（[**CartPole-v0**](https://www.gymlibrary.dev/environments/classic_control/cart_pole/)）游戏。“一根杆子通过一个关节连接在一个小车上，这个小车可以沿着没有摩擦的轨道上左右移动。系统会对这个小车施加+1或者-1的力。这个游戏的目标是防止这根杆子从车上倒下。“

<img src="../../docs/images/cartpole.gif" alt="cartpole" style="zoom: 67%;" />

## 如何运行C51

在运行C51前，首先需要安装[MindSpore](https://www.mindspore.cn/install)和MindSpore-Reinforcement。除此之外，还需要安装以下依赖。请根据官网的教程安装。

- MindSpore >= 1.9.0
- Reinforcement >= 0.6
- numpy >= 1.17.0
- [gym](https://github.com/openai/gym) <= 0.21

安装成功之后，可以直接通过输入如下指令来运行C51。

### 训练

```shell
> cd example/c51/scripts
> bash run_standalone_train.sh [CKPT_PATH] [DEVICE_TARGET](可选)
```

你会在`example/c51/scripts/c51_train_log.txt`中获得和下面内容相似的输出

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

### 推理

```shell
> cd example/c51/scripts
> bash run_standalone_eval.sh [CKPT_PATH] [DEVICE_TARGET](可选)
```

你会在`example/c51/scripts/c51_eval_log.txt`中获得和下面内容相似的输出

```shell
Load file  /ckpt/policy_net/policy_net_300.ckpt
-----------------------------------------
Evaluate result is 200.000, checkpoint file in /ckpt/policy_net/policy_net_300.ckpt
-----------------------------------------
eval end
```

## 支持平台

C51算法支持CPU。
