# Importance Weighted Actor-Learner Architecture (IMPALA)

## 相关论文

1. Espeholt L, Soyer H, Munos R, et al. [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](https://arxiv.org/abs/1802.01561)

IMPALA：Importance Weighted Actor-Learner Architectures，模型与A3C类似，主要区别是actor不再进行梯度计算，只进行环境数据的收集。收集得到的轨迹（Trajectory）传给learner进行学习，并从learner更新最新的参数。同时引入了V-trace方法，使得模型可以接受更多的延迟(policy-lag)，实现离线学习(off-policy)，拥有更大的吞吐量。

![IMPALA](../../docs/images/IMPALA_arch.png)

## 使用的游戏

使用了OpenAI开发的一个强化学习环境库叫做[Gym](https://github.com/openai/gym)，来作为算法的游戏环境。

使用IMPALA模型测试了倒立摆（[**CartPole-v0**](https://www.gymlibrary.dev/environments/classic_control/cart_pole/)）游戏。“一根杆子通过一个关节连接在一个小车上，这个小车可以沿着没有摩擦的轨道上左右移动。系统会对这个小车施加+1或者-1的力。这个游戏的目标是防止这根杆子从车上倒下。“[1](https://www.gymlibrary.dev/environments/classic_control/cart_pole/)

![A3C](../../docs/images/cartpole.gif)

## 如何运行IMPALA

在运行IMPALA前，首先需要安装[MindSpore](https://www.mindspore.cn/install)和MindSpore-Reinforcement。除此之外，还需要安装以下依赖。请根据官网的教程安装。

- MindSpore > 2.0.0
- Reinforcement > 0.6
- numpy >= 1.17.0
- [gym](https://github.com/openai/gym) >= 0.18.3

安装成功之后，可以直接通过输入如下指令来运行IMPALA模型。

### 训练

### 单机训练

以单机4卡(1 learner + 3 actors)为例：

```shell
> cd example/impala
> bash start_standalone.sh train.py 4
```

你会在`example/impala`下找到`worker_0.txt`到`worker_3.txt`四个文件，分别代表一个learner和3个actor的输出，输出和下面类似。

learner对应的输出：

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

actor对应的输出：
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

## 支持平台

IMPALA算法支持GPU。
