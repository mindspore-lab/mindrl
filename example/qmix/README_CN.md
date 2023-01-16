# QMIX

## 相关论文

1. Rashid et al., 2018 ["QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning"](https://arxiv.org/pdf/1803.11485.pdf)

## 使用的游戏

在QMIX中，我们使用了[SMAC - StarCraft Multi-Agent Challenge](https://github.com/oxwhirl/smac)环境。SMAC是WhiRL的一个基于暴雪星际争霸2这个战略游戏开发的用于多智能体强化学习（MARL）在合作场景的环境。SMAC通过使用暴雪星际争霸2的机器学习API和DeepMind的PySC2提供了易用的界面方便智能体与星际争霸2的交互来获得环境的状态和合法的动作。不像PySC2，SMAC专注于去中心的细微操控场景，这种场景下游戏中的每个单位都会被一个独立的RL智能体操控。

## 如何运行QMIX

QMIX依赖SMAC环境，在运行QMIX之前需要安装SMAC。请参考SMAC的官方Github：[https://github.com/oxwhirl/smac](https://github.com/oxwhirl/smac)。

在安装完SMAC环境后，可以通过以下命令运行QMIX：

```python
python train.py --device_target [YOUR DEVICE]
```

## 支持平台

QMIX算法支持GPU和CPU。
