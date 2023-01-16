# Dreamer

## 相关论文

1. Hafner et al., 2019 ["Dream to Control: Learning Behaviors by Latent Imagination"](https://arxiv.org/abs/1912.01603)

## 使用的游戏

在Dreamer中，我们使用的DeepMind Control Suite(DMC)（<https://github.com/deepmind/dm_control>）中的Walker walk环境。这个游戏模拟人在环境中行走。系统操控的是双脚每个关节的力矩。

<!-- <img src="../../docs/images/mpe_simple_spread.gif" alt="mpe_simple_spread" style="zoom: 67%;" /> -->

## 如何运行Dreamer

需要安装依赖：

```shell
pip install dm_control
```

在安装完以上依赖后，可以通过以下命令运行Dreamer：

```python
python train.py
```

## 支持平台

Dreamer算法支持GPU。
