# Multi-Agent PPO (MAPPO)

## 相关论文

1. Yu et al., 2021 ["The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games"](https://arxiv.org/abs/2103.01955)

## 使用的游戏

在MAPPO中，我们使用了MAPPO原作者修改后的[Multi-Agent Particle Environment(MPE)](https://github.com/marlbenchmark/on-policy/tree/main/onpolicy/envs/mpe)中的`simple spread`环境，他在接口和具体算法逻辑上与openai版本的[MPE](https://github.com/openai/multiagent-particle-envs)有一定差异。MPE是一个简单的多智能体粒子环境，具有连续的观测和离散的动作空间，以及一些基本的物理属性模拟。

<img src="../../docs/images/mpe_simple_spread.gif" alt="mpe_simple_spread" style="zoom: 67%;" />

## 如何运行MAPPO

需要安装依赖：

```shell
pip install seaborn
```

在安装完以上依赖后，可以通过以下命令运行MAPPO：

```python
python train.py
```

## 支持平台

MAPPO算法支持GPU。
