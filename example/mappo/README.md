# Multi-Agent PPO (MAPPO)

## Related Paper

1. Yu et al., 2021 ["The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games"](https://arxiv.org/abs/2103.01955)

## Game that this algorithm used

In MAPPO, we use `simple spread` in [Multi-Agent Particle Environment(MPE)](https://github.com/marlbenchmark/on-policy/tree/main/onpolicy/envs/mpe) which is modified by MAPPO author. There are some difference between openai's [MPE](https://github.com/openai/multiagent-particle-envs) and MAPPO author's. A simple multi-agent particle world with a continuous observation and discrete action space, along with some basic simulated physics.

<img src="../../docs/images/mpe_simple_spread.gif" alt="mpe_simple_spread" style="zoom: 67%;" />

## How to run MAPPO

User needs to install the dependency:

```shell
pip install seaborn
```

After installation, user can run MAPPO by using following command:

```python
python train.py
```

## Supported Platform

MAPPO algorithm currently supports GPU.
