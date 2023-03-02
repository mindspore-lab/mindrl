# Multi-agent Deep Deterministic Policy Gradient (MADDPG)

## Related Paper

1. [1] Lowe R, Wu Y, Tamar A, et al. [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/pdf/1706.02275v1.pdf)

## Game that this algorithm used

MADDPG uses  a Python library for conducting research in multi-agent reinforcement learning called  [PettingZoo[MPE]](https://pettingzoo.farama.org/).

In MADDPG run the game called ([**Simple-spread**](https://pettingzoo.farama.org/environments/mpe/simple_spread/)) in MPE。This environment has N agents, N landmarks (default N=3). At a high level, agents must learn to cover all the landmarks while avoiding collisions.

<img src="../../docs/images/mpe_simple_spread.gif" alt="mpe_simple_spread" style="zoom: 67%;" />

## How to run MADDPG

Before running MADDPG, you should first install [MindSpore](https://www.mindspore.cn/install) and MindSpore-Reinforcement. Besides, you should also install following dependencies. Please follow the instruction on the official website.

- mindspore >= 2.0.0
- mindspore-rl >= 0.6.0
- numpy >= 1.17.0
- matplotlib >=3.1.3
- [gym](https://github.com/openai/gym) >= 0.18.3
- [pettingzoo[mpe]](https://pettingzoo.farama.org/environments/mpe)  == 1.17.0

After installation, you can directly use the following command to run the MADDPG algorithm.

For comprehensive performance considerations on CPU, it is recommended to set `OMP_NUM_THREADS` and configure a unified configuration to 1/4 of the number of physical cores, such as export `OMP_NUM_THREADS=32`(edit in `run_standalone_train.sh`).

### Train & Eval

```shell
> cd example/maddpg/scripts
> bash run_standalone_train.sh [EPISODE] [DEVICE_TARGET] [PRECISION]
```

You will obtain outputs which is similar with the things below in `example/maddpg/scripts/maddpg_train_log.txt`.

```shell
-----------------------------------------
In episode 0, mean episode reward is -187.1343514245 , cost 34.322 s.
-----------------------------------------
In episode 1000, mean episode reward is -157.09264173 , cost 64.231 s.
-----------------------------------------
In episode 2000, mean episode reward is -109.13431034 , cost 66.023 s.
-----------------------------------------
In episode 3000, mean episode reward is -110.139262323 , cost 65.432 s.
-----------------------------------------
In episode 4000, mean episode reward is -109.901873233 , cost 64.521 s.
-----------------------------------------
In episode 5000, mean episode reward is -107.236288131 , cost 64.524 s.
```

## Supported Platform

MADDPG algorithm supports GPU, CPU and Ascend platform
