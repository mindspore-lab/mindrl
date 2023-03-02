# Multi-agent Deep Deterministic Policy Gradient (MADDPG)

## 相关论文

1. [1] Lowe R, Wu Y, Tamar A, et al. [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/pdf/1706.02275v1.pdf)


## 使用的游戏

MADDPG算法使用了强化学习多智能体的环境库[PettingZoo[MPE]](https://pettingzoo.farama.org/)，来作为算法的游戏环境。

在MADDPG算法中，解决了多粒子([**Simple-spread**](https://pettingzoo.farama.org/environments/mpe/simple_spread/))游戏。该环境有N（默认N=3）个智能体，每个智能体需要学会覆盖目标，但同时避免相互碰撞。

<img src="../../docs/images/mpe_simple_spread.gif" alt="mpe_simple_spread" style="zoom: 67%;" />

## 如何运行MADDPG

在运行MADDPG前，首先需要安装[MindSpore](https://www.mindspore.cn/install) 和MindSpore-Reinforcement。除此之外，还需要安装以下依赖。请根据官网的教程安装。

- mindspore >= 2.0.0
- mindspore-rl >= 0.6.0
- numpy >= 1.17.0
- matplotlib >=3.1.3
- [gym](https://github.com/openai/gym) >= 0.18.3
- [pettingzoo[mpe]](https://pettingzoo.farama.org/environments/mpe/) == 1.17.0

安装成功之后，可以直接通过输入如下指令来运行MADDPG。

在CPU综合性能上的考虑，建议统一配置`OMP_NUM_THREADS` 的值为物理核数的1/4，比如`export OMP_NUM_THREADS=32`(需在`run_standalone_train.sh`中修改)。

### 训练&推理

```shell
> cd example/maddpg/scripts
> bash run_standalone_train.sh [EPISODE] [DEVICE_TARGET] [PRECISION]
```

你会在`example/maddpg/scripts/maddpg_train_log.txt`中获得和下面内容相似的输出

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

## 支持平台

MADDPG算法支持GPU，CPU和Ascend。
